"""DiaFoot.AI — paper-grade aggregation for the dataset-composition study.

Turns per-image segmentation metrics into citable summaries: bootstrap
confidence intervals, the false-positive-rate on empty ground truth (the actual
cost of dropping negatives from training), paired significance between two
compositions on the *same* test images, and a provenance block that pins every
number to a split file, a checkpoint, a git commit, and a seed.

Everything here is pure/CPU and unit-testable; the training + inference live
elsewhere. Kept separate from ``src.evaluation.metrics`` (per-image metric
definitions) so the study's statistics have one auditable home.
"""

from __future__ import annotations

import hashlib
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

from src.evaluation.metrics import aggregate_metrics

# Class label used by DFUDataset for DFU (see src/data/torch_dataset.py).
DFU_LABEL = 2


def bootstrap_ci(
    values: Sequence[float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, float]:
    """Percentile bootstrap CI for the mean of ``values``.

    Returns ``{mean, ci_low, ci_high, n}``. An empty input yields NaNs with
    ``n == 0`` rather than raising, so callers can aggregate ragged slices.
    """
    v = np.asarray(list(values), dtype=float)
    if v.size == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    boot_means = v[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean": float(v.mean()), "ci_low": float(lo), "ci_high": float(hi), "n": int(v.size)}


def paired_delta_ci(
    a: Sequence[float],
    b: Sequence[float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, float]:
    """Bootstrap CI + two-sided p-value for the paired mean difference ``a - b``.

    ``a`` and ``b`` must be aligned per-image measurements from two models
    evaluated on the *same* test images in the same order. The p-value is the
    bootstrap proportion of resampled mean-differences on the far side of zero
    (two-sided), clamped to [0, 1].
    """
    a_arr = np.asarray(list(a), dtype=float)
    b_arr = np.asarray(list(b), dtype=float)
    if a_arr.shape != b_arr.shape:
        msg = f"paired arrays must have equal length, got {a_arr.shape} vs {b_arr.shape}"
        raise ValueError(msg)
    if a_arr.size == 0:
        return {
            "mean_delta": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "p_value": float("nan"),
            "n": 0,
        }
    d = a_arr - b_arr
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    boot = d[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p = 2.0 * min(float((boot <= 0).mean()), float((boot >= 0).mean()))
    return {
        "mean_delta": float(d.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_value": min(1.0, p),
        "n": int(d.size),
    }


def fp_rate_on_empty(
    pred_areas: Sequence[float],
    gt_areas: Sequence[float],
) -> dict[str, float]:
    """False-positive rate on empty ground truth.

    An "empty GT" image has ``gt_area == 0`` (no wound present — e.g. a healthy
    foot). Defining emptiness from the *mask*, not the class label, is
    deliberate: non-DFU images carry real wound masks, so a class-based rule
    would be wrong. A false positive is any nonzero predicted area on an empty
    GT. Returns ``{n_empty, n_false_positive, fp_rate}`` (rate NaN if no empties).
    """
    pred = np.asarray(list(pred_areas), dtype=float)
    gt = np.asarray(list(gt_areas), dtype=float)
    if pred.shape != gt.shape:
        msg = f"pred/gt area arrays must align, got {pred.shape} vs {gt.shape}"
        raise ValueError(msg)
    empty = gt <= 0
    n_empty = int(empty.sum())
    n_fp = int(np.count_nonzero((pred > 0) & empty))
    rate = float(n_fp / n_empty) if n_empty else float("nan")
    return {"n_empty": n_empty, "n_false_positive": n_fp, "fp_rate": rate}


def summarize_run(
    per_image_metrics: list[dict[str, float]],
    labels: Sequence[int],
    *,
    seed: int = 42,
) -> dict[str, Any]:
    """Summarize one model's per-image metrics into a study row.

    Produces the DFU-only slice (label == DFU), the full mixed slice, bootstrap
    CIs on Dice/IoU for each, the false-positive rate on empty GT, and the raw
    per-image Dice arrays needed for later paired significance tests.
    """
    labels = list(labels)
    if len(labels) != len(per_image_metrics):
        msg = f"labels/metrics length mismatch: {len(labels)} vs {len(per_image_metrics)}"
        raise ValueError(msg)

    dfu = [m for m, lb in zip(per_image_metrics, labels, strict=True) if lb == DFU_LABEL]

    dfu_dice = [m["dice"] for m in dfu]
    mixed_dice = [m["dice"] for m in per_image_metrics]
    dfu_iou = [m["iou"] for m in dfu]
    mixed_iou = [m["iou"] for m in per_image_metrics]

    return {
        "dfu_only": {
            "aggregate": aggregate_metrics(dfu),
            "dice_ci": bootstrap_ci(dfu_dice, seed=seed),
            "iou_ci": bootstrap_ci(dfu_iou, seed=seed),
        },
        "mixed": {
            "aggregate": aggregate_metrics(per_image_metrics),
            "dice_ci": bootstrap_ci(mixed_dice, seed=seed),
            "iou_ci": bootstrap_ci(mixed_iou, seed=seed),
        },
        "false_positive_on_empty": fp_rate_on_empty(
            [m["wound_area_mm2"] for m in per_image_metrics],
            [m["wound_area_gt_mm2"] for m in per_image_metrics],
        ),
        "per_image": {
            "dfu_dice": dfu_dice,
            "mixed_dice": mixed_dice,
            "labels": labels,
        },
    }


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    """SHA-256 of a file's bytes (streamed)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def git_commit(cwd: str | Path | None = None) -> str | None:
    """Current git commit hash, or None if unavailable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607 — 'git' from PATH is intended
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip() or None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _pkg_version(name: str) -> str | None:
    """Installed version of a package, or None."""
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def build_provenance(
    *,
    split_csv: str | Path,
    checkpoint: str | Path | None,
    arch: str,
    composition: str,
    seed: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a provenance block that makes a result number reproducible.

    Records the test split (path + content hash), the evaluated checkpoint
    (path + hash), architecture, composition, seed, git commit, timestamp, and
    key package versions. Missing files hash to ``None`` rather than raising, so
    a partially-staged run still produces an (honest, gap-flagged) record.
    """
    split_path = Path(split_csv)
    ckpt_path = Path(checkpoint) if checkpoint is not None else None
    prov: dict[str, Any] = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "git_commit": git_commit(split_path.parent if split_path.parent.exists() else None),
        "arch": arch,
        "composition": composition,
        "seed": seed,
        "split_csv": str(split_path),
        "split_csv_sha256": sha256_file(split_path) if split_path.exists() else None,
        "checkpoint": str(ckpt_path) if ckpt_path else None,
        "checkpoint_sha256": (sha256_file(ckpt_path) if ckpt_path and ckpt_path.exists() else None),
        "versions": {
            "torch": _pkg_version("torch"),
            "numpy": _pkg_version("numpy"),
            "segmentation-models-pytorch": _pkg_version("segmentation-models-pytorch"),
        },
    }
    if extra:
        prov.update(extra)
    return prov
