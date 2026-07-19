"""DiaFoot.AI — aggregate composition-study cells into the paper table.

Reads every per-cell JSON written by ``run_composition_experiment.py``, and
emits:
  * ``composition_comparison.json`` — condensed per-cell rows + fold-averaged
    condition summaries + paired significance vs the DFU-only reference;
  * ``composition_comparison.md`` — the fold-averaged results table (DFU Dice
    mean ± std per composition × architecture).

All cells are evaluated on the identical fixed clean test set in the same order,
so the per-image DFU-Dice arrays are aligned across cells — that alignment is
what lets us run a *paired* bootstrap between two compositions on the same fold.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.composition_report import paired_delta_ci

logger = logging.getLogger("aggregate")

REFERENCE_COMPOSITION = "dfu_only"


def condense_run(run: dict[str, Any]) -> dict[str, Any]:
    """Extract the citable numbers from one per-cell result dict."""
    s = run["summary"]
    dfu, mixed = s["dfu_only"], s["mixed"]
    fp = s["false_positive_on_empty"]
    prov = run.get("provenance", {})
    return {
        "run_tag": run["run_tag"],
        "arch": run["arch"],
        "composition": run["composition"],
        "seed": run["seed"],
        "fold": run.get("fold"),
        "n_train_by_class": prov.get("n_train_by_class", {}),
        "n_test": prov.get("n_test"),
        "dfu_dice": dfu["dice_ci"]["mean"],
        "dfu_dice_ci": [dfu["dice_ci"]["ci_low"], dfu["dice_ci"]["ci_high"]],
        "dfu_dice_n": dfu["dice_ci"]["n"],
        "dfu_iou": dfu["aggregate"].get("iou", {}).get("mean"),
        "dfu_nsd5mm": dfu["aggregate"].get("nsd_5mm", {}).get("mean"),
        "dfu_hd95": dfu["aggregate"].get("hd95", {}).get("mean"),
        "mixed_dice_mean": mixed["aggregate"].get("dice", {}).get("mean"),
        "mixed_dice_median": mixed["aggregate"].get("dice", {}).get("median"),
        "fp_on_empty_rate": fp["fp_rate"],
        "fp_on_empty": f"{fp['n_false_positive']}/{fp['n_empty']}",
        "checkpoint_sha256": prov.get("checkpoint_sha256"),
        "split_csv_sha256": prov.get("split_csv_sha256"),
        "git_commit": prov.get("git_commit"),
    }


def _mean_std(values: list[float]) -> tuple[float, float]:
    """Mean and sample std (std=0 for a single value), ignoring NaNs."""
    vals = [v for v in values if v == v]
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan")
    mean = sum(vals) / n
    std = (sum((v - mean) ** 2 for v in vals) / (n - 1)) ** 0.5 if n > 1 else 0.0
    return mean, std


def condition_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Average the metrics of one (arch, composition) condition across its CV folds."""
    dfu_mean, dfu_std = _mean_std([r["dfu_dice"] for r in rows])
    iou_mean, _ = _mean_std([r["dfu_iou"] for r in rows if r["dfu_iou"] is not None])
    mixed_mean, _ = _mean_std(
        [r["mixed_dice_mean"] for r in rows if r["mixed_dice_mean"] is not None]
    )
    fp_mean, _ = _mean_std(
        [r["fp_on_empty_rate"] for r in rows if r["fp_on_empty_rate"] is not None]
    )
    example = rows[0]
    return {
        "arch": example["arch"],
        "composition": example["composition"],
        "n_folds": len(rows),
        "folds": sorted(r["fold"] for r in rows if r["fold"] is not None),
        "seeds": sorted({r["seed"] for r in rows}),
        "n_train_by_class": example["n_train_by_class"],
        "dfu_dice_mean": dfu_mean,
        "dfu_dice_std": dfu_std,
        "dfu_iou_mean": iou_mean,
        "mixed_dice_mean": mixed_mean,
        "fp_on_empty_rate_mean": fp_mean,
    }


def paired_vs_reference(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Paired bootstrap of DFU-Dice: each composition vs DFU-only, per arch/seed/fold.

    Pairs are formed within the SAME (arch, seed, fold) so both models were
    evaluated on the identical fixed test set (aligned per-image DFU-Dice).
    """
    by_key: dict[tuple[str, int, object, str], dict[str, Any]] = {
        (r["arch"], r["seed"], r.get("fold"), r["composition"]): r for r in runs
    }
    out: list[dict[str, Any]] = []
    for (arch, seed, fold, comp), run in by_key.items():
        if comp == REFERENCE_COMPOSITION:
            continue
        ref = by_key.get((arch, seed, fold, REFERENCE_COMPOSITION))
        if ref is None:
            continue
        a = run["summary"]["per_image"]["dfu_dice"]
        b = ref["summary"]["per_image"]["dfu_dice"]
        if len(a) != len(b) or not a:
            logger.warning("skip paired %s vs ref (seed %d fold %s): misaligned", comp, seed, fold)
            continue
        delta = paired_delta_ci(a, b)  # composition - dfu_only
        out.append(
            {
                "arch": arch,
                "seed": seed,
                "fold": fold,
                "composition": comp,
                "vs": REFERENCE_COMPOSITION,
                **delta,
            }
        )
    return out


def _fmt(x: Any, nd: int = 3) -> str:
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) and x == x else "—"


def markdown_table(summaries: list[dict[str, Any]]) -> str:
    """Render fold-averaged condition summaries as the paper table (best-first per arch)."""

    def _sort_key(s: dict[str, Any]) -> tuple[str, float]:
        d = s["dfu_dice_mean"]
        return (s["arch"], -(d if d == d else -1.0))  # NaN sorts last within arch

    summaries = sorted(summaries, key=_sort_key)
    header = (
        "| Arch | Composition | Folds | Train (dfu/nonDFU/healthy) | "
        "DFU Dice (mean±std) | DFU IoU | Mixed Dice | FP-on-empty |\n"
        "|---|---|---|---|---|---|---|---|\n"
    )
    lines = []
    for s in summaries:
        c = s["n_train_by_class"]
        train = f"{c.get('dfu', 0)}/{c.get('non_dfu', 0)}/{c.get('healthy', 0)}"
        dice = f"{_fmt(s['dfu_dice_mean'])} ± {_fmt(s['dfu_dice_std'])}"
        lines.append(
            f"| {s['arch']} | {s['composition']} | {s['n_folds']} | {train} | "
            f"{dice} | {_fmt(s['dfu_iou_mean'])} | {_fmt(s['mixed_dice_mean'])} | "
            f"{_fmt(s['fp_on_empty_rate_mean'])} |"
        )
    return header + "\n".join(lines) + "\n"


def main() -> None:
    """Aggregate all per-cell JSONs into the paper comparison table."""
    parser = argparse.ArgumentParser(description="Aggregate composition cells")
    parser.add_argument("--results-dir", default="results/composition")
    parser.add_argument("--output-json", default="results/composition_comparison.json")
    parser.add_argument("--output-md", default="results/composition_comparison.md")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    result_dir = Path(args.results_dir)
    files = sorted(result_dir.glob("*.json"))
    if not files:
        logger.error("no per-cell result JSONs in %s", result_dir)
        raise SystemExit(1)

    runs = [json.loads(p.read_text()) for p in files]
    condensed = [condense_run(r) for r in runs]

    by_cond: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in condensed:
        by_cond[(r["arch"], r["composition"])].append(r)
    summaries = [condition_summary(rows) for rows in by_cond.values()]

    payload = {
        "condition_summaries": summaries,
        "runs": condensed,
        "paired_vs_dfu_only": paired_vs_reference(runs),
        "n_cells": len(runs),
        "n_conditions": len(summaries),
    }
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2))
    Path(args.output_md).write_text(markdown_table(summaries))

    logger.info(
        "aggregated %d cells / %d conditions -> %s and %s",
        len(runs),
        len(summaries),
        out_json,
        args.output_md,
    )
    print(markdown_table(summaries))


if __name__ == "__main__":
    main()
