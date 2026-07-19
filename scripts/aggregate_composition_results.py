"""DiaFoot.AI — aggregate composition-study cells into the paper table.

Reads every per-cell JSON written by ``run_composition_experiment.py``, and
emits:
  * ``composition_comparison.json`` — condensed, machine-readable rows +
    across-seed summaries + paired significance vs the DFU-only reference;
  * ``composition_comparison.md`` — a ready-to-paste results table with 95% CIs.

All cells are evaluated on the identical clean test set in the same order, so
the per-image DFU-Dice arrays are aligned across cells — that alignment is what
lets us run a *paired* bootstrap between two compositions.
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


def across_seed_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean and std of DFU Dice across seeds for one (arch, composition)."""
    dice = [r["dfu_dice"] for r in rows if r["dfu_dice"] == r["dfu_dice"]]  # drop NaN
    n = len(dice)
    mean = sum(dice) / n if n else float("nan")
    if n > 1:
        var = sum((d - mean) ** 2 for d in dice) / (n - 1)
        std = var**0.5
    else:
        std = 0.0
    return {
        "seeds": sorted(r["seed"] for r in rows),
        "dfu_dice_mean": mean,
        "dfu_dice_std": std,
        "n_seeds": n,
    }


def paired_vs_reference(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Paired bootstrap of DFU-Dice: each composition vs DFU-only, per arch/seed."""
    by_key: dict[tuple[str, int, str], dict[str, Any]] = {
        (r["arch"], r["seed"], r["composition"]): r for r in runs
    }
    out: list[dict[str, Any]] = []
    for (arch, seed, comp), run in by_key.items():
        if comp == REFERENCE_COMPOSITION:
            continue
        ref = by_key.get((arch, seed, REFERENCE_COMPOSITION))
        if ref is None:
            continue
        a = run["summary"]["per_image"]["dfu_dice"]
        b = ref["summary"]["per_image"]["dfu_dice"]
        if len(a) != len(b) or not a:
            logger.warning("skip paired %s vs ref (seed %d): misaligned slices", comp, seed)
            continue
        delta = paired_delta_ci(a, b)  # composition - dfu_only
        out.append(
            {"arch": arch, "seed": seed, "composition": comp, "vs": REFERENCE_COMPOSITION, **delta}
        )
    return out


def _fmt(x: Any, nd: int = 3) -> str:
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) and x == x else "—"


def markdown_table(rows: list[dict[str, Any]]) -> str:
    """Render condensed rows as a markdown results table (sorted best-first)."""

    def _sort_key(r: dict[str, Any]) -> tuple[str, float]:
        dice = r["dfu_dice"]
        return (r["arch"], -(dice if dice == dice else -1.0))  # NaN sorts last within arch

    rows = sorted(rows, key=_sort_key)
    header = (
        "| Arch | Composition | Seed | Train (dfu/nonDFU/healthy) | "
        "DFU Dice [95% CI] | DFU IoU | Mixed Dice mean/median | FP-on-empty |\n"
        "|---|---|---|---|---|---|---|---|\n"
    )
    lines = []
    for r in rows:
        c = r["n_train_by_class"]
        train = f"{c.get('dfu', 0)}/{c.get('non_dfu', 0)}/{c.get('healthy', 0)}"
        ci = r["dfu_dice_ci"]
        dice = f"{_fmt(r['dfu_dice'])} [{_fmt(ci[0])}, {_fmt(ci[1])}]"
        mixed = f"{_fmt(r['mixed_dice_mean'])} / {_fmt(r['mixed_dice_median'])}"
        lines.append(
            f"| {r['arch']} | {r['composition']} | {r['seed']} | {train} | "
            f"{dice} | {_fmt(r['dfu_iou'])} | {mixed} | "
            f"{r['fp_on_empty']} ({_fmt(r['fp_on_empty_rate'])}) |"
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
    seed_summary = {
        f"{arch}::{comp}": across_seed_summary(rows) for (arch, comp), rows in by_cond.items()
    }

    payload = {
        "runs": condensed,
        "across_seed_summary": seed_summary,
        "paired_vs_dfu_only": paired_vs_reference(runs),
        "n_cells": len(runs),
    }
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2))
    Path(args.output_md).write_text(markdown_table(condensed))

    logger.info("aggregated %d cells -> %s and %s", len(runs), out_json, args.output_md)
    print(markdown_table(condensed))


if __name__ == "__main__":
    main()
