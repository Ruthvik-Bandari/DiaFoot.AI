"""Tests for scripts/aggregate_composition_results.py — the paper-table builder."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

_SPEC = importlib.util.spec_from_file_location(
    "aggregate_composition_results",
    Path(__file__).resolve().parent.parent / "scripts" / "aggregate_composition_results.py",
)
assert _SPEC is not None
assert _SPEC.loader is not None
agg = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(agg)


def _run(
    arch: str,
    comp: str,
    seed: int,
    dfu_dice: float,
    dfu_slice: list[float],
    fold: int = 0,
) -> dict[str, Any]:
    return {
        "run_tag": f"{arch}_{comp}_seed{seed}_fold{fold}",
        "arch": arch,
        "composition": comp,
        "seed": seed,
        "fold": fold,
        "summary": {
            "dfu_only": {
                "aggregate": {
                    "dice": {"mean": dfu_dice, "median": dfu_dice},
                    "iou": {"mean": dfu_dice - 0.05},
                    "nsd_5mm": {"mean": 0.95},
                    "hd95": {"mean": 12.0},
                },
                "dice_ci": {
                    "mean": dfu_dice,
                    "ci_low": dfu_dice - 0.02,
                    "ci_high": dfu_dice + 0.02,
                    "n": len(dfu_slice),
                },
                "iou_ci": {
                    "mean": dfu_dice - 0.05,
                    "ci_low": 0.0,
                    "ci_high": 1.0,
                    "n": len(dfu_slice),
                },
            },
            "mixed": {
                "aggregate": {"dice": {"mean": 0.72, "median": 0.93}, "iou": {"mean": 0.67}},
                "dice_ci": {"mean": 0.72, "ci_low": 0.70, "ci_high": 0.74, "n": 20},
                "iou_ci": {"mean": 0.67, "ci_low": 0.65, "ci_high": 0.69, "n": 20},
            },
            "false_positive_on_empty": {"n_empty": 5, "n_false_positive": 1, "fp_rate": 0.2},
            "per_image": {
                "dfu_dice": dfu_slice,
                "mixed_dice": dfu_slice,
                "labels": [2] * len(dfu_slice),
            },
        },
        "provenance": {
            "n_train_by_class": {"dfu": 1483, "non_dfu": 1880, "healthy": 2311},
            "n_test": 1161,
            "checkpoint_sha256": "abc",
            "split_csv_sha256": "def",
            "git_commit": "cafe",
        },
    }


def test_condense_run_pulls_citable_numbers() -> None:
    c = agg.condense_run(_run("unetpp", "dfu_only", 42, 0.851, [0.85, 0.86, 0.84], fold=3))
    assert c["arch"] == "unetpp"
    assert c["dfu_dice"] == 0.851
    assert c["fold"] == 3
    assert c["n_train_by_class"] == {"dfu": 1483, "non_dfu": 1880, "healthy": 2311}
    assert c["fp_on_empty"] == "1/5"


def test_condition_summary_averages_across_folds() -> None:
    rows = [
        agg.condense_run(_run("unetpp", "all", 42, 0.80 + 0.01 * f, [0.8], fold=f))
        for f in range(5)
    ]
    s = agg.condition_summary(rows)
    assert s["n_folds"] == 5
    assert s["folds"] == [0, 1, 2, 3, 4]
    assert abs(s["dfu_dice_mean"] - 0.82) < 1e-9  # mean of 0.80..0.84
    assert s["dfu_dice_std"] > 0


def test_paired_vs_reference_pairs_within_same_fold() -> None:
    runs = [
        _run("unetpp", "dfu_only", 42, 0.851, [0.90, 0.88, 0.86, 0.92], fold=1),
        _run("unetpp", "all", 42, 0.824, [0.80, 0.78, 0.79, 0.81], fold=1),  # lower, same images
        _run("unetpp", "all", 42, 0.83, [0.99, 0.99, 0.99, 0.99], fold=2),  # no ref at fold 2
    ]
    paired = agg.paired_vs_reference(runs)
    # only the fold-1 pair has a matching dfu_only reference
    assert len(paired) == 1
    p = paired[0]
    assert p["composition"] == "all"
    assert p["fold"] == 1
    assert p["mean_delta"] < 0  # adding negatives lowered DFU dice here


def test_markdown_table_renders_condition_summaries() -> None:
    rows = [
        agg.condense_run(_run("unetpp", "dfu_only", 42, 0.851, [0.85], fold=f)) for f in range(3)
    ]
    summary = agg.condition_summary(rows)
    md = agg.markdown_table([summary])
    assert "DFU Dice (mean±std)" in md
    assert "dfu_only" in md
    assert "1483/1880/2311" in md
