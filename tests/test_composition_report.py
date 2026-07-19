"""Tests for src/evaluation/composition_report.py — study statistics + provenance."""

from __future__ import annotations

import hashlib
import math
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from src.evaluation.composition_report import (
    bootstrap_ci,
    build_provenance,
    fp_rate_on_empty,
    paired_delta_ci,
    sha256_file,
    summarize_run,
)


def test_bootstrap_ci_mean_and_bracket() -> None:
    vals = [0.8, 0.9, 0.85, 0.95, 0.7, 0.88]
    ci = bootstrap_ci(vals, seed=0)
    assert math.isclose(ci["mean"], sum(vals) / len(vals), rel_tol=1e-9)
    assert ci["ci_low"] <= ci["mean"] <= ci["ci_high"]
    assert ci["n"] == 6


def test_bootstrap_ci_empty() -> None:
    ci = bootstrap_ci([])
    assert ci["n"] == 0
    assert math.isnan(ci["mean"])


def test_paired_delta_identical_is_not_significant() -> None:
    a = [0.5, 0.6, 0.7, 0.8]
    res = paired_delta_ci(a, a, seed=0)
    assert math.isclose(res["mean_delta"], 0.0, abs_tol=1e-9)
    assert res["p_value"] > 0.5  # no difference -> far from significant


def test_paired_delta_clear_difference_is_significant() -> None:
    a = [0.90, 0.92, 0.88, 0.91, 0.89]
    b = [0.70, 0.72, 0.68, 0.71, 0.69]  # a consistently higher, same images
    res = paired_delta_ci(a, b, seed=0)
    assert res["mean_delta"] > 0.15
    assert res["ci_low"] > 0  # whole CI above zero
    assert res["p_value"] < 0.05


def test_paired_delta_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="equal length"):
        paired_delta_ci([1.0, 2.0], [1.0])


def test_fp_rate_on_empty_counts_false_positives() -> None:
    # 4 empty-GT images (gt=0); model predicts nonzero area on 1 of them.
    gt = [0.0, 0.0, 0.0, 0.0, 100.0, 250.0]
    pred = [0.0, 5.0, 0.0, 0.0, 90.0, 0.0]
    res = fp_rate_on_empty(pred, gt)
    assert res["n_empty"] == 4
    assert res["n_false_positive"] == 1
    assert math.isclose(res["fp_rate"], 0.25)


def test_fp_rate_no_empties_is_nan() -> None:
    res = fp_rate_on_empty([10.0, 20.0], [5.0, 25.0])
    assert res["n_empty"] == 0
    assert math.isnan(res["fp_rate"])


def _img(dice: float, iou: float, pred_area: float, gt_area: float) -> dict[str, float]:
    return {"dice": dice, "iou": iou, "wound_area_mm2": pred_area, "wound_area_gt_mm2": gt_area}


def test_summarize_run_slices_dfu_and_computes_fp() -> None:
    # labels: 0=healthy(empty GT), 1=non_dfu(has GT), 2=dfu(has GT)
    metrics = [
        _img(0.0, 0.0, 12.0, 0.0),  # healthy, false positive
        _img(1.0, 1.0, 0.0, 0.0),  # healthy, true negative
        _img(0.80, 0.70, 300.0, 320.0),  # non_dfu wound
        _img(0.90, 0.83, 500.0, 510.0),  # dfu wound
        _img(0.86, 0.79, 410.0, 400.0),  # dfu wound
    ]
    labels = [0, 0, 1, 2, 2]
    out = summarize_run(metrics, labels, seed=0)

    # DFU-only slice uses the two label==2 images.
    assert out["dfu_only"]["dice_ci"]["n"] == 2
    assert math.isclose(out["dfu_only"]["aggregate"]["dice"]["mean"], (0.90 + 0.86) / 2)
    # Mixed slice uses all five.
    assert out["mixed"]["dice_ci"]["n"] == 5
    # FP-on-empty: 2 empty-GT images, 1 false positive.
    assert out["false_positive_on_empty"]["n_empty"] == 2
    assert out["false_positive_on_empty"]["n_false_positive"] == 1
    # Per-image dice arrays preserved for paired tests.
    assert out["per_image"]["dfu_dice"] == [0.90, 0.86]
    assert len(out["per_image"]["mixed_dice"]) == 5


def test_summarize_run_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        summarize_run([_img(1.0, 1.0, 0.0, 0.0)], [0, 1])


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    p = tmp_path / "x.bin"
    payload = b"diafoot-composition" * 100
    p.write_bytes(payload)
    assert sha256_file(p) == hashlib.sha256(payload).hexdigest()


def test_build_provenance_records_split_hash_and_seed(tmp_path: Path) -> None:
    split = tmp_path / "test.csv"
    split.write_text("image,class\nd_0.png,dfu\n")
    prov = build_provenance(
        split_csv=split,
        checkpoint=None,
        arch="unetpp",
        composition="dfu_only",
        seed=42,
    )
    assert prov["arch"] == "unetpp"
    assert prov["composition"] == "dfu_only"
    assert prov["seed"] == 42
    assert prov["split_csv_sha256"] == sha256_file(split)
    assert prov["checkpoint"] is None
    assert "timestamp_utc" in prov
    assert "torch" in prov["versions"]
