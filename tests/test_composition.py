"""Tests for src/data/composition.py — training-set composition controls.

These are pure-logic tests (no images, no GPU): they pin down exactly which
rows each composition/ratio setting selects, because "which rows train the
model" is the independent variable of the whole dataset-composition study.
"""

from __future__ import annotations

import pytest

from src.data.composition import (
    class_counts,
    filter_by_classes,
    group_kfold,
    random_mixed,
    read_split_csv,
    select_composition,
    subsample_negatives,
    write_split_csv,
)


def _rows() -> list[dict[str, str]]:
    """20 healthy + 10 non_dfu + 8 dfu synthetic split rows."""
    rows: list[dict[str, str]] = []
    for i in range(20):
        rows.append({"image": f"h_{i:03d}.png", "mask": "", "class": "healthy"})
    for i in range(10):
        rows.append({"image": f"n_{i:03d}.png", "mask": f"n_{i:03d}.png", "class": "non_dfu"})
    for i in range(8):
        rows.append({"image": f"d_{i:03d}.png", "mask": f"d_{i:03d}.png", "class": "dfu"})
    return rows


def test_select_composition_dfu_only() -> None:
    out = select_composition(_rows(), "dfu_only")
    assert class_counts(out) == {"dfu": 8}


def test_select_composition_dfu_nondfu() -> None:
    out = select_composition(_rows(), "dfu_nondfu")
    assert class_counts(out) == {"non_dfu": 10, "dfu": 8}


def test_select_composition_dfu_healthy() -> None:
    out = select_composition(_rows(), "dfu_healthy")
    assert class_counts(out) == {"healthy": 20, "dfu": 8}


def test_select_composition_all_keeps_everything() -> None:
    out = select_composition(_rows(), "all")
    assert class_counts(out) == {"healthy": 20, "non_dfu": 10, "dfu": 8}


def test_random_mixed_size_matches_dfu_count() -> None:
    # 8 DFU in the pool -> random_mixed draws exactly 8 rows total, across classes.
    out = random_mixed(_rows(), seed=42)
    assert len(out) == 8
    # Drawn from the whole pool, so it should contain negatives (not pure DFU).
    assert len(class_counts(out)) > 1


def test_random_mixed_is_deterministic() -> None:
    a = [r["image"] for r in random_mixed(_rows(), seed=42)]
    b = [r["image"] for r in random_mixed(_rows(), seed=42)]
    assert a == b


def test_random_mixed_subset_of_pool() -> None:
    pool = {r["image"] for r in _rows()}
    out = {r["image"] for r in random_mixed(_rows(), seed=7)}
    assert out <= pool


def test_select_composition_unknown_raises() -> None:
    with pytest.raises(ValueError, match="unknown composition"):
        select_composition(_rows(), "dfu_plus_moon")


def test_filter_by_classes_preserves_order() -> None:
    out = filter_by_classes(_rows(), ["dfu", "healthy"])
    # healthy rows come first in the source, dfu last — order preserved.
    assert out[0]["class"] == "healthy"
    assert out[-1]["class"] == "dfu"
    assert class_counts(out) == {"healthy": 20, "dfu": 8}


def test_subsample_negatives_zero_is_dfu_only() -> None:
    out = subsample_negatives(_rows(), neg_frac=0.0)
    assert class_counts(out) == {"dfu": 8}


def test_subsample_negatives_one_keeps_all() -> None:
    out = subsample_negatives(_rows(), neg_frac=1.0)
    assert class_counts(out) == {"healthy": 20, "non_dfu": 10, "dfu": 8}


def test_subsample_negatives_half_is_class_stratified() -> None:
    out = subsample_negatives(_rows(), neg_frac=0.5, seed=42)
    # positives always fully retained; each negative class halved independently.
    assert class_counts(out) == {"healthy": 10, "non_dfu": 5, "dfu": 8}


def test_subsample_negatives_is_deterministic() -> None:
    a = subsample_negatives(_rows(), neg_frac=0.5, seed=42)
    b = subsample_negatives(_rows(), neg_frac=0.5, seed=42)
    assert [r["image"] for r in a] == [r["image"] for r in b]


def test_subsample_negatives_seed_changes_selection() -> None:
    a = {r["image"] for r in subsample_negatives(_rows(), neg_frac=0.5, seed=1)}
    b = {r["image"] for r in subsample_negatives(_rows(), neg_frac=0.5, seed=2)}
    # Different seeds should generally pick different negatives (not identical).
    assert a != b


def test_subsample_negatives_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match="neg_frac"):
        subsample_negatives(_rows(), neg_frac=1.5)


def test_missing_class_column_raises() -> None:
    with pytest.raises(KeyError, match="class"):
        select_composition([{"image": "x.png"}], "all")


def _grouped_rows() -> list[dict[str, str]]:
    """30 rows: 5 patients with 3 images each (grouped) + 15 singletons."""
    rows: list[dict[str, str]] = []
    for p in range(5):
        for i in range(3):
            rows.append({"image": f"p{p}_{i}.png", "class": "dfu", "patient_id": f"patient_{p}"})
    for s in range(15):
        rows.append({"image": f"s{s}.png", "class": "dfu", "patient_id": f"solo_{s}"})
    return rows


def test_group_kfold_returns_n_folds() -> None:
    folds = group_kfold(_grouped_rows(), n_folds=5, seed=42)
    assert len(folds) == 5


def test_group_kfold_val_sets_partition_the_pool() -> None:
    rows = _grouped_rows()
    folds = group_kfold(rows, n_folds=5, seed=42)
    seen: list[str] = []
    for _train, val in folds:
        seen += [r["image"] for r in val]
    # every row appears in exactly one fold's validation split
    assert sorted(seen) == sorted(r["image"] for r in rows)


def test_group_kfold_no_patient_straddles_a_fold() -> None:
    folds = group_kfold(_grouped_rows(), n_folds=5, seed=42)
    for train, val in folds:
        train_patients = {r["patient_id"] for r in train}
        val_patients = {r["patient_id"] for r in val}
        assert train_patients.isdisjoint(val_patients)


def test_group_kfold_train_val_cover_pool() -> None:
    rows = _grouped_rows()
    for train, val in group_kfold(rows, n_folds=5, seed=42):
        assert len(train) + len(val) == len(rows)


def test_group_kfold_deterministic() -> None:
    a = group_kfold(_grouped_rows(), seed=42)
    b = group_kfold(_grouped_rows(), seed=42)
    assert [[r["image"] for r in val] for _, val in a] == [
        [r["image"] for r in val] for _, val in b
    ]


def test_group_kfold_rejects_one_fold() -> None:
    with pytest.raises(ValueError, match="n_folds"):
        group_kfold(_grouped_rows(), n_folds=1)


def test_write_then_read_roundtrip(tmp_path: object) -> None:
    from pathlib import Path

    assert isinstance(tmp_path, Path)
    rows = select_composition(_rows(), "dfu_nondfu")
    out = tmp_path / "sub" / "train.csv"
    n = write_split_csv(rows, out)
    assert n == 18
    back = read_split_csv(out)
    assert class_counts(back) == {"non_dfu": 10, "dfu": 8}


def test_write_empty_without_fieldnames_raises(tmp_path: object) -> None:
    from pathlib import Path

    assert isinstance(tmp_path, Path)
    with pytest.raises(ValueError, match="fieldnames"):
        write_split_csv([], tmp_path / "empty.csv")
