"""Training-set composition controls for the DFU segmentation study.

Pure, file-light helpers so the composition logic is unit-testable without
images or a GPU. A *row* is one record from a split CSV (a ``dict`` of
column -> value); the only column this module reasons about is ``class`` in
``{healthy, non_dfu, dfu}``.

Two orthogonal composition axes are supported:

* **categorical composition** (:func:`select_composition`) — which classes are
  present in training: ``dfu_only`` / ``dfu_nondfu`` / ``all``.
* **negative-ratio sweep** (:func:`subsample_negatives`) — keep every DFU
  (positive) row and a deterministic, class-stratified fraction of the negative
  pool (healthy + non_dfu), for a dose-response curve.

The point of a single canonical module is that the exact rows that go into each
training run are reproducible and auditable — the whole reason this study exists
is that composition changes the result, so *which rows* is part of the science.
"""

from __future__ import annotations

import csv
import random
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

Row = dict[str, str]

POSITIVE_CLASS = "dfu"
NEGATIVE_CLASSES: tuple[str, ...] = ("healthy", "non_dfu")
ALL_CLASSES: tuple[str, ...] = ("healthy", "non_dfu", "dfu")

# Categorical composition presets (which classes are present in TRAINING).
COMPOSITIONS: dict[str, tuple[str, ...]] = {
    "dfu_only": ("dfu",),
    "dfu_nondfu": ("dfu", "non_dfu"),
    "all": ("healthy", "non_dfu", "dfu"),
}


def _class_of(row: Row) -> str:
    """Return a row's class, failing loudly if the column is absent.

    Silent defaulting here would corrupt the composition (e.g. treating an
    unlabeled row as healthy), so a missing ``class`` is an error, not a guess.
    """
    cls = row.get("class")
    if cls is None:
        msg = "row is missing required 'class' column"
        raise KeyError(msg)
    return cls


def filter_by_classes(rows: Iterable[Row], include: Iterable[str]) -> list[Row]:
    """Keep only rows whose class is in ``include`` (order preserved)."""
    keep = set(include)
    return [r for r in rows if _class_of(r) in keep]


def select_composition(rows: Iterable[Row], composition: str) -> list[Row]:
    """Select rows for a categorical composition preset.

    Args:
        rows: Split rows (each a dict with a ``class`` column).
        composition: One of :data:`COMPOSITIONS` (``dfu_only`` / ``dfu_nondfu``
            / ``all``).

    Returns:
        The subset of rows whose class belongs to the composition, order
        preserved.
    """
    if composition not in COMPOSITIONS:
        msg = f"unknown composition {composition!r}; expected one of {sorted(COMPOSITIONS)}"
        raise ValueError(msg)
    return filter_by_classes(rows, COMPOSITIONS[composition])


def subsample_negatives(
    rows: Iterable[Row],
    neg_frac: float,
    seed: int = 42,
    sort_key: str = "image",
) -> list[Row]:
    """Keep all positives + a deterministic, class-stratified fraction of negatives.

    Every ``dfu`` (positive) row is always kept. For each negative class
    (healthy, non_dfu) independently, a ``neg_frac`` fraction is retained so the
    healthy:non_dfu ratio is preserved as the pool shrinks. Selection is
    deterministic given ``seed``: the pool is sorted by ``sort_key`` for a
    stable ordering, then sampled with a per-class-seeded RNG.

    ``neg_frac == 0.0`` reduces to DFU-only; ``neg_frac == 1.0`` keeps every
    negative (equivalent to the ``all`` composition on this row set).

    Args:
        rows: Split rows.
        neg_frac: Fraction of each negative class to retain, in [0, 1].
        seed: Base RNG seed (each negative class gets a distinct derived seed).
        sort_key: Column used to impose a stable order before sampling.

    Returns:
        Positives followed by the retained negatives.
    """
    if not 0.0 <= neg_frac <= 1.0:
        msg = f"neg_frac must be in [0, 1], got {neg_frac}"
        raise ValueError(msg)

    rows = list(rows)
    kept: list[Row] = [r for r in rows if _class_of(r) == POSITIVE_CLASS]

    for class_index, neg_cls in enumerate(NEGATIVE_CLASSES):
        pool = sorted(
            (r for r in rows if _class_of(r) == neg_cls),
            key=lambda r: r.get(sort_key, ""),
        )
        keep_n = round(neg_frac * len(pool))
        if keep_n <= 0:
            continue
        if keep_n >= len(pool):
            kept.extend(pool)
            continue
        rng = random.Random(seed * 1000 + class_index)  # noqa: S311 — reproducible sampling, not crypto
        kept.extend(rng.sample(pool, keep_n))

    return kept


def class_counts(rows: Iterable[Row]) -> dict[str, int]:
    """Return a ``{class: count}`` mapping for the given rows."""
    return dict(Counter(_class_of(r) for r in rows))


def read_split_csv(path: str | Path) -> list[Row]:
    """Read a split CSV into a list of row dicts."""
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_split_csv(rows: list[Row], path: str | Path, fieldnames: list[str] | None = None) -> int:
    """Write rows to ``path`` as CSV, creating parent dirs. Returns row count.

    ``fieldnames`` defaults to the keys of the first row. Writing an empty row
    set requires an explicit ``fieldnames`` (there is no header to infer).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        if not rows:
            msg = "cannot infer fieldnames from empty rows; pass fieldnames explicitly"
            raise ValueError(msg)
        fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)
