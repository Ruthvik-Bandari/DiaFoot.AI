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
# These four are the class-subset conditions from the manuscript; "random_mixed"
# is NOT here because it is a size-matched draw, not a class subset (see
# :func:`random_mixed`).
COMPOSITIONS: dict[str, tuple[str, ...]] = {
    "dfu_only": ("dfu",),
    "dfu_healthy": ("dfu", "healthy"),
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


def random_mixed(rows: Iterable[Row], seed: int = 42, sort_key: str = "image") -> list[Row]:
    """Size-matched random draw across all classes (the ``random_mixed`` condition).

    Draws ``N`` rows uniformly at random from the *entire* pool (all classes),
    where ``N`` is the number of DFU rows in the pool. This is the manuscript's
    key control: a training set the **same size** as DFU-only but with mixed
    content, so any performance difference is attributable to composition, not
    dataset size. Deterministic given ``seed`` (pool sorted by ``sort_key``
    first for stable ordering).

    If DFU rows are >= the whole pool (degenerate), the full pool is returned.
    """
    rows = list(rows)
    n_dfu = sum(1 for r in rows if _class_of(r) == POSITIVE_CLASS)
    pool = sorted(rows, key=lambda r: r.get(sort_key, ""))
    if n_dfu >= len(pool):
        return pool
    rng = random.Random(seed)  # noqa: S311 — reproducible sampling, not crypto
    return rng.sample(pool, n_dfu)


def group_kfold(
    rows: Iterable[Row],
    n_folds: int = 5,
    seed: int = 42,
    group_key: str = "patient_id",
) -> list[tuple[list[Row], list[Row]]]:
    """Grouped k-fold partition of ``rows`` (returns ``[(train, val)]`` per fold).

    Rows sharing a ``group_key`` value are always kept in the same fold, so a
    group never straddles train/val within a fold. Groups are shuffled with a
    seeded RNG then round-robined across folds (balances the group count per
    fold). Deterministic given ``seed``.

    Note on this dataset: the public sources lack true patient IDs
    (``patient_id`` is ~unique per image), so this is the strongest available
    provenance grouping, not a guarantee of patient independence — state that
    honestly. Because the composition study evaluates every fold on a *fixed*
    held-out test set (the fold's val split only drives early stopping), the
    reported test metric is unaffected by any residual intra-pool fold leakage.
    """
    if n_folds < 2:
        msg = f"n_folds must be >= 2, got {n_folds}"
        raise ValueError(msg)
    groups: dict[str, list[Row]] = {}
    for r in rows:
        key = r.get(group_key) or r.get("image") or ""
        groups.setdefault(key, []).append(r)
    keys = sorted(groups)
    rng = random.Random(seed)  # noqa: S311 — reproducible fold assignment, not crypto
    rng.shuffle(keys)
    fold_of = {k: i % n_folds for i, k in enumerate(keys)}
    folds: list[tuple[list[Row], list[Row]]] = []
    for f in range(n_folds):
        val = [r for k in keys if fold_of[k] == f for r in groups[k]]
        train = [r for k in keys if fold_of[k] != f for r in groups[k]]
        folds.append((train, val))
    return folds


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
