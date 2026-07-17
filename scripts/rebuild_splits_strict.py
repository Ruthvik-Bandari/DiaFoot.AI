"""Rebuild train/val/test splits with stricter leakage control.

Strategy:
1) Load all rows from existing train/val/test CSV files
2) Recompute robust per-sample metadata (class/source/patient)
3) Build near-duplicate components (class-aware dHash graph)
4) Assign full components to splits (70/15/15) to avoid cross-split duplicates
5) Write refreshed split CSVs

Usage:
    python scripts/rebuild_splits_strict.py --split-dir data/splits --near-threshold 6
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dedup import canonical_stem
from src.data.dedup import dhash as _dhash
from src.data.dedup import hamming as _hamming

logger = logging.getLogger(__name__)


def infer_source_id(class_name: str, image_path: Path) -> str:
    stem = image_path.stem.lower()
    parts = {p.lower() for p in image_path.parts}
    if stem.startswith("azh_") or "azh" in parts:
        return "azh"
    if stem.startswith("fuseg_") or "fuseg" in parts:
        return "fuseg"
    if "mendeley" in parts:
        return "mendeley"
    if "kaggle" in parts:
        return "kaggle"
    return class_name


def infer_patient_id(class_name: str, image_path: Path) -> str:
    stem = canonical_stem(str(image_path))

    # AZH pattern: azh_train_<case>_<patch>
    if stem.startswith("azh_"):
        m = re.match(r"azh_(?:train|test|val)_([0-9a-f]{16,})_\d+$", stem)
        if m:
            return f"azh_{m.group(1)}"
        # An azh_-prefixed name that does not match means format drift. Falling
        # through to the per-image id below would silently stop grouping that
        # capture's patches — the exact leakage this rebuild prevents — so warn
        # loudly instead of failing quietly.
        logger.warning(
            "AZH-style filename %r does not match azh_<split>_<case>_<patch>; "
            "its patches will NOT be grouped. Check for filename format drift.",
            stem,
        )

    # keep numeric suffixes for healthy/non_dfu (prevents collapsing to one patient)
    # e.g. male_normal_1383, wound_main_0010
    return stem


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def load_rows(split_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen = set()
    for name in ("train.csv", "val.csv", "test.csv"):
        path = split_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Missing split file: {path}")
        with open(path) as f:
            for row in csv.DictReader(f):
                key = row.get("image", "")
                if key and key not in seen:
                    seen.add(key)
                    rows.append(row)
    return rows


def enrich_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in rows:
        image = row["image"]
        p = Path(image)
        class_name = (row.get("class") or p.parent.parent.name).strip().lower()
        # Recompute identity from the filename rather than trusting the CSV
        # columns. The existing patient_id was over-collapsed (e.g. every
        # "male_normal" foot mapped to the single id "male_normal"), which both
        # destroys grouping granularity and cannot express the AZH per-capture
        # id that keeps a wound's patches together.
        source_id = infer_source_id(class_name, p).strip().lower()
        patient_id = infer_patient_id(class_name, p).strip().lower()

        out.append(
            {
                "image": image,
                "mask": row.get("mask", image.replace("/images/", "/masks/")),
                "class": class_name,
                "ita": row.get("ita", ""),
                "ita_group": row.get("ita_group", "unknown"),
                "source_id": source_id,
                "patient_id": patient_id,
            }
        )
    return out


def build_components(rows: list[dict[str, str]], near_threshold: int) -> dict[str, list[list[int]]]:
    by_class = defaultdict(list)
    for idx, row in enumerate(rows):
        by_class[row["class"]].append(idx)

    class_components: dict[str, list[list[int]]] = {}

    for cls, indices in by_class.items():
        uf = UnionFind(len(indices))

        local_to_global = dict(enumerate(indices))

        # 1) Primary grouping: source-image identity from the filename.
        #    All patches / augmentations of one source capture share a
        #    component. This is the check that actually prevents the observed
        #    leakage: distinct crops ("_0", "_1") of one wound never collide
        #    under dHash, so a pixel-only grouping would let them straddle
        #    splits. It is filename-derivable, so it works even when the image
        #    files are absent (grouping runs before any pixels load).
        group_to_locals: dict[str, list[int]] = defaultdict(list)
        for li, gi in local_to_global.items():
            row = rows[gi]
            p = Path(row["image"])
            key = f"{infer_source_id(cls, p)}::{infer_patient_id(cls, p)}"
            group_to_locals[key].append(li)
        for locs in group_to_locals.values():
            root = locs[0]
            for li in locs[1:]:
                uf.union(root, li)

        # 2) Merge perceptual near-duplicates across source images (dHash).
        #    Full pairwise within class — NOT top-16-bit prefix bucketing,
        #    which drops genuine near-duplicates (Hamming distance <= threshold)
        #    into different buckets whenever any prefix bit differs, so they are
        #    never compared and leakage survives. The rebuild is offline, so the
        #    O(n^2)-per-class scan is acceptable. Runs only when pixels are
        #    available; with images absent, dHash yields nothing and this step
        #    is a no-op (source-image grouping above still applies).
        hashed: list[tuple[int, int]] = []
        for li, gi in local_to_global.items():
            hv = _dhash(Path(rows[gi]["image"]))
            if hv is not None:
                hashed.append((li, hv))

        for i in range(len(hashed)):
            li1, h1 = hashed[i]
            for j in range(i + 1, len(hashed)):
                li2, h2 = hashed[j]
                if _hamming(h1, h2) <= near_threshold:
                    uf.union(li1, li2)

        groups = defaultdict(list)
        for li in range(len(indices)):
            groups[uf.find(li)].append(local_to_global[li])

        class_components[cls] = list(groups.values())

    return class_components


def assign_components(
    rows: list[dict[str, str]],
    class_components: dict[str, list[list[int]]],
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> dict[str, list[dict[str, str]]]:
    rng = random.Random(seed)

    split_rows: dict[str, list[dict[str, str]]] = {"train": [], "val": [], "test": []}

    for cls, components in class_components.items():
        rng.shuffle(components)
        total = sum(len(c) for c in components)
        targets = {
            "train": round(total * train_ratio),
            "val": round(total * val_ratio),
            "test": max(0, total - round(total * train_ratio) - round(total * val_ratio)),
        }
        current = {"train": 0, "val": 0, "test": 0}
        # Track whole components per split. Assignment and repair operate at
        # component granularity; rows are materialized only afterwards. A
        # component is a set of rows (patches / near-duplicates) that MUST stay
        # together in one split, so it must never be broken apart.
        comps_in_split: dict[str, list[list[int]]] = {"train": [], "val": [], "test": []}

        for comp in components:
            size = len(comp)
            deficits = {s: targets[s] - current[s] for s in ("train", "val", "test")}
            # prioritize split with biggest deficit, then smallest current size
            chosen = sorted(
                ("train", "val", "test"),
                key=lambda s: (deficits[s], -current[s]),
                reverse=True,
            )[0]
            comps_in_split[chosen].append(comp)
            current[chosen] += size

        # Class-presence repair: give every empty split a WHOLE component,
        # donated from the split that currently holds the most. Moving whole
        # components (never individual rows) is essential — plucking a row out
        # of a component would split one source image across two splits and
        # reintroduce the exact leakage this rebuild exists to prevent. If a
        # class has too few components to seed all three splits, leave a split
        # without it rather than split a component.
        for empty in ("train", "val", "test"):
            if comps_in_split[empty]:
                continue
            donor = max(("train", "val", "test"), key=lambda s: len(comps_in_split[s]))
            if len(comps_in_split[donor]) <= 1:
                logger.warning(
                    "class %r: too few components to give split %r its own; "
                    "leaving it without this class (splitting one would leak).",
                    cls,
                    empty,
                )
                continue
            smallest = min(comps_in_split[donor], key=len)
            comps_in_split[donor].remove(smallest)
            comps_in_split[empty].append(smallest)

        for split_name in ("train", "val", "test"):
            for comp in comps_in_split[split_name]:
                for gi in comp:
                    split_rows[split_name].append(rows[gi])

    for s in split_rows:
        rng.shuffle(split_rows[s])

    return split_rows


def write_split(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image", "mask", "class", "ita", "ita_group", "source_id", "patient_id"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows_by_split: dict[str, list[dict[str, str]]]) -> None:
    print("\nRebuilt split summary")
    print("=" * 60)
    for split in ("train", "val", "test"):
        rows = rows_by_split[split]
        class_counts: dict[str, int] = defaultdict(int)
        for r in rows:
            class_counts[r["class"]] += 1
        print(f"{split}: total={len(rows)} class_counts={dict(sorted(class_counts.items()))}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild strict train/val/test splits")
    parser.add_argument("--split-dir", type=str, default="data/splits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--near-threshold", type=int, default=6)
    args = parser.parse_args()

    split_dir = Path(args.split_dir)
    rows = load_rows(split_dir)
    rows = enrich_rows(rows)
    class_components = build_components(rows, near_threshold=args.near_threshold)
    rebuilt = assign_components(
        rows=rows,
        class_components=class_components,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    write_split(split_dir / "train.csv", rebuilt["train"])
    write_split(split_dir / "val.csv", rebuilt["val"])
    write_split(split_dir / "test.csv", rebuilt["test"])

    print_summary(rebuilt)
    print(f"Wrote: {split_dir / 'train.csv'}")
    print(f"Wrote: {split_dir / 'val.csv'}")
    print(f"Wrote: {split_dir / 'test.csv'}")


if __name__ == "__main__":
    main()
