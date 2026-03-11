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
import random
import re
from collections import defaultdict
from pathlib import Path

import cv2

_AUG_TAIL = re.compile(
    r"(_aug\d+|_flip|_hflip|_vflip|_rot\d+|_copy\d+|_dup\d+)$",
    flags=re.IGNORECASE,
)


def canonical_stem(path_str: str) -> str:
    stem = Path(path_str).stem.lower().replace("-", "_")
    return _AUG_TAIL.sub("", stem)


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

    # keep numeric suffixes for healthy/non_dfu (prevents collapsing to one patient)
    # e.g. male_normal_1383, wound_main_0010
    return stem


def _dhash(path: Path, hash_size: int = 8) -> int | None:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    resized = cv2.resize(img, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    value = 0
    for bit in diff.flatten():
        value = (value << 1) | int(bool(bit))
    return value


def _hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


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
        source_id = (row.get("source_id") or infer_source_id(class_name, p)).strip().lower()
        patient_id = (row.get("patient_id") or infer_patient_id(class_name, p)).strip().lower()

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

        local_to_global = {li: gi for li, gi in enumerate(indices)}
        hash_to_locals: dict[int, list[int]] = defaultdict(list)

        for li, gi in local_to_global.items():
            hv = _dhash(Path(rows[gi]["image"]))
            if hv is None:
                continue
            hash_to_locals[hv].append(li)

        # same-hash images -> same component
        for locs in hash_to_locals.values():
            root = locs[0]
            for li in locs[1:]:
                uf.union(root, li)

        # near-hash linking using class-aware top-16-bit buckets
        prefix_to_hashes: dict[int, list[int]] = defaultdict(list)
        for hv in hash_to_locals.keys():
            prefix = (hv >> 48) & 0xFFFF
            prefix_to_hashes[prefix].append(hv)

        hash_rep_local = {hv: locs[0] for hv, locs in hash_to_locals.items()}

        for hashes in prefix_to_hashes.values():
            m = len(hashes)
            for i in range(m):
                h1 = hashes[i]
                for j in range(i + 1, m):
                    h2 = hashes[j]
                    if _hamming(h1, h2) <= near_threshold:
                        uf.union(hash_rep_local[h1], hash_rep_local[h2])

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

    split_rows = {"train": [], "val": [], "test": []}

    for cls, components in class_components.items():
        rng.shuffle(components)
        total = sum(len(c) for c in components)
        targets = {
            "train": int(round(total * train_ratio)),
            "val": int(round(total * val_ratio)),
            "test": max(0, total - int(round(total * train_ratio)) - int(round(total * val_ratio))),
        }
        current = {"train": 0, "val": 0, "test": 0}

        for comp in components:
            size = len(comp)
            deficits = {
                s: targets[s] - current[s]
                for s in ("train", "val", "test")
            }
            # prioritize split with biggest deficit, then smallest current size
            chosen = sorted(
                ("train", "val", "test"),
                key=lambda s: (deficits[s], -current[s]),
                reverse=True,
            )[0]
            for gi in comp:
                split_rows[chosen].append(rows[gi])
            current[chosen] += size

        # lightweight class-presence fix: ensure each split has at least one sample if possible
        per_split_cls = {
            s: sum(1 for r in split_rows[s] if r["class"] == cls)
            for s in ("train", "val", "test")
        }
        empty_splits = [s for s, n in per_split_cls.items() if n == 0]
        if empty_splits and len(components) >= 3:
            # move one smallest component for this class from largest split to each empty split
            cls_in_split = {
                s: [r for r in split_rows[s] if r["class"] == cls]
                for s in ("train", "val", "test")
            }
            donor = max(("train", "val", "test"), key=lambda s: len(cls_in_split[s]))
            donor_rows = cls_in_split[donor]
            for es in empty_splits:
                if len(donor_rows) <= 1:
                    break
                moved = donor_rows.pop()
                split_rows[donor].remove(moved)
                split_rows[es].append(moved)

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
        class_counts = defaultdict(int)
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
