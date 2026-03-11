"""DiaFoot.AI v2 — Split Leakage Audit utilities.

Provides reusable checks for train/val/test leakage:
1) Exact path overlap
2) Exact file-content overlap (SHA256)
3) Canonical filename ID overlap (renamed copies)
4) Perceptual near-duplicates via dHash + Hamming distance
"""

from __future__ import annotations

import csv
import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

_AUGMENT_TOKENS_PATTERN = re.compile(
    r"(_aug\d+|_flip|_vflip|_hflip|_rot\d+|_copy\d+|_dup\d+)$",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class Sample:
    """Minimal sample representation used by leakage checks."""

    split: str
    path: str
    class_name: str


def canonical_sample_id(path: str | Path) -> str:
    """Normalize filename to a canonical ID for split overlap checks.

    Removes common augmentation/copy suffixes from stem, then lowercases.
    """
    stem = Path(path).stem.lower()
    stem = _AUGMENT_TOKENS_PATTERN.sub("", stem)
    return stem


def _sha256(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha256(data).hexdigest()


def _dhash(path: Path, hash_size: int = 8) -> int | None:
    """Compute 64-bit dHash for an image.

    Returns None when image cannot be read.
    """
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


def _intersections_by_key(samples: list[Sample], key_fn: Callable[[Sample], str]) -> dict[str, int]:
    by_split: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    for s in samples:
        by_split.setdefault(s.split, set()).add(str(key_fn(s)))

    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    out: dict[str, int] = {}
    for a, b in pairs:
        out[f"{a}_x_{b}"] = len(by_split.get(a, set()) & by_split.get(b, set()))
    return out


def audit_samples_for_leakage(
    train_samples: list[dict[str, Any]],
    val_samples: list[dict[str, Any]],
    test_samples: list[dict[str, Any]],
    near_duplicate_threshold: int = 6,
    max_near_duplicate_pairs: int = 50,
) -> dict[str, Any]:
    """Audit leakage between splits using multiple overlap checks."""
    merged: list[Sample] = []
    for split, rows in (("train", train_samples), ("val", val_samples), ("test", test_samples)):
        for row in rows:
            merged.append(
                Sample(
                    split=split,
                    path=str(row["image"]),
                    class_name=str(row.get("class", "unknown")),
                )
            )

    # 1) Path overlap
    path_overlap = _intersections_by_key(merged, lambda s: s.path)

    # 2) Canonical filename overlap
    canonical_overlap = _intersections_by_key(merged, lambda s: canonical_sample_id(s.path))

    # 3) Exact content overlap via SHA256
    hashes: dict[str, str] = {}
    hash_samples: list[tuple[Sample, str]] = []
    for s in merged:
        p = Path(s.path)
        if not p.exists():
            continue
        h = hashes.get(s.path)
        if h is None:
            h = _sha256(p)
            hashes[s.path] = h
        hash_samples.append((s, h))
    by_split_hash: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    for s, h in hash_samples:
        by_split_hash.setdefault(s.split, set()).add(h)

    content_overlap = {
        "train_x_val": len(by_split_hash["train"] & by_split_hash["val"]),
        "train_x_test": len(by_split_hash["train"] & by_split_hash["test"]),
        "val_x_test": len(by_split_hash["val"] & by_split_hash["test"]),
    }

    # 4) Perceptual near duplicates (class-aware + dHash prefix buckets)
    # Bucket key: (class_name, split, top_16_bits)
    dhash_cache: dict[str, int] = {}
    hashed_entries: list[tuple[Sample, int]] = []
    for s in merged:
        p = Path(s.path)
        if not p.exists():
            continue
        hv = dhash_cache.get(s.path)
        if hv is None:
            computed = _dhash(p)
            if computed is None:
                continue
            hv = computed
            dhash_cache[s.path] = hv
        hashed_entries.append((s, hv))

    split_pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    near_dup_counts: dict[str, int] = {"train_x_val": 0, "train_x_test": 0, "val_x_test": 0}
    near_dup_examples: list[dict[str, Any]] = []

    by_split_class_prefix: dict[tuple[str, str, int], list[tuple[Sample, int]]] = {}
    for s, hv in hashed_entries:
        prefix = (hv >> 48) & 0xFFFF
        key = (s.split, s.class_name, prefix)
        by_split_class_prefix.setdefault(key, []).append((s, hv))

    all_classes = {s.class_name for s, _ in hashed_entries}
    prefixes_by_split_class: dict[tuple[str, str], set[int]] = {}
    for split_name, class_name, prefix in by_split_class_prefix:
        prefixes_by_split_class.setdefault((split_name, class_name), set()).add(prefix)

    for split_a, split_b in split_pairs:
        pair_name = f"{split_a}_x_{split_b}"
        for class_name in all_classes:
            common_prefixes = prefixes_by_split_class.get((split_a, class_name), set()) & (
                prefixes_by_split_class.get((split_b, class_name), set())
            )
            for prefix in common_prefixes:
                bucket_a = by_split_class_prefix.get((split_a, class_name, prefix))
                bucket_b = by_split_class_prefix.get((split_b, class_name, prefix))
                if not bucket_a or not bucket_b:
                    continue
                for sample_a, hash_a in bucket_a:
                    for sample_b, hash_b in bucket_b:
                        dist = _hamming(hash_a, hash_b)
                        if dist <= near_duplicate_threshold:
                            near_dup_counts[pair_name] += 1
                            if len(near_dup_examples) < max_near_duplicate_pairs:
                                near_dup_examples.append(
                                    {
                                        "pair": pair_name,
                                        "class": class_name,
                                        "distance": dist,
                                        "a": sample_a.path,
                                        "b": sample_b.path,
                                    }
                                )

    leakage_flags = {
        "path_overlap": any(v > 0 for v in path_overlap.values()),
        "canonical_overlap": any(v > 0 for v in canonical_overlap.values()),
        "content_overlap": any(v > 0 for v in content_overlap.values()),
        "near_duplicates": any(v > 0 for v in near_dup_counts.values()),
    }

    return {
        "counts": {
            "train": len(train_samples),
            "val": len(val_samples),
            "test": len(test_samples),
        },
        "path_overlap": path_overlap,
        "canonical_overlap": canonical_overlap,
        "content_overlap": content_overlap,
        "near_duplicates": {
            "threshold": near_duplicate_threshold,
            "counts": near_dup_counts,
            "examples": near_dup_examples,
        },
        "leakage_flags": leakage_flags,
        "has_any_leakage": any(leakage_flags.values()),
    }


def load_split_csv(path: str | Path) -> list[dict[str, str]]:
    """Load split CSV with at least an 'image' column."""
    csv_path = Path(path)
    rows: list[dict[str, str]] = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows
