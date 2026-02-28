"""DiaFoot.AI v2 — Stratified Train/Val/Test Split Creator.

Phase 1, Commit 7: Create doubly-stratified splits by ITA + class label.
"""

from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def load_ita_scores(ita_csv: str | Path) -> dict[str, str]:
    """Load ITA category mapping from CSV.

    Args:
        ita_csv: Path to ita_scores.csv.

    Returns:
        Dict mapping filename -> ITA category.
    """
    mapping: dict[str, str] = {}
    csv_path = Path(ita_csv)
    if not csv_path.exists():
        logger.warning("ITA CSV not found: %s", csv_path)
        return mapping

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[row["filename"]] = row.get("category", "Unknown")
    return mapping


def create_stratified_splits(
    processed_dir: str | Path,
    splits_dir: str | Path,
    ita_csv: str | Path = "data/metadata/ita_scores.csv",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict:
    """Create doubly-stratified train/val/test splits.

    Stratified by:
    1. Class label (healthy / non_dfu / dfu)
    2. ITA skin tone category (within each class)

    Args:
        processed_dir: Directory with preprocessed data per category.
        splits_dir: Output directory for split files.
        ita_csv: Path to combined ITA scores CSV.
        train_ratio: Training set proportion.
        val_ratio: Validation set proportion.
        test_ratio: Test set proportion.
        seed: Random seed.

    Returns:
        Dict with split statistics.
    """
    processed_dir = Path(processed_dir)
    splits_dir = Path(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(seed)

    # Load ITA mapping
    ita_mapping = load_ita_scores(ita_csv)

    # Collect all images by class
    categories = ["dfu", "healthy", "non_dfu"]
    all_entries: list[dict] = []

    for category in categories:
        img_dir = processed_dir / category / "images"
        if not img_dir.exists():
            logger.warning("Category dir not found: %s", img_dir)
            continue

        for img_path in sorted(img_dir.glob("*.png")):
            ita_cat = ita_mapping.get(img_path.name, "Unknown")
            all_entries.append(
                {
                    "filename": img_path.name,
                    "class": category,
                    "ita_category": ita_cat,
                    "image_path": str(img_path),
                    "mask_path": str(processed_dir / category / "masks" / img_path.name),
                }
            )

    if not all_entries:
        logger.error("No images found in %s", processed_dir)
        return {"error": "no images found"}

    # Group by (class, ita_category) for stratification
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for entry in all_entries:
        key = (entry["class"], entry["ita_category"])
        groups[key].append(entry)

    train_entries: list[dict] = []
    val_entries: list[dict] = []
    test_entries: list[dict] = []

    for (_cls, _ita), entries in sorted(groups.items()):
        rng.shuffle(entries)
        n = len(entries)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio)) if n > 2 else 0
        # Remaining goes to test
        train_entries.extend(entries[:n_train])
        val_entries.extend(entries[n_train : n_train + n_val])
        test_entries.extend(entries[n_train + n_val :])

    # Save split CSVs
    fieldnames = [
        "filename",
        "class",
        "ita_category",
        "image_path",
        "mask_path",
    ]
    for split_name, split_entries in [
        ("train", train_entries),
        ("val", val_entries),
        ("test", test_entries),
    ]:
        csv_path = splits_dir / f"{split_name}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(split_entries)
        logger.info("Split %s: %d entries -> %s", split_name, len(split_entries), csv_path)

    # Compute statistics
    stats = {
        "total": len(all_entries),
        "train": len(train_entries),
        "val": len(val_entries),
        "test": len(test_entries),
        "class_distribution": {},
        "ita_distribution": {},
    }

    for split_name, entries in [
        ("train", train_entries),
        ("val", val_entries),
        ("test", test_entries),
    ]:
        class_counts: dict[str, int] = defaultdict(int)
        ita_counts: dict[str, int] = defaultdict(int)
        for e in entries:
            class_counts[e["class"]] += 1
            ita_counts[e["ita_category"]] += 1
        stats["class_distribution"][split_name] = dict(class_counts)
        stats["ita_distribution"][split_name] = dict(ita_counts)

    # Save stats
    stats_path = splits_dir / "split_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    return stats
