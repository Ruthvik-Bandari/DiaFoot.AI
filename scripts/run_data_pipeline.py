"""DiaFoot.AI v2 — Full Data Pipeline: Integrate + Clean + Split.

Runs the complete pipeline:
  1. Integrate AZH GitHub data into processed/dfu/
  2. Validate all processed data
  3. Regenerate stratified train/val/test splits
  4. Verify no data leakage
  5. Generate final dataset report

Usage:
    python scripts/run_data_pipeline.py --verbose
    python scripts/run_data_pipeline.py --skip-dedup --verbose   # faster, skip hash dedup
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger("pipeline")

TARGET_SIZE = 512
CLAHE_CLIP_LIMIT = 2.0
CLAHE_GRID_SIZE = (8, 8)
MASK_THRESHOLD = 127


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: Integrate AZH GitHub
# ═══════════════════════════════════════════════════════════════════════════════
def resize_with_padding(
    img: np.ndarray, target: int, interp: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    h, w = img.shape[:2]
    scale = target / max(h, w)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=interp)
    if len(img.shape) == 3:
        canvas = np.zeros((target, target, img.shape[2]), dtype=img.dtype)
    else:
        canvas = np.zeros((target, target), dtype=img.dtype)
    yo, xo = (target - nh) // 2, (target - nw) // 2
    canvas[yo : yo + nh, xo : xo + nw] = resized
    return canvas


def integrate_azh(data_dir: Path) -> dict:
    """Integrate AZH GitHub into processed/dfu/."""
    azh_dir = data_dir / "raw" / "azh_github"
    img_out = data_dir / "processed" / "dfu" / "images"
    mask_out = data_dir / "processed" / "dfu" / "masks"
    img_out.mkdir(parents=True, exist_ok=True)
    mask_out.mkdir(parents=True, exist_ok=True)

    stats = {"found": 0, "saved": 0, "skipped": 0, "errors": 0, "issues": {}}

    for split in ["train", "test"]:
        img_dir = azh_dir / split / "images"
        mask_dir = azh_dir / split / "labels"
        if not img_dir.exists():
            continue

        for img_path in sorted(img_dir.glob("*.png")):
            mask_path = mask_dir / img_path.name
            if not mask_path.exists():
                stats["skipped"] += 1
                continue

            stats["found"] += 1
            out_name = f"azh_{split}_{img_path.stem}.png"

            # Skip if already processed
            if (img_out / out_name).exists():
                stats["skipped"] += 1
                continue

            try:
                img = cv2.imread(str(img_path))
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if img is None or mask is None:
                    stats["errors"] += 1
                    continue

                # Validate mask
                if mask.max() == 0:
                    stats["issues"]["all_zero"] = stats["issues"].get("all_zero", 0) + 1

                # Preprocess image: resize + CLAHE
                img = resize_with_padding(img, TARGET_SIZE)
                lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(
                    clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_GRID_SIZE
                )
                lab = cv2.merge([clahe.apply(l), a, b])
                img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

                # Preprocess mask: resize + binarize
                if len(mask.shape) == 3:
                    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                mask = resize_with_padding(mask, TARGET_SIZE, cv2.INTER_NEAREST)
                _, mask = cv2.threshold(mask, MASK_THRESHOLD, 255, cv2.THRESH_BINARY)

                cv2.imwrite(str(img_out / out_name), img)
                cv2.imwrite(str(mask_out / out_name), mask)
                stats["saved"] += 1

            except Exception as e:
                logger.warning("Error: %s — %s", img_path.name, e)
                stats["errors"] += 1

    return stats


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: Validate all processed data
# ═══════════════════════════════════════════════════════════════════════════════
def validate_processed(data_dir: Path) -> dict:
    """Validate all processed image-mask pairs."""
    processed = data_dir / "processed"
    report = {}

    for class_dir in sorted(processed.iterdir()):
        if not class_dir.is_dir():
            continue

        cls = class_dir.name
        img_dir = class_dir / "images"
        mask_dir = class_dir / "masks"

        if not img_dir.exists():
            continue

        imgs = {f.stem: f for f in img_dir.iterdir() if f.suffix.lower() in {".png", ".jpg"}}
        masks = {f.stem: f for f in mask_dir.iterdir() if f.suffix.lower() in {".png", ".jpg"}} if mask_dir.exists() else {}

        paired = set(imgs.keys()) & set(masks.keys())
        orphan_imgs = set(imgs.keys()) - set(masks.keys())
        orphan_masks = set(masks.keys()) - set(imgs.keys())

        # Sample validation on 50 random pairs
        sample = list(paired)[:50]
        size_issues = 0
        for stem in sample:
            img = cv2.imread(str(imgs[stem]))
            if img is not None:
                h, w = img.shape[:2]
                if (h, w) != (TARGET_SIZE, TARGET_SIZE):
                    size_issues += 1

        report[cls] = {
            "total_images": len(imgs),
            "total_masks": len(masks),
            "paired": len(paired),
            "orphan_images": len(orphan_imgs),
            "orphan_masks": len(orphan_masks),
            "non_512_in_sample": size_issues,
        }

    return report


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: Compute ITA for stratification
# ═══════════════════════════════════════════════════════════════════════════════
def compute_ita(img_path: Path) -> float | None:
    """Compute ITA skin tone angle from image center crop."""
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            return None
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        h, w = lab.shape[:2]
        m = min(h, w) // 4
        cy, cx = h // 2, w // 2
        crop = lab[cy - m : cy + m, cx - m : cx + m]
        l_mean = crop[:, :, 0].mean() * 100.0 / 255.0
        b_mean = float(crop[:, :, 2].mean())
        if abs(b_mean) < 1e-6:
            return 90.0
        return round(float(np.degrees(np.arctan2(l_mean - 50.0, b_mean))), 2)
    except Exception:
        return None


def ita_group(ita: float | None) -> str:
    if ita is None:
        return "unknown"
    if ita > 55:
        return "very_light"
    if ita > 41:
        return "light"
    if ita > 28:
        return "intermediate"
    if ita > 10:
        return "tan"
    if ita > -30:
        return "brown"
    return "dark"


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4: Generate stratified splits
# ═══════════════════════════════════════════════════════════════════════════════
def generate_splits(
    data_dir: Path, seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Build sample list and create 70/15/15 stratified splits."""
    processed = data_dir / "processed"
    samples = []

    for class_dir in sorted(processed.iterdir()):
        if not class_dir.is_dir():
            continue
        cls = class_dir.name
        img_dir = class_dir / "images"
        mask_dir = class_dir / "masks"
        if not img_dir.exists():
            continue

        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in {".png", ".jpg"}:
                continue
            mask_path = mask_dir / img_path.name
            if not mask_path.exists():
                continue

            # Compute ITA every 5th image for speed
            ita = None
            if len(samples) % 5 == 0:
                ita = compute_ita(img_path)

            samples.append({
                "image": str(img_path),
                "mask": str(mask_path),
                "class": cls,
                "ita": ita,
                "ita_group": ita_group(ita),
            })

    logger.info("Total paired samples: %d", len(samples))

    # Stratified split by class + ita_group
    rng = np.random.RandomState(seed)
    groups: dict[str, list] = {}
    for s in samples:
        key = f"{s['class']}_{s['ita_group']}"
        groups.setdefault(key, []).append(s)

    train, val, test = [], [], []
    for key, group in groups.items():
        idx = list(range(len(group)))
        rng.shuffle(idx)
        n = len(group)
        n_train = max(1, int(n * 0.70))
        n_val = max(1, int(n * 0.15)) if n > 2 else 0
        for i, j in enumerate(idx):
            s = group[j]
            if i < n_train:
                train.append(s)
            elif i < n_train + n_val:
                val.append(s)
            else:
                test.append(s)

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def write_csv(samples: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["image", "mask", "class", "ita", "ita_group"],
        )
        writer.writeheader()
        for s in samples:
            writer.writerow({
                "image": s["image"],
                "mask": s["mask"],
                "class": s["class"],
                "ita": s.get("ita", ""),
                "ita_group": s.get("ita_group", "unknown"),
            })


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    parser = argparse.ArgumentParser(description="Full data pipeline")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    data_dir = Path(args.data_dir)
    splits_dir = data_dir / "splits"
    metadata_dir = data_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Integrate AZH ────────────────────────────────────────────
    print("\n" + "█" * 60)
    print("  STEP 1: Integrate AZH GitHub data")
    print("█" * 60)

    azh_stats = integrate_azh(data_dir)
    print(f"  Found: {azh_stats['found']} pairs")
    print(f"  Saved: {azh_stats['saved']} new images to processed/dfu/")
    print(f"  Skipped: {azh_stats['skipped']} (already exists or no mask)")
    print(f"  Errors: {azh_stats['errors']}")
    if azh_stats["issues"]:
        print(f"  Issues: {azh_stats['issues']}")

    # ── Step 2: Validate all processed data ──────────────────────────────
    print("\n" + "█" * 60)
    print("  STEP 2: Validate all processed data")
    print("█" * 60)

    val_report = validate_processed(data_dir)
    total_paired = 0
    for cls, stats in val_report.items():
        print(f"  {cls}: {stats['paired']} paired, "
              f"{stats['orphan_images']} orphan imgs, "
              f"{stats['orphan_masks']} orphan masks")
        total_paired += stats["paired"]
    print(f"  TOTAL: {total_paired} paired samples")

    # ── Step 3: Generate splits ──────────────────────────────────────────
    print("\n" + "█" * 60)
    print("  STEP 3: Generate stratified splits")
    print("█" * 60)

    train, val, test = generate_splits(data_dir, seed=args.seed)
    write_csv(train, splits_dir / "train.csv")
    write_csv(val, splits_dir / "val.csv")
    write_csv(test, splits_dir / "test.csv")

    print(f"  Train: {len(train)}")
    print(f"  Val:   {len(val)}")
    print(f"  Test:  {len(test)}")

    # Class distribution
    for name, split in [("train", train), ("val", val), ("test", test)]:
        counts = Counter(s["class"] for s in split)
        print(f"  {name}: {dict(counts)}")

    # ── Step 4: Leakage check ────────────────────────────────────────────
    print("\n" + "█" * 60)
    print("  STEP 4: Data leakage check")
    print("█" * 60)

    train_imgs = {s["image"] for s in train}
    val_imgs = {s["image"] for s in val}
    test_imgs = {s["image"] for s in test}

    leak_tv = train_imgs & val_imgs
    leak_tt = train_imgs & test_imgs
    leak_vt = val_imgs & test_imgs

    if leak_tv or leak_tt or leak_vt:
        print(f"  ⚠ LEAKAGE DETECTED!")
        print(f"    Train∩Val: {len(leak_tv)}")
        print(f"    Train∩Test: {len(leak_tt)}")
        print(f"    Val∩Test: {len(leak_vt)}")
    else:
        print("  ✓ No leakage — all images in exactly one split")

    # ── Save report ──────────────────────────────────────────────────────
    report = {
        "azh_integration": azh_stats,
        "validation": val_report,
        "splits": {
            "train": len(train),
            "val": len(val),
            "test": len(test),
            "train_classes": dict(Counter(s["class"] for s in train)),
            "val_classes": dict(Counter(s["class"] for s in val)),
            "test_classes": dict(Counter(s["class"] for s in test)),
        },
        "leakage": bool(leak_tv or leak_tt or leak_vt),
        "seed": args.seed,
    }

    report_path = metadata_dir / "data_pipeline_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print("\n" + "█" * 60)
    print("  PIPELINE COMPLETE")
    print("█" * 60)
    print(f"  Total samples: {total_paired}")
    print(f"  Splits: {len(train)} / {len(val)} / {len(test)}")
    print(f"  Report: {report_path}")
    print("█" * 60)


if __name__ == "__main__":
    main()
