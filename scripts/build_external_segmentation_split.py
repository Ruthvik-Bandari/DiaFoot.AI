"""Build an external DFU segmentation split with real masks.

Default source:
- data/raw/dfu/fuseg/validation/images
- data/raw/dfu/fuseg/validation/labels

Writes resized image/mask pairs to `data/processed_external_seg/dfu/*`
and a CSV at `data/splits/external_segmentation.csv`.

To reduce leakage, samples are excluded when their canonical stem appears in
internal train/val/test split image stems.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import cv2
import numpy as np

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
_AUG_TAIL = re.compile(r"(_aug\d+|_flip|_hflip|_vflip|_rot\d+|_copy\d+|_dup\d+)$", re.IGNORECASE)


def canonical_stem(path_str: str) -> str:
    stem = Path(path_str).stem.lower().replace("-", "_")
    return _AUG_TAIL.sub("", stem)


def resize_with_padding(
    img: np.ndarray, target: int = 512, interp: int = cv2.INTER_AREA
) -> np.ndarray:
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError("Invalid image size")
    scale = target / max(h, w)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=interp)

    if img.ndim == 2:
        canvas = np.zeros((target, target), dtype=img.dtype)
    else:
        canvas = np.zeros((target, target, img.shape[2]), dtype=img.dtype)

    yo, xo = (target - nh) // 2, (target - nw) // 2
    canvas[yo : yo + nh, xo : xo + nw] = resized
    return canvas


def load_internal_canonical_stems(splits_dir: Path) -> set[str]:
    stems: set[str] = set()
    for split_name in ("train.csv", "val.csv", "test.csv"):
        p = splits_dir / split_name
        if not p.exists():
            continue
        with open(p) as f:
            for row in csv.DictReader(f):
                image = row.get("image") or row.get("image_path")
                if image:
                    stems.add(canonical_stem(image))
    return stems


def main() -> None:
    parser = argparse.ArgumentParser(description="Build external DFU segmentation split")
    parser.add_argument("--splits-dir", type=str, default="data/splits")
    parser.add_argument("--image-dir", type=str, default="data/raw/dfu/fuseg/validation/images")
    parser.add_argument("--mask-dir", type=str, default="data/raw/dfu/fuseg/validation/labels")
    parser.add_argument("--processed-root", type=str, default="data/processed_external_seg")
    parser.add_argument("--output", type=str, default="data/splits/external_segmentation.csv")
    args = parser.parse_args()

    splits_dir = Path(args.splits_dir)
    image_dir = Path(args.image_dir)
    mask_dir = Path(args.mask_dir)
    processed_root = Path(args.processed_root)
    output_csv = Path(args.output)

    if not image_dir.exists():
        raise FileNotFoundError(f"Missing image dir: {image_dir}")
    if not mask_dir.exists():
        raise FileNotFoundError(f"Missing mask dir: {mask_dir}")

    internal_stems = load_internal_canonical_stems(splits_dir)

    img_out_dir = processed_root / "dfu" / "images"
    mask_out_dir = processed_root / "dfu" / "masks"
    img_out_dir.mkdir(parents=True, exist_ok=True)
    mask_out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    skipped_overlap = 0
    skipped_no_mask = 0
    unreadable = 0

    for img in sorted(image_dir.iterdir()):
        if not img.is_file() or img.suffix.lower() not in EXTS:
            continue

        stem = img.stem
        stem_c = canonical_stem(stem)
        if stem_c in internal_stems:
            skipped_overlap += 1
            continue

        mask = mask_dir / img.name
        if not mask.exists():
            alt = list(mask_dir.glob(stem + ".*"))
            if not alt:
                skipped_no_mask += 1
                continue
            mask = alt[0]

        bgr = cv2.imread(str(img))
        msk = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)
        if bgr is None or msk is None:
            unreadable += 1
            continue

        bgr = resize_with_padding(bgr, target=512, interp=cv2.INTER_AREA)
        msk = resize_with_padding(msk, target=512, interp=cv2.INTER_NEAREST)
        msk = (msk > 0).astype(np.uint8) * 255

        out_stem = f"ext_fuseg_val_{stem}"
        out_img = img_out_dir / f"{out_stem}.png"
        out_msk = mask_out_dir / f"{out_stem}.png"

        cv2.imwrite(str(out_img), bgr)
        cv2.imwrite(str(out_msk), msk)

        rows.append(
            {
                "image": str(out_img),
                "mask": str(out_msk),
                "class": "dfu",
                "ita": "",
                "ita_group": "unknown",
                "source_id": "fuseg_validation_external",
                "patient_id": stem_c,
            }
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image", "mask", "class", "ita", "ita_group", "source_id", "patient_id"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\nExternal segmentation split built")
    print("=" * 60)
    print(f"Output: {output_csv}")
    print(f"Rows: {len(rows)}")
    print(f"Skipped by canonical overlap with internal splits: {skipped_overlap}")
    print(f"Skipped (no mask): {skipped_no_mask}")
    print(f"Unreadable skipped: {unreadable}")
    print(f"Processed root: {processed_root}")
    print("=" * 60)


if __name__ == "__main__":
    main()
