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
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dedup import canonical_stem
from src.data.external_split import EXTS, resize_with_padding


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
