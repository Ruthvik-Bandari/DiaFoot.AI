"""Build a real external classification split from DFU patch folders.

Default source folders:
- DFU/Patches/Normal(Healthy skin)  -> healthy
- DFU/Patches/Abnormal(Ulcer)       -> dfu
- DFU/Original Images               -> non_dfu (heuristic external lesion set)

The script also removes any exact-content overlap against internal
train/val/test splits using SHA256 hashes.

Usage:
    python scripts/build_external_split_from_patches.py \
        --output data/splits/external.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import cv2
import numpy as np

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def resize_with_padding(img: np.ndarray, target: int = 512) -> np.ndarray:
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        raise ValueError("Invalid empty image")
    scale = target / max(h, w)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target, target, 3), dtype=np.uint8)
    yo, xo = (target - nh) // 2, (target - nw) // 2
    canvas[yo : yo + nh, xo : xo + nw] = resized
    return canvas


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_internal_images(splits_dir: Path) -> set[Path]:
    out: set[Path] = set()
    for name in ("train.csv", "val.csv", "test.csv"):
        p = splits_dir / name
        if not p.exists():
            continue
        with open(p) as f:
            for row in csv.DictReader(f):
                image = row.get("image") or row.get("image_path")
                if image:
                    out.add(Path(image))
    return out


def collect_images(folder: Path) -> list[Path]:
    imgs = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in EXTS]
    imgs.sort()
    return imgs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build external split from patch folders")
    parser.add_argument("--splits-dir", type=str, default="data/splits")
    parser.add_argument("--normal-dir", type=str, default="DFU/Patches/Normal(Healthy skin)")
    parser.add_argument("--ulcer-dir", type=str, default="DFU/Patches/Abnormal(Ulcer)")
    parser.add_argument("--non-dfu-dir", type=str, default="DFU/Original Images")
    parser.add_argument("--processed-root", type=str, default="data/processed_external")
    parser.add_argument("--output", type=str, default="data/splits/external.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    splits_dir = Path(args.splits_dir)
    normal_dir = Path(args.normal_dir)
    ulcer_dir = Path(args.ulcer_dir)
    non_dfu_dir = Path(args.non_dfu_dir)
    processed_root = Path(args.processed_root)
    out_csv = Path(args.output)

    if not normal_dir.exists():
        raise FileNotFoundError(f"Missing normal directory: {normal_dir}")
    if not ulcer_dir.exists():
        raise FileNotFoundError(f"Missing ulcer directory: {ulcer_dir}")
    if not non_dfu_dir.exists():
        raise FileNotFoundError(f"Missing non_dfu directory: {non_dfu_dir}")

    internal_images = load_internal_images(splits_dir)
    internal_hashes: set[str] = set()
    for p in sorted(internal_images):
        if p.exists():
            internal_hashes.add(sha256_file(p))

    normal_imgs = collect_images(normal_dir)
    ulcer_imgs = collect_images(ulcer_dir)
    non_dfu_imgs = collect_images(non_dfu_dir)

    rows: list[dict[str, str]] = []
    removed_overlap = 0
    unreadable = 0

    healthy_img_out = processed_root / "healthy" / "images"
    healthy_mask_out = processed_root / "healthy" / "masks"
    non_dfu_img_out = processed_root / "non_dfu" / "images"
    non_dfu_mask_out = processed_root / "non_dfu" / "masks"
    dfu_img_out = processed_root / "dfu" / "images"
    dfu_mask_out = processed_root / "dfu" / "masks"
    for p in (
        healthy_img_out,
        healthy_mask_out,
        non_dfu_img_out,
        non_dfu_mask_out,
        dfu_img_out,
        dfu_mask_out,
    ):
        p.mkdir(parents=True, exist_ok=True)

    external_seen_hashes: set[str] = set()

    for cls, imgs, source in (
        ("healthy", normal_imgs, "dfu_patches_normal"),
        ("non_dfu", non_dfu_imgs, "dfu_original_images"),
        ("dfu", ulcer_imgs, "dfu_patches_ulcer"),
    ):
        for img in imgs:
            digest = sha256_file(img)
            if digest in internal_hashes:
                removed_overlap += 1
                continue
            if digest in external_seen_hashes:
                continue
            external_seen_hashes.add(digest)

            bgr = cv2.imread(str(img))
            if bgr is None:
                unreadable += 1
                continue
            proc_img = resize_with_padding(bgr, target=512)
            proc_mask = np.zeros((512, 512), dtype=np.uint8)

            patient_id = img.stem.lower().replace(" ", "_").replace("-", "_")

            out_stem = f"ext_{source}_{img.stem}"
            out_stem = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in out_stem)
            if cls == "healthy":
                out_img = healthy_img_out / f"{out_stem}.png"
                out_mask = healthy_mask_out / f"{out_stem}.png"
            elif cls == "non_dfu":
                out_img = non_dfu_img_out / f"{out_stem}.png"
                out_mask = non_dfu_mask_out / f"{out_stem}.png"
            else:
                out_img = dfu_img_out / f"{out_stem}.png"
                out_mask = dfu_mask_out / f"{out_stem}.png"

            cv2.imwrite(str(out_img), proc_img)
            cv2.imwrite(str(out_mask), proc_mask)

            rows.append(
                {
                    "image": str(out_img),
                    "mask": str(out_mask),
                    "class": cls,
                    "ita": "",
                    "ita_group": "unknown",
                    "source_id": source,
                    "patient_id": patient_id,
                }
            )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image", "mask", "class", "ita", "ita_group", "source_id", "patient_id"],
        )
        writer.writeheader()
        writer.writerows(rows)

    healthy_n = sum(1 for r in rows if r["class"] == "healthy")
    non_dfu_n = sum(1 for r in rows if r["class"] == "non_dfu")
    dfu_n = sum(1 for r in rows if r["class"] == "dfu")

    print("\nExternal split built")
    print("=" * 60)
    print(f"Output: {out_csv}")
    print(f"Rows: {len(rows)} (healthy={healthy_n}, non_dfu={non_dfu_n}, dfu={dfu_n})")
    print(f"Removed by content-overlap with internal splits: {removed_overlap}")
    print(f"Unreadable files skipped: {unreadable}")
    print(f"Processed image root: {processed_root}")
    print(
        "Note: masks are synthetic blank masks; "
        "segmentation external metrics are not clinically valid."
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
