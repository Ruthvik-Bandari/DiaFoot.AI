"""DiaFoot.AI v2 — ITA Skin Tone Analysis.

Phase 1, Commit 6: Compute Individual Typology Angle (ITA) for fairness analysis.

ITA measures skin tone objectively from CIELAB color space:
    ITA = arctan((L* - 50) / b*) x (180 / pi)

ITA Categories (Chardon et al., 1991):
    Very Light:  ITA > 55°
    Light:       41° < ITA ≤ 55°
    Intermediate: 28° < ITA ≤ 41°
    Tan:         10° < ITA ≤ 28°
    Brown:       -30° < ITA ≤ 10°
    Dark:        ITA ≤ -30°
"""

from __future__ import annotations

import csv
import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

# ITA category boundaries (degrees)
ITA_CATEGORIES = [
    ("Very Light", 55.0, float("inf")),
    ("Light", 41.0, 55.0),
    ("Intermediate", 28.0, 41.0),
    ("Tan", 10.0, 28.0),
    ("Brown", -30.0, 10.0),
    ("Dark", float("-inf"), -30.0),
]


def rgb_to_lab(rgb_array: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert RGB image array to CIELAB color space.

    Uses the D65 illuminant reference white point.

    Args:
        rgb_array: RGB image as numpy array (H, W, 3), uint8.

    Returns:
        Tuple of (L*, a*, b*) arrays, each (H, W) float64.
    """
    # Normalize to [0, 1]
    rgb = rgb_array.astype(np.float64) / 255.0

    # sRGB to linear RGB
    mask = rgb > 0.04045
    rgb_linear = np.where(mask, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)

    # Linear RGB to XYZ (D65 reference)
    r, g, b = rgb_linear[:, :, 0], rgb_linear[:, :, 1], rgb_linear[:, :, 2]
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

    # Normalize by D65 white point
    x_n, y_n, z_n = x / 0.95047, y / 1.00000, z / 1.08883

    # XYZ to Lab
    epsilon = 0.008856
    kappa = 903.3

    def f(t: np.ndarray) -> np.ndarray:
        return np.where(t > epsilon, np.cbrt(t), (kappa * t + 16.0) / 116.0)

    fx, fy, fz = f(x_n), f(y_n), f(z_n)

    l_star = 116.0 * fy - 16.0
    a_star = 500.0 * (fx - fy)
    b_star = 200.0 * (fy - fz)

    return l_star, a_star, b_star


def compute_ita(image_path: str | Path, mask_path: str | Path | None = None) -> float | None:
    """Compute ITA (Individual Typology Angle) for an image.

    ITA = arctan((L* - 50) / b*) x (180 / pi)

    Computed on skin region only (non-wound area). If a mask is provided,
    the wound region is excluded from the calculation.

    Args:
        image_path: Path to the RGB image.
        mask_path: Optional path to wound mask (wound pixels > 0 are excluded).

    Returns:
        ITA value in degrees, or None if computation fails.
    """
    try:
        img = np.array(Image.open(image_path).convert("RGB"))

        # Compute CIELAB
        l_star, _, b_star = rgb_to_lab(img)

        # Create skin mask (exclude wound region if mask provided)
        if mask_path and Path(mask_path).exists():
            wound_mask = np.array(Image.open(mask_path).convert("L"))
            # Resize wound mask to match image if needed
            if wound_mask.shape != l_star.shape:
                wound_mask = np.array(
                    Image.open(mask_path)
                    .convert("L")
                    .resize(
                        (img.shape[1], img.shape[0]),
                        Image.Resampling.NEAREST,
                    )
                )
            skin_mask = wound_mask == 0  # Non-wound pixels
        else:
            skin_mask = np.ones(l_star.shape, dtype=bool)

        # Get skin-region L* and b* values
        l_vals = l_star[skin_mask]
        b_vals = b_star[skin_mask]

        if len(l_vals) == 0:
            return None

        # Median values (robust to outliers)
        l_median = float(np.median(l_vals))
        b_median = float(np.median(b_vals))

        # Avoid division by zero
        if abs(b_median) < 1e-6:
            b_median = 1e-6

        ita = math.atan2(l_median - 50.0, b_median) * (180.0 / math.pi)
        return round(ita, 2)

    except Exception as e:
        logger.warning("Failed to compute ITA for %s: %s", image_path, e)
        return None


def classify_ita(ita_value: float) -> str:
    """Classify ITA value into skin tone category.

    Args:
        ita_value: ITA angle in degrees.

    Returns:
        Category name (Very Light, Light, Intermediate, Tan, Brown, Dark).
    """
    for name, lower, upper in ITA_CATEGORIES:
        if lower < ita_value <= upper:
            return name
    # Edge case: exactly at boundary
    if ita_value > 55.0:
        return "Very Light"
    return "Dark"


def analyze_dataset_ita(
    image_dir: str | Path,
    mask_dir: str | Path | None = None,
    output_csv: str | Path | None = None,
    max_samples: int | None = None,
) -> dict[str, Any]:
    """Compute ITA scores for all images in a directory.

    Args:
        image_dir: Directory containing images.
        mask_dir: Optional directory with matching masks.
        output_csv: Optional path to save per-image ITA CSV.
        max_samples: Limit number of images to process.

    Returns:
        Dict with ITA distribution statistics.
    """
    image_dir = Path(image_dir)
    images = sorted(f for f in image_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS)

    if max_samples:
        images = images[:max_samples]

    if not images:
        logger.warning("No images found in %s", image_dir)
        return {"total": 0}

    mask_dir_path = Path(mask_dir) if mask_dir else None

    results: list[dict[str, Any]] = []
    for img_path in images:
        # Try to find matching mask
        mask_path = None
        if mask_dir_path and mask_dir_path.exists():
            for ext in [".png", ".jpg", ".jpeg"]:
                candidate = mask_dir_path / f"{img_path.stem}{ext}"
                if candidate.exists():
                    mask_path = candidate
                    break

        ita = compute_ita(img_path, mask_path)
        if ita is not None:
            category = classify_ita(ita)
            results.append(
                {
                    "filename": img_path.name,
                    "ita": ita,
                    "category": category,
                }
            )

    # Save CSV
    if output_csv and results:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "ita", "category"])
            writer.writeheader()
            writer.writerows(results)
        logger.info("Saved ITA scores to %s", output_csv)

    # Compute distribution
    ita_values = [r["ita"] for r in results]
    categories = [r["category"] for r in results]
    cat_counts = {name: categories.count(name) for name, _, _ in ITA_CATEGORIES}

    report = {
        "total_analyzed": len(results),
        "failed": len(images) - len(results),
        "ita_stats": {
            "mean": round(np.mean(ita_values), 2) if ita_values else 0,
            "median": round(np.median(ita_values), 2) if ita_values else 0,
            "std": round(np.std(ita_values), 2) if ita_values else 0,
            "min": round(min(ita_values), 2) if ita_values else 0,
            "max": round(max(ita_values), 2) if ita_values else 0,
        },
        "category_distribution": cat_counts,
    }
    return report


def run_ita_analysis(
    data_root: str | Path = "data/raw",
    output_dir: str | Path = "data/metadata",
) -> dict[str, Any]:
    """Run ITA analysis across all dataset categories.

    Args:
        data_root: Root data directory.
        output_dir: Directory for output reports.

    Returns:
        Combined ITA report.
    """
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    combined: dict[str, Any] = {"datasets": {}}

    # Analyze each dataset
    datasets = [
        ("fuseg", data_root / "dfu" / "fuseg", "train/images", "train/labels"),
        ("healthy", data_root / "healthy", "images", "masks"),
        ("non_dfu", data_root / "non_dfu", "images", "masks"),
    ]

    all_ita_rows: list[dict[str, Any]] = []

    for name, base_dir, img_sub, mask_sub in datasets:
        img_dir = base_dir / img_sub if (base_dir / img_sub).exists() else base_dir
        mask_dir = base_dir / mask_sub if (base_dir / mask_sub).exists() else None

        if not img_dir.exists():
            logger.warning("Skipping %s: %s not found", name, img_dir)
            continue

        logger.info("Computing ITA for %s (%s)...", name, img_dir)
        csv_path = output_dir / f"ita_scores_{name}.csv"
        report = analyze_dataset_ita(
            img_dir,
            mask_dir=mask_dir,
            output_csv=csv_path,
        )
        combined["datasets"][name] = report

        # Collect for combined CSV
        if csv_path.exists():
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row["dataset"] = name
                    all_ita_rows.append(row)

    # Save combined CSV
    if all_ita_rows:
        combined_csv = output_dir / "ita_scores.csv"
        with open(combined_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["dataset", "filename", "ita", "category"])
            writer.writeheader()
            writer.writerows(all_ita_rows)
        logger.info("Combined ITA scores: %s", combined_csv)

    # Save JSON report
    report_path = output_dir / "ita_report.json"
    with open(report_path, "w") as f:
        json.dump(combined, f, indent=2)
    logger.info("ITA report saved to %s", report_path)

    return combined
