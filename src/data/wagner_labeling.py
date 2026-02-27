"""DiaFoot.AI v2 — Wagner Grade Labeling & Label Quality Audit.

Phase 1, Commit 5: Label quality checks + Wagner grade annotation infrastructure.

Cleanlab's full segmentation audit requires model predictions (Phase 3).
For now, we perform structural label quality checks that don't need a model:
- Mask validity (binary, correct dimensions, not corrupt)
- Mask coverage statistics (detect anomalous wound sizes)
- Empty mask detection (images with zero wound pixels)
- Mask-image pairing verification

Wagner grading is set up as metadata CSV for DFU images.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}


@dataclass
class MaskQualityResult:
    """Result of mask quality analysis for a single image."""

    filename: str
    is_valid: bool
    width: int = 0
    height: int = 0
    unique_values: int = 0
    wound_coverage_pct: float = 0.0
    issue: str = ""


def check_mask_quality(
    mask_path: Path, expected_size: tuple[int, int] | None = None
) -> MaskQualityResult:
    """Check quality of a single segmentation mask.

    Validates:
    - File can be opened
    - Mask is grayscale/binary
    - Dimensions match expected size (if provided)
    - Wound coverage is within reasonable range

    Args:
        mask_path: Path to mask image.
        expected_size: Expected (width, height) to match against.

    Returns:
        MaskQualityResult with findings.
    """
    result = MaskQualityResult(filename=mask_path.name, is_valid=True)

    try:
        with Image.open(mask_path) as img:
            mask = np.array(img.convert("L"))
            result.height, result.width = mask.shape
            result.unique_values = len(np.unique(mask))

            # Check dimensions match
            if expected_size and (result.width, result.height) != expected_size:
                result.issue = (
                    f"size_mismatch: expected {expected_size},"
                    f" got ({result.width}, {result.height})"
                )
                result.is_valid = False

            # Compute wound coverage
            wound_pixels = (mask > 0).sum()
            total_pixels = mask.size
            result.wound_coverage_pct = (
                round(wound_pixels / total_pixels * 100, 2) if total_pixels > 0 else 0.0
            )

            # Flag anomalous masks
            if result.unique_values > 10:
                result.issue = f"not_binary: {result.unique_values} unique values (expected 2-3)"
                result.is_valid = False
            elif result.wound_coverage_pct > 90:
                result.issue = (
                    f"coverage_too_high: {result.wound_coverage_pct}% (likely inverted mask)"
                )
                result.is_valid = False

    except Exception as e:
        result.is_valid = False
        result.issue = f"corrupt: {e}"

    return result


def audit_masks(
    mask_dir: str | Path,
    image_dir: str | Path | None = None,
) -> dict:
    """Run structural quality audit on all masks in a directory.

    Args:
        mask_dir: Directory containing mask files.
        image_dir: Optional paired image directory for size matching.

    Returns:
        Dict with audit results and statistics.
    """
    mask_dir = Path(mask_dir)
    masks = sorted(f for f in mask_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS)

    if not masks:
        logger.warning("No masks found in %s", mask_dir)
        return {"total": 0}

    # Build image lookup for size matching
    expected_sizes: dict[str, tuple[int, int]] = {}
    if image_dir:
        image_dir = Path(image_dir)
        for img_path in image_dir.iterdir():
            if img_path.suffix.lower() in IMAGE_EXTS:
                try:
                    with Image.open(img_path) as img:
                        expected_sizes[img_path.stem] = img.size
                except Exception:  # noqa: S110
                    pass

    results: list[MaskQualityResult] = []
    for mask_path in masks:
        expected = expected_sizes.get(mask_path.stem)
        result = check_mask_quality(mask_path, expected_size=expected)
        results.append(result)

    # Aggregate statistics
    valid = [r for r in results if r.is_valid]
    invalid = [r for r in results if not r.is_valid]
    coverages = [r.wound_coverage_pct for r in valid]
    empty_masks = [r for r in valid if r.wound_coverage_pct == 0]

    report = {
        "total_masks": len(results),
        "valid": len(valid),
        "invalid": len(invalid),
        "empty_masks": len(empty_masks),
        "coverage_stats": {},
        "issues": {},
    }

    if coverages:
        report["coverage_stats"] = {
            "mean_pct": round(np.mean(coverages), 2),
            "median_pct": round(np.median(coverages), 2),
            "min_pct": round(min(coverages), 2),
            "max_pct": round(max(coverages), 2),
            "std_pct": round(np.std(coverages), 2),
        }

    # Categorize issues
    issue_types: dict[str, list[str]] = {}
    for r in invalid:
        category = r.issue.split(":")[0] if r.issue else "unknown"
        if category not in issue_types:
            issue_types[category] = []
        issue_types[category].append(r.filename)
    report["issues"] = {k: {"count": len(v), "examples": v[:10]} for k, v in issue_types.items()}

    return report


def create_wagner_grade_csv(
    dfu_image_dir: str | Path,
    output_path: str | Path,
) -> int:
    """Create a CSV template for Wagner grade annotation.

    Generates a CSV with columns: filename, wagner_grade, notes
    where wagner_grade is initially empty (to be filled by annotator).

    Wagner Grades:
        0: Pre-ulcerative / at-risk (intact skin, callus, redness)
        1: Superficial ulcer (partial/full thickness, no tendon/bone)
        2: Deep ulcer (extends to tendon, joint capsule, bone)
        3: Deep with abscess (osteomyelitis, abscess)
        4: Partial gangrene (localized necrotic tissue)
        5: Extensive gangrene (entire foot involvement)

    Args:
        dfu_image_dir: Directory with DFU images.
        output_path: Where to save the CSV.

    Returns:
        Number of entries created.
    """
    dfu_image_dir = Path(dfu_image_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    images = sorted(
        f.name
        for f in dfu_image_dir.rglob("*")
        if f.suffix.lower() in IMAGE_EXTS
        and "label" not in str(f).lower()
        and "mask" not in str(f).lower()
    )

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "wagner_grade", "confidence", "notes"])
        for img_name in images:
            writer.writerow([img_name, "", "", ""])

    logger.info("Created Wagner grade CSV with %d entries: %s", len(images), output_path)
    return len(images)


def run_label_audit(
    data_root: str | Path = "data/raw",
    output_dir: str | Path = "data/metadata",
) -> dict:
    """Run label quality audit across all dataset categories.

    Args:
        data_root: Root data directory containing dfu/, healthy/, non_dfu/.
        output_dir: Directory for output reports.

    Returns:
        Combined audit results.
    """
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    combined_report: dict = {"datasets": {}}

    # Audit DFU masks
    for dataset_name in ["fuseg", "azh"]:
        mask_candidates = [
            data_root / "dfu" / dataset_name / "train" / "labels",
            data_root / "dfu" / dataset_name / "masks",
            data_root / "dfu" / dataset_name / "validation" / "labels",
        ]
        image_candidates = [
            data_root / "dfu" / dataset_name / "train" / "images",
            data_root / "dfu" / dataset_name / "images",
            data_root / "dfu" / dataset_name / "validation" / "images",
        ]

        for mask_dir, img_dir in zip(mask_candidates, image_candidates, strict=False):
            if mask_dir.exists():
                logger.info("Auditing masks: %s", mask_dir)
                report = audit_masks(mask_dir, img_dir if img_dir.exists() else None)
                split_name = mask_dir.parent.name
                combined_report["datasets"][f"{dataset_name}_{split_name}"] = report

    # Audit non-DFU masks
    non_dfu_masks = data_root / "non_dfu" / "masks"
    if non_dfu_masks.exists():
        logger.info("Auditing non-DFU masks: %s", non_dfu_masks)
        report = audit_masks(non_dfu_masks, data_root / "non_dfu" / "images")
        combined_report["datasets"]["non_dfu"] = report

    # Audit healthy masks (should all be empty)
    healthy_masks = data_root / "healthy" / "masks"
    if healthy_masks.exists():
        logger.info("Auditing healthy masks: %s", healthy_masks)
        report = audit_masks(healthy_masks)
        combined_report["datasets"]["healthy"] = report

    # Create Wagner grade CSV for DFU images
    dfu_dir = data_root / "dfu"
    if dfu_dir.exists():
        wagner_path = output_dir / "wagner_grades.csv"
        n = create_wagner_grade_csv(dfu_dir, wagner_path)
        combined_report["wagner_grade_entries"] = n

    # Save combined report
    report_path = output_dir / "label_issues.json"
    with open(report_path, "w") as f:
        json.dump(combined_report, f, indent=2, default=str)
    logger.info("Label audit report saved to %s", report_path)

    return combined_report
