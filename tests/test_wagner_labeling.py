"""DiaFoot.AI v2 — Label Quality & Wagner Grade Tests (Phase 1, Commit 5)."""

import csv
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.data.wagner_labeling import (
    audit_masks,
    check_mask_quality,
    create_wagner_grade_csv,
)


@pytest.fixture
def sample_masks(tmp_path: Path) -> Path:
    """Create sample mask images."""
    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()
    # Normal binary mask (wound region)
    mask = np.zeros((128, 128), dtype=np.uint8)
    mask[30:80, 30:80] = 255
    Image.fromarray(mask).save(mask_dir / "mask_001.png")
    # Empty mask (no wound)
    empty = np.zeros((128, 128), dtype=np.uint8)
    Image.fromarray(empty).save(mask_dir / "mask_002.png")
    # Small wound
    small = np.zeros((128, 128), dtype=np.uint8)
    small[60:65, 60:65] = 255
    Image.fromarray(small).save(mask_dir / "mask_003.png")
    return mask_dir


@pytest.fixture
def sample_images(tmp_path: Path) -> Path:
    """Create sample DFU images for Wagner grading."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for i in range(5):
        img = Image.fromarray(np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8))
        img.save(img_dir / f"dfu_{i:03d}.png")
    return img_dir


class TestCheckMaskQuality:
    def test_valid_mask(self, sample_masks: Path) -> None:
        result = check_mask_quality(sample_masks / "mask_001.png")
        assert result.is_valid
        assert result.wound_coverage_pct > 0

    def test_empty_mask(self, sample_masks: Path) -> None:
        result = check_mask_quality(sample_masks / "mask_002.png")
        assert result.is_valid
        assert result.wound_coverage_pct == 0.0

    def test_nonexistent_mask(self, tmp_path: Path) -> None:
        result = check_mask_quality(tmp_path / "nonexistent.png")
        assert not result.is_valid
        assert "corrupt" in result.issue


class TestAuditMasks:
    def test_audit_returns_stats(self, sample_masks: Path) -> None:
        report = audit_masks(sample_masks)
        assert report["total_masks"] == 3
        assert report["valid"] == 3
        assert report["empty_masks"] == 1
        assert "coverage_stats" in report

    def test_audit_empty_dir(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        report = audit_masks(empty)
        assert report["total"] == 0


class TestCreateWagnerGradeCsv:
    def test_creates_csv(self, sample_images: Path, tmp_path: Path) -> None:
        output = tmp_path / "wagner.csv"
        n = create_wagner_grade_csv(sample_images, output)
        assert n == 5
        assert output.exists()
        with open(output) as f:
            reader = csv.reader(f)
            header = next(reader)
            assert "wagner_grade" in header
            rows = list(reader)
            assert len(rows) == 5
