"""DiaFoot.AI v2 — Healthy Feet Module Tests (Phase 1, Commit 3)."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.data.healthy_feet import create_empty_masks, validate_healthy_images


@pytest.fixture
def sample_healthy_dir(tmp_path: Path) -> Path:
    """Create sample healthy foot images."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for i in range(5):
        img = Image.fromarray(np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8))
        img.save(img_dir / f"healthy_{i:03d}.png")
    return img_dir


class TestCreateEmptyMasks:
    def test_creates_masks(self, sample_healthy_dir: Path, tmp_path: Path) -> None:
        mask_dir = tmp_path / "masks"
        count = create_empty_masks(sample_healthy_dir, mask_dir)
        assert count == 5
        assert mask_dir.exists()

    def test_masks_are_all_zero(self, sample_healthy_dir: Path, tmp_path: Path) -> None:
        mask_dir = tmp_path / "masks"
        create_empty_masks(sample_healthy_dir, mask_dir, image_size=(64, 64))
        for mask_path in mask_dir.iterdir():
            mask = np.array(Image.open(mask_path))
            assert mask.sum() == 0, "Healthy mask should be all zeros"
            assert mask.shape == (64, 64)


class TestValidateHealthyImages:
    def test_all_valid(self, sample_healthy_dir: Path) -> None:
        results = validate_healthy_images(sample_healthy_dir)
        assert len(results["valid"]) == 5
        assert len(results["corrupt"]) == 0

    def test_too_small(self, tmp_path: Path) -> None:
        img_dir = tmp_path / "small"
        img_dir.mkdir()
        tiny = Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8))
        tiny.save(img_dir / "tiny.png")
        results = validate_healthy_images(img_dir, min_size=64)
        assert len(results["too_small"]) == 1
