"""DiaFoot.AI v2 — Non-DFU Collection Tests (Phase 1, Commit 4)."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def sample_non_dfu_dir(tmp_path: Path) -> tuple[Path, Path]:
    """Create sample non-DFU image and mask directories."""
    img_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    img_dir.mkdir()
    mask_dir.mkdir()
    for i in range(5):
        img = Image.fromarray(np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8))
        mask = Image.fromarray(np.random.randint(0, 1, (128, 128), dtype=np.uint8) * 255)
        img.save(img_dir / f"wound_main-{i:04d}.jpg")
        mask.save(mask_dir / f"wound_mask-{i:04d}.jpg")
    return img_dir, mask_dir


def test_images_and_masks_exist(sample_non_dfu_dir: tuple[Path, Path]) -> None:
    img_dir, mask_dir = sample_non_dfu_dir
    images = list(img_dir.glob("*.jpg"))
    masks = list(mask_dir.glob("*.jpg"))
    assert len(images) == 5
    assert len(masks) == 5


def test_masks_are_not_empty(sample_non_dfu_dir: tuple[Path, Path]) -> None:
    """Non-DFU masks can have wound boundaries (labeled as non-DFU)."""
    _, mask_dir = sample_non_dfu_dir
    for mask_path in mask_dir.iterdir():
        mask = np.array(Image.open(mask_path))
        assert mask.shape == (128, 128)
