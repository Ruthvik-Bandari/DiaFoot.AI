"""DiaFoot.AI v2 — ITA Analysis Tests (Phase 1, Commit 6)."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.data.ita_analysis import classify_ita, compute_ita, rgb_to_lab


@pytest.fixture
def light_skin_image(tmp_path: Path) -> Path:
    """Create a light-toned image."""
    img = np.full((64, 64, 3), [220, 190, 170], dtype=np.uint8)
    path = tmp_path / "light.jpg"
    Image.fromarray(img).save(path)
    return path


@pytest.fixture
def dark_skin_image(tmp_path: Path) -> Path:
    """Create a dark-toned image."""
    img = np.full((64, 64, 3), [80, 50, 30], dtype=np.uint8)
    path = tmp_path / "dark.jpg"
    Image.fromarray(img).save(path)
    return path


class TestRgbToLab:
    def test_white_pixel(self) -> None:
        white = np.full((1, 1, 3), 255, dtype=np.uint8)
        l_star, _a_star, _b_star = rgb_to_lab(white)
        assert l_star[0, 0] > 99  # L* should be ~100 for white

    def test_black_pixel(self) -> None:
        black = np.zeros((1, 1, 3), dtype=np.uint8)
        l_star, _a_star, _b_star = rgb_to_lab(black)
        assert l_star[0, 0] < 1  # L* should be ~0 for black


class TestComputeIta:
    def test_light_skin(self, light_skin_image: Path) -> None:
        ita = compute_ita(light_skin_image)
        assert ita is not None
        assert ita > 20  # Light skin should have high ITA

    def test_dark_skin(self, dark_skin_image: Path) -> None:
        ita = compute_ita(dark_skin_image)
        assert ita is not None
        assert ita < 30  # Dark skin should have low ITA

    def test_nonexistent_image(self, tmp_path: Path) -> None:
        result = compute_ita(tmp_path / "nonexistent.jpg")
        assert result is None


class TestClassifyIta:
    def test_very_light(self) -> None:
        assert classify_ita(60.0) == "Very Light"

    def test_intermediate(self) -> None:
        assert classify_ita(35.0) == "Intermediate"

    def test_dark(self) -> None:
        assert classify_ita(-40.0) == "Dark"

    def test_brown(self) -> None:
        assert classify_ita(5.0) == "Brown"
