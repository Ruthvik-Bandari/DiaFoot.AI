"""DiaFoot.AI v2 — Robustness Tests (Phase 4, Commit 22)."""

import numpy as np

from src.evaluation.robustness import (
    apply_brightness_shift,
    apply_contrast_reduction,
    apply_gaussian_blur,
    apply_gaussian_noise,
    apply_jpeg_compression,
)


class TestDegradations:
    def test_blur_preserves_shape(self) -> None:
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result = apply_gaussian_blur(img, severity=3)
        assert result.shape == img.shape

    def test_noise_preserves_range(self) -> None:
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result = apply_gaussian_noise(img, severity=3)
        assert result.min() >= 0
        assert result.max() <= 255

    def test_brightness_shift(self) -> None:
        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        darker = apply_brightness_shift(img, severity=1)
        assert darker.mean() < img.mean()

    def test_contrast_reduction(self) -> None:
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        reduced = apply_contrast_reduction(img, severity=3)
        assert reduced.std() < img.std()

    def test_jpeg_compression(self) -> None:
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        compressed = apply_jpeg_compression(img, severity=5)
        assert compressed.shape == img.shape

    def test_all_severities(self) -> None:
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        for severity in range(1, 6):
            result = apply_gaussian_blur(img, severity=severity)
            assert result.shape == img.shape
