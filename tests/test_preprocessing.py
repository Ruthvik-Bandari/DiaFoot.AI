"""DiaFoot.AI v2 — Preprocessing Tests (Phase 1, Commit 7)."""

from pathlib import Path

import cv2
import numpy as np

from src.data.preprocessing import (
    apply_clahe,
    binarize_mask,
    preprocess_image,
    preprocess_mask,
    resize_with_padding,
)


class TestResizeWithPadding:
    def test_square_image(self) -> None:
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = resize_with_padding(img, (512, 512))
        assert result.shape == (512, 512, 3)

    def test_rectangular_image(self) -> None:
        img = np.random.randint(0, 255, (200, 100, 3), dtype=np.uint8)
        result = resize_with_padding(img, (512, 512))
        assert result.shape == (512, 512, 3)

    def test_mask_resize(self) -> None:
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 20:80] = 255
        result = resize_with_padding(mask, (512, 512))
        assert result.shape == (512, 512)
        assert set(np.unique(result)).issubset({0, 255})


class TestClahe:
    def test_output_shape(self) -> None:
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result = apply_clahe(img)
        assert result.shape == img.shape


class TestBinarizeMask:
    def test_clean_binary(self) -> None:
        mask = np.array([0, 50, 128, 200, 255], dtype=np.uint8).reshape(1, 5)
        result = binarize_mask(mask)
        expected = np.array([0, 0, 255, 255, 255], dtype=np.uint8).reshape(1, 5)
        np.testing.assert_array_equal(result, expected)

    def test_color_mask(self) -> None:
        mask = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result = binarize_mask(mask)
        assert len(result.shape) == 2


class TestPreprocessImage:
    def test_creates_output(self, tmp_path: Path) -> None:
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        input_path = tmp_path / "input.jpg"
        cv2.imwrite(str(input_path), img)
        output_path = tmp_path / "output.png"
        result = preprocess_image(input_path, output_path)
        assert result is True
        assert output_path.exists()
        loaded = cv2.imread(str(output_path))
        assert loaded.shape == (512, 512, 3)


class TestPreprocessMask:
    def test_binarizes_and_resizes(self, tmp_path: Path) -> None:
        mask = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        input_path = tmp_path / "mask.jpg"
        cv2.imwrite(str(input_path), mask)
        output_path = tmp_path / "mask_out.png"
        result = preprocess_mask(input_path, output_path)
        assert result is True
        loaded = cv2.imread(str(output_path), cv2.IMREAD_GRAYSCALE)
        assert loaded.shape == (512, 512)
        assert set(np.unique(loaded)).issubset({0, 255})
