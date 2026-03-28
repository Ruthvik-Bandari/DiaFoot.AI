"""DiaFoot.AI v2 — Preprocessing Tests (Phase 1, Commit 7)."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.data.preprocessing import (
    apply_clahe,
    binarize_mask,
    preprocess_dataset,
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

    def test_missing_input_returns_false(self, tmp_path: Path) -> None:
        output_path = tmp_path / "output.png"
        result = preprocess_image(tmp_path / "does_not_exist.jpg", output_path)
        assert result is False

    def test_disable_clahe_still_writes_image(self, tmp_path: Path) -> None:
        img = np.random.randint(0, 255, (120, 80, 3), dtype=np.uint8)
        input_path = tmp_path / "input.jpg"
        cv2.imwrite(str(input_path), img)
        output_path = tmp_path / "output_no_clahe.png"

        result = preprocess_image(input_path, output_path, apply_clahe_flag=False)
        assert result is True
        loaded = cv2.imread(str(output_path))
        assert loaded is not None
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

    def test_missing_mask_returns_false(self, tmp_path: Path) -> None:
        output_path = tmp_path / "mask_out.png"
        result = preprocess_mask(tmp_path / "does_not_exist.png", output_path)
        assert result is False


class TestPreprocessDataset:
    def test_preprocess_dataset_with_matching_masks(self, tmp_path: Path) -> None:
        image_dir = tmp_path / "images"
        mask_dir = tmp_path / "masks"
        out_image_dir = tmp_path / "out_images"
        out_mask_dir = tmp_path / "out_masks"
        image_dir.mkdir()
        mask_dir.mkdir()

        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        cv2.imwrite(str(image_dir / "case_1.jpg"), img)

        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[16:48, 16:48] = 255
        cv2.imwrite(str(mask_dir / "case_1.png"), mask)

        stats = preprocess_dataset(
            image_dir=image_dir,
            mask_dir=mask_dir,
            output_image_dir=out_image_dir,
            output_mask_dir=out_mask_dir,
            target_size=(128, 128),
        )

        assert stats == {"success": 1, "failed": 0}
        assert (out_image_dir / "case_1.png").exists()
        assert (out_mask_dir / "case_1.png").exists()

    def test_preprocess_dataset_uses_wound_main_to_mask_fallback(self, tmp_path: Path) -> None:
        image_dir = tmp_path / "images"
        mask_dir = tmp_path / "masks"
        out_image_dir = tmp_path / "out_images"
        out_mask_dir = tmp_path / "out_masks"
        image_dir.mkdir()
        mask_dir.mkdir()

        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        cv2.imwrite(str(image_dir / "wound_main_001.jpg"), img)

        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[8:40, 8:40] = 255
        cv2.imwrite(str(mask_dir / "wound_mask_001.jpg"), mask)

        stats = preprocess_dataset(
            image_dir=image_dir,
            mask_dir=mask_dir,
            output_image_dir=out_image_dir,
            output_mask_dir=out_mask_dir,
            target_size=(128, 128),
        )

        assert stats == {"success": 1, "failed": 0}
        out_mask = cv2.imread(str(out_mask_dir / "wound_main_001.png"), cv2.IMREAD_GRAYSCALE)
        assert out_mask is not None
        assert int(out_mask.sum()) > 0

    def test_preprocess_dataset_creates_empty_mask_when_missing(self, tmp_path: Path) -> None:
        image_dir = tmp_path / "images"
        mask_dir = tmp_path / "masks"
        out_image_dir = tmp_path / "out_images"
        out_mask_dir = tmp_path / "out_masks"
        image_dir.mkdir()
        mask_dir.mkdir()

        img = np.random.randint(0, 255, (50, 90, 3), dtype=np.uint8)
        cv2.imwrite(str(image_dir / "sample_a.jpg"), img)

        stats = preprocess_dataset(
            image_dir=image_dir,
            mask_dir=mask_dir,
            output_image_dir=out_image_dir,
            output_mask_dir=out_mask_dir,
            target_size=(64, 64),
        )

        assert stats == {"success": 1, "failed": 0}
        out_mask = cv2.imread(str(out_mask_dir / "sample_a.png"), cv2.IMREAD_GRAYSCALE)
        assert out_mask is not None
        assert out_mask.shape == (64, 64)
        assert int(out_mask.sum()) == 0

    def test_preprocess_dataset_without_masks(self, tmp_path: Path) -> None:
        image_dir = tmp_path / "images"
        out_image_dir = tmp_path / "out_images"
        image_dir.mkdir()

        img = np.random.randint(0, 255, (80, 80, 3), dtype=np.uint8)
        cv2.imwrite(str(image_dir / "healthy_1.jpg"), img)

        stats = preprocess_dataset(
            image_dir=image_dir,
            mask_dir=None,
            output_image_dir=out_image_dir,
            output_mask_dir=None,
            target_size=(96, 96),
            apply_clahe_flag=False,
        )

        assert stats == {"success": 1, "failed": 0}
        assert (out_image_dir / "healthy_1.png").exists()

    def test_preprocess_dataset_counts_failed_image(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        image_dir = tmp_path / "images"
        out_image_dir = tmp_path / "out_images"
        image_dir.mkdir()

        img = np.random.randint(0, 255, (80, 80, 3), dtype=np.uint8)
        cv2.imwrite(str(image_dir / "bad_case.jpg"), img)

        monkeypatch.setattr(
            "src.data.preprocessing.preprocess_image",
            lambda *_args, **_kwargs: False,
        )

        stats = preprocess_dataset(
            image_dir=image_dir,
            mask_dir=None,
            output_image_dir=out_image_dir,
            output_mask_dir=None,
            target_size=(96, 96),
        )

        assert stats == {"success": 0, "failed": 1}
