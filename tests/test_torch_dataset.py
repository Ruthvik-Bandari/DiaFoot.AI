"""Tests for DFUDataset loading and sample formatting."""

from __future__ import annotations

import csv
from typing import TYPE_CHECKING

import cv2
import numpy as np
import pytest
import torch

from src.data.torch_dataset import CLASS_TO_IDX, DFUDataset

if TYPE_CHECKING:
    from pathlib import Path


class _NumpyMaskTransform:
    """Simple transform that returns torch image and numpy mask."""

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> dict[str, torch.Tensor | np.ndarray]:
        img = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        return {"image": img, "mask": mask}


class _TensorMaskTransform:
    """Simple transform that returns torch image and torch mask."""

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> dict[str, torch.Tensor]:
        img = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        m = torch.from_numpy(mask)
        return {"image": img, "mask": m}


def _write_rgb_png(path: Path, shape: tuple[int, int, int] = (16, 16, 3)) -> None:
    arr = np.random.randint(0, 255, size=shape, dtype=np.uint8)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(str(path), bgr)
    assert ok


def _write_mask_png(path: Path, shape: tuple[int, int] = (16, 16)) -> None:
    arr = np.zeros(shape, dtype=np.uint8)
    arr[2:10, 3:12] = 255
    ok = cv2.imwrite(str(path), arr)
    assert ok


def _write_split_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = sorted({k for row in rows for k in row})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class TestDFUDataset:
    def test_missing_csv_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            DFUDataset(tmp_path / "missing.csv")

    def test_len_and_basic_item_with_metadata(self, tmp_path: Path) -> None:
        image_path = tmp_path / "image.png"
        mask_path = tmp_path / "mask.png"
        split_csv = tmp_path / "split.csv"

        _write_rgb_png(image_path)
        _write_mask_png(mask_path)
        _write_split_csv(
            split_csv,
            [
                {
                    "image_path": str(image_path),
                    "mask_path": str(mask_path),
                    "class": "dfu",
                    "ita_group": "Brown",
                }
            ],
        )

        ds = DFUDataset(split_csv=split_csv, return_metadata=True)
        assert len(ds) == 1

        sample = ds[0]
        assert sample["label"] == CLASS_TO_IDX["dfu"]
        image = sample["image"]
        mask = sample["mask"]
        assert isinstance(image, torch.Tensor)
        assert isinstance(mask, torch.Tensor)
        assert image.shape == (3, 16, 16)
        assert mask.shape == (16, 16)
        assert int(mask.max().item()) == 1

        metadata = sample["metadata"]
        assert isinstance(metadata, dict)
        assert metadata["class_name"] == "dfu"
        assert metadata["ita_category"] == "Brown"
        assert metadata["filename"] == image_path.name

    def test_missing_mask_defaults_to_empty(self, tmp_path: Path) -> None:
        image_path = tmp_path / "image.png"
        split_csv = tmp_path / "split.csv"

        _write_rgb_png(image_path)
        _write_split_csv(
            split_csv,
            [
                {
                    "image": str(image_path),
                    "class": "healthy",
                }
            ],
        )

        ds = DFUDataset(split_csv=split_csv)
        sample = ds[0]
        mask = sample["mask"]
        assert isinstance(mask, torch.Tensor)
        assert int(mask.sum().item()) == 0
        assert sample["label"] == CLASS_TO_IDX["healthy"]

    def test_transform_path_numpy_mask(self, tmp_path: Path) -> None:
        image_path = tmp_path / "image.png"
        mask_path = tmp_path / "mask.png"
        split_csv = tmp_path / "split.csv"

        _write_rgb_png(image_path)
        _write_mask_png(mask_path)
        _write_split_csv(
            split_csv,
            [
                {
                    "image_path": str(image_path),
                    "mask_path": str(mask_path),
                    "class": "non_dfu",
                }
            ],
        )

        ds = DFUDataset(split_csv=split_csv, transform=_NumpyMaskTransform())
        sample = ds[0]
        assert isinstance(sample["image"], torch.Tensor)
        assert isinstance(sample["mask"], torch.Tensor)
        assert sample["mask"].dtype == torch.long
        assert sample["label"] == CLASS_TO_IDX["non_dfu"]

    def test_transform_path_tensor_mask(self, tmp_path: Path) -> None:
        image_path = tmp_path / "image.png"
        mask_path = tmp_path / "mask.png"
        split_csv = tmp_path / "split.csv"

        _write_rgb_png(image_path)
        _write_mask_png(mask_path)
        _write_split_csv(
            split_csv,
            [
                {
                    "image_path": str(image_path),
                    "mask_path": str(mask_path),
                    "class": "dfu",
                }
            ],
        )

        ds = DFUDataset(split_csv=split_csv, transform=_TensorMaskTransform())
        sample = ds[0]
        assert isinstance(sample["mask"], torch.Tensor)
        assert sample["mask"].dtype == torch.long

    def test_missing_image_column_raises(self, tmp_path: Path) -> None:
        split_csv = tmp_path / "split.csv"
        _write_split_csv(split_csv, [{"class": "dfu"}])

        ds = DFUDataset(split_csv=split_csv)
        with pytest.raises(RuntimeError, match="Missing image path column"):
            _ = ds[0]

    def test_unreadable_image_raises(self, tmp_path: Path) -> None:
        split_csv = tmp_path / "split.csv"
        _write_split_csv(
            split_csv,
            [
                {
                    "image_path": str(tmp_path / "not_found.png"),
                    "class": "dfu",
                }
            ],
        )

        ds = DFUDataset(split_csv=split_csv)
        with pytest.raises(RuntimeError, match="Failed to load image"):
            _ = ds[0]
