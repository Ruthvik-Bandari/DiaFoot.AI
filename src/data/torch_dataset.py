"""DiaFoot.AI v2 — PyTorch Dataset for Multi-Class DFU.

Phase 2, Commit 8-9: Dataset class that loads from split CSVs.
Supports classification (3-class), segmentation (binary mask), and staging.
"""

from __future__ import annotations

import csv
from pathlib import Path

import albumentations as A  # noqa: N812, TC002
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# Class label mapping
CLASS_TO_IDX = {"healthy": 0, "non_dfu": 1, "dfu": 2}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}


class DFUDataset(Dataset):
    """Multi-task dataset for DFU classification and segmentation.

    Loads image-mask pairs from split CSV files produced by the
    preprocessing pipeline (Commit 7).

    Each sample returns:
        - image: (3, H, W) float tensor (normalized)
        - mask: (H, W) long tensor (binary: 0=background, 1=wound)
        - label: int (0=healthy, 1=non_dfu, 2=dfu)
        - metadata: dict with filename, ita_category, etc.

    Args:
        split_csv: Path to split CSV (train.csv, val.csv, test.csv).
        transform: Albumentations transform pipeline.
        return_metadata: Whether to include metadata dict.
    """

    def __init__(
        self,
        split_csv: str | Path,
        transform: A.Compose | None = None,
        return_metadata: bool = False,
    ) -> None:
        """Initialize dataset from split CSV."""
        self.transform = transform
        self.return_metadata = return_metadata
        self.samples: list[dict] = []

        csv_path = Path(split_csv)
        if not csv_path.exists():
            msg = f"Split CSV not found: {csv_path}"
            raise FileNotFoundError(msg)

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append(row)

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int | dict]:
        """Load and return a single sample.

        Returns:
            Dict with keys: image, mask, label, and optionally metadata.
        """
        sample = self.samples[idx]

        image_path = sample.get("image_path") or sample.get("image")
        if not image_path:
            msg = "Missing image path column in split CSV (expected image_path or image)"
            raise RuntimeError(msg)

        # Load image (BGR -> RGB)
        image = cv2.imread(image_path)
        if image is None:
            msg = f"Failed to load image: {image_path}"
            raise RuntimeError(msg)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load mask
        mask_path = sample.get("mask_path") or sample.get("mask") or ""
        if mask_path and Path(mask_path).exists():
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
            # Binarize: any nonzero pixel = wound
            mask = (mask > 0).astype(np.uint8)
        else:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)

        # Apply augmentation (mask-aware)
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image_tensor = transformed["image"]  # (3, H, W) float
            mask_out = transformed["mask"]
            if isinstance(mask_out, torch.Tensor):
                mask_tensor = mask_out.long()
            else:
                mask_tensor = torch.from_numpy(mask_out).long()
        else:
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask_tensor = torch.from_numpy(mask).long()

        # Class label
        class_name = sample.get("class", "healthy")
        label = CLASS_TO_IDX.get(class_name, 0)

        result: dict[str, torch.Tensor | int | dict] = {
            "image": image_tensor,
            "mask": mask_tensor,
            "label": label,
        }

        if self.return_metadata:
            filename = sample.get("filename")
            if not filename and image_path:
                filename = Path(image_path).name
            result["metadata"] = {
                "filename": filename or "",
                "ita_category": sample.get("ita_category") or sample.get("ita_group", "Unknown"),
                "class_name": class_name,
            }

        return result
