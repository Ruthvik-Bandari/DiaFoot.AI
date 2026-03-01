"""DiaFoot.AI v2 — nnU-Net v2 Wrapper.

Phase 2, Commit 10: Self-configuring nnU-Net v2 for multi-class segmentation.
nnU-Net auto-configures architecture, preprocessing, and training based on data.

Classes: background, healthy_skin, non_dfu_wound, dfu_wound
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class NNUNetConfig:
    """Configuration for nnU-Net v2 experiment.

    Args:
        dataset_name: Name identifier for nnU-Net dataset.
        num_classes: Number of segmentation classes (including background).
        image_size: Target image dimensions.
        modality: Image modality mapping.
        class_names: Mapping of class index to name.
    """

    dataset_name: str = "Dataset001_DiaFootAI"
    num_classes: int = 4
    image_size: tuple[int, int] = (512, 512)
    modality: dict[str, str] = field(default_factory=lambda: {"0": "RGB"})
    class_names: dict[str, str] = field(
        default_factory=lambda: {
            "0": "background",
            "1": "healthy_skin",
            "2": "non_dfu_wound",
            "3": "dfu_wound",
        }
    )


def generate_dataset_json(
    config: NNUNetConfig,
    output_dir: str | Path,
    num_training: int = 0,
    num_test: int = 0,
) -> Path:
    """Generate nnU-Net dataset.json descriptor.

    This file tells nnU-Net about the dataset structure,
    classes, and modalities.

    Args:
        config: nnU-Net configuration.
        output_dir: Where to write dataset.json.
        num_training: Number of training cases.
        num_test: Number of test cases.

    Returns:
        Path to generated dataset.json.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_json: dict[str, Any] = {
        "channel_names": config.modality,
        "labels": config.class_names,
        "numTraining": num_training,
        "numTest": num_test,
        "file_ending": ".png",
    }

    json_path = output_dir / "dataset.json"
    with open(json_path, "w") as f:
        json.dump(dataset_json, f, indent=2)

    logger.info("Generated dataset.json at %s", json_path)
    return json_path


def prepare_nnunet_directory(
    processed_dir: str | Path,
    nnunet_raw_dir: str | Path,
    config: NNUNetConfig | None = None,
) -> Path:
    """Prepare directory structure expected by nnU-Net v2.

    nnU-Net expects:
        nnUNet_raw/DatasetXXX_Name/
        ├── dataset.json
        ├── imagesTr/       (training images: case_0000.png)
        └── labelsTr/       (training labels: case.png)

    Args:
        processed_dir: Our preprocessed data directory.
        nnunet_raw_dir: nnU-Net raw data root.
        config: nnU-Net configuration.

    Returns:
        Path to the prepared nnU-Net dataset directory.
    """
    config = config or NNUNetConfig()
    processed_dir = Path(processed_dir)
    dataset_dir = Path(nnunet_raw_dir) / config.dataset_name

    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)

    logger.info(
        "nnU-Net directory prepared at %s. "
        "Copy preprocessed images/masks here in nnU-Net naming convention.",
        dataset_dir,
    )

    return dataset_dir
