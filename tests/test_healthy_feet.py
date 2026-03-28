"""DiaFoot.AI v2 — Healthy Feet Module Tests (Phase 1, Commit 3)."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.data import cleaning
from src.data.healthy_feet import (
    create_empty_masks,
    organize_kaggle_dfu_normal,
    organize_mendeley_normal,
    run_quality_audit_on_healthy,
    validate_healthy_images,
)


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

    def test_match_source_size_when_image_size_none(self, tmp_path: Path) -> None:
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        Image.fromarray(np.zeros((30, 50, 3), dtype=np.uint8)).save(img_dir / "a.png")

        mask_dir = tmp_path / "masks"
        count = create_empty_masks(img_dir, mask_dir, image_size=None)
        assert count == 1

        mask = np.array(Image.open(mask_dir / "a.png"))
        assert mask.shape == (30, 50)


class TestOrganizeHealthySources:
    def test_organize_kaggle_normal_standard_folder(self, tmp_path: Path) -> None:
        kaggle_root = tmp_path / "kaggle"
        normal_dir = kaggle_root / "Normal"
        normal_dir.mkdir(parents=True)
        Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(normal_dir / "n1.jpg")
        Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(normal_dir / "n2.png")

        out_dir = tmp_path / "out"
        count = organize_kaggle_dfu_normal(kaggle_root, out_dir)

        assert count == 2
        copied = sorted(out_dir.iterdir())
        assert len(copied) == 2
        assert copied[0].name.startswith("kaggle_normal_")

    def test_organize_kaggle_normal_fallback_search(self, tmp_path: Path) -> None:
        kaggle_root = tmp_path / "kaggle"
        fallback = kaggle_root / "nested" / "my_normal_examples"
        fallback.mkdir(parents=True)
        Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(fallback / "n1.jpg")

        out_dir = tmp_path / "out"
        count = organize_kaggle_dfu_normal(kaggle_root, out_dir)
        assert count == 1

    def test_organize_kaggle_normal_not_found_returns_zero(self, tmp_path: Path) -> None:
        kaggle_root = tmp_path / "kaggle"
        kaggle_root.mkdir()

        out_dir = tmp_path / "out"
        count = organize_kaggle_dfu_normal(kaggle_root, out_dir)
        assert count == 0

    def test_organize_mendeley_normal_finds_healthy_folder(self, tmp_path: Path) -> None:
        mendeley_root = tmp_path / "mendeley"
        healthy_dir = mendeley_root / "dataset" / "healthy_controls"
        healthy_dir.mkdir(parents=True)
        Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(healthy_dir / "h1.jpg")
        Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(healthy_dir / "h2.bmp")

        out_dir = tmp_path / "out"
        count = organize_mendeley_normal(mendeley_root, out_dir)

        assert count == 2
        copied = sorted(out_dir.iterdir())
        assert copied[0].name.startswith("mendeley_normal_")

    def test_organize_mendeley_normal_not_found_returns_zero(self, tmp_path: Path) -> None:
        mendeley_root = tmp_path / "mendeley"
        mendeley_root.mkdir()

        out_dir = tmp_path / "out"
        count = organize_mendeley_normal(mendeley_root, out_dir)
        assert count == 0


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

    def test_grayscale_and_corrupt_paths(self, tmp_path: Path) -> None:
        img_dir = tmp_path / "mixed"
        img_dir.mkdir()

        # Grayscale image
        gray = Image.fromarray(np.zeros((80, 80), dtype=np.uint8))
        gray.save(img_dir / "gray.png")

        # Corrupt image bytes
        (img_dir / "corrupt.jpg").write_bytes(b"not-an-image")

        # Non-image file should be ignored
        (img_dir / "notes.txt").write_text("hello")

        results = validate_healthy_images(img_dir, min_size=64)
        assert "gray.png" in results["grayscale"]
        assert "corrupt.jpg" in results["corrupt"]


class TestHealthyAuditHook:
    def test_run_quality_audit_on_healthy_invokes_auditor(
        self,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        image_dir = tmp_path / "images"
        image_dir.mkdir()
        output = tmp_path / "report.json"

        calls: list[str] = []

        class _FakeAuditor:
            def __init__(self, _image_dir: str | Path, dataset_name: str) -> None:
                assert dataset_name == "healthy_feet"

            def run_audit(self) -> None:
                calls.append("run")

            def print_summary(self) -> None:
                calls.append("print")

            def save_report(self, output_path: str | Path) -> None:
                calls.append("save")
                Path(output_path).write_text("{}")

        monkeypatch.setattr(cleaning, "DataQualityAuditor", _FakeAuditor)
        run_quality_audit_on_healthy(image_dir=image_dir, output_path=output)

        assert calls == ["run", "print", "save"]
        assert output.exists()
