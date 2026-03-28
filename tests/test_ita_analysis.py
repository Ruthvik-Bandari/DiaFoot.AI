"""DiaFoot.AI v2 — ITA Analysis Tests (Phase 1, Commit 6)."""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.data.ita_analysis import (
    analyze_dataset_ita,
    classify_ita,
    compute_ita,
    rgb_to_lab,
    run_ita_analysis,
)


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

    def test_mask_exclusion_and_resize(self, tmp_path: Path) -> None:
        img = np.full((64, 64, 3), [190, 150, 120], dtype=np.uint8)
        image_path = tmp_path / "skin.jpg"
        Image.fromarray(img).save(image_path)

        # Deliberately different size to exercise resize branch
        mask = np.zeros((16, 16), dtype=np.uint8)
        mask[4:12, 4:12] = 255
        mask_path = tmp_path / "mask.png"
        Image.fromarray(mask).save(mask_path)

        ita = compute_ita(image_path, mask_path)
        assert ita is not None

    def test_all_wound_pixels_returns_none(self, tmp_path: Path) -> None:
        img = np.full((32, 32, 3), [180, 140, 110], dtype=np.uint8)
        image_path = tmp_path / "skin.jpg"
        Image.fromarray(img).save(image_path)

        # Entire image marked as wound => no non-wound pixels
        mask = np.full((32, 32), 255, dtype=np.uint8)
        mask_path = tmp_path / "mask.png"
        Image.fromarray(mask).save(mask_path)

        ita = compute_ita(image_path, mask_path)
        assert ita is None


class TestClassifyIta:
    def test_very_light(self) -> None:
        assert classify_ita(60.0) == "Very Light"

    def test_intermediate(self) -> None:
        assert classify_ita(35.0) == "Intermediate"

    def test_dark(self) -> None:
        assert classify_ita(-40.0) == "Dark"

    def test_brown(self) -> None:
        assert classify_ita(5.0) == "Brown"

    def test_boundary_values(self) -> None:
        assert classify_ita(55.0) == "Light"
        assert classify_ita(41.0) == "Intermediate"
        assert classify_ita(28.0) == "Tan"
        assert classify_ita(10.0) == "Brown"
        assert classify_ita(-30.0) == "Dark"


class TestAnalyzeDatasetIta:
    def test_empty_directory(self, tmp_path: Path) -> None:
        report = analyze_dataset_ita(tmp_path)
        assert report == {"total": 0}

    def test_analyze_and_write_csv(self, tmp_path: Path) -> None:
        image_dir = tmp_path / "images"
        image_dir.mkdir()
        for i in range(3):
            img = np.full((32, 32, 3), [200 - i * 20, 160, 130], dtype=np.uint8)
            Image.fromarray(img).save(image_dir / f"img_{i}.jpg")

        output_csv = tmp_path / "ita_scores.csv"
        report = analyze_dataset_ita(image_dir, output_csv=output_csv, max_samples=2)

        assert report["total_analyzed"] == 2
        assert report["failed"] == 0
        assert "ita_stats" in report
        assert output_csv.exists()


class TestRunItaAnalysis:
    def test_run_end_to_end(self, tmp_path: Path) -> None:
        data_root = tmp_path / "raw"

        # fuseg path branch: base/train/images + labels
        fuseg_img = data_root / "dfu" / "fuseg" / "train" / "images"
        fuseg_mask = data_root / "dfu" / "fuseg" / "train" / "labels"
        fuseg_img.mkdir(parents=True)
        fuseg_mask.mkdir(parents=True)

        # healthy + non_dfu direct base dirs with images/masks
        healthy_img = data_root / "healthy" / "images"
        healthy_mask = data_root / "healthy" / "masks"
        non_dfu_img = data_root / "non_dfu" / "images"
        non_dfu_mask = data_root / "non_dfu" / "masks"
        healthy_img.mkdir(parents=True)
        healthy_mask.mkdir(parents=True)
        non_dfu_img.mkdir(parents=True)
        non_dfu_mask.mkdir(parents=True)

        # Seed one sample per dataset
        sample = np.full((24, 24, 3), [180, 140, 110], dtype=np.uint8)
        zero_mask = np.zeros((24, 24), dtype=np.uint8)
        Image.fromarray(sample).save(fuseg_img / "f1.jpg")
        Image.fromarray(zero_mask).save(fuseg_mask / "f1.png")
        Image.fromarray(sample).save(healthy_img / "h1.jpg")
        Image.fromarray(zero_mask).save(healthy_mask / "h1.png")
        Image.fromarray(sample).save(non_dfu_img / "n1.jpg")
        Image.fromarray(zero_mask).save(non_dfu_mask / "n1.png")

        output_dir = tmp_path / "meta"
        report = run_ita_analysis(data_root=data_root, output_dir=output_dir)

        assert "datasets" in report
        assert set(report["datasets"].keys()) == {"fuseg", "healthy", "non_dfu"}
        assert (output_dir / "ita_report.json").exists()
        assert (output_dir / "ita_scores.csv").exists()

        saved = json.loads((output_dir / "ita_report.json").read_text())
        assert "datasets" in saved
