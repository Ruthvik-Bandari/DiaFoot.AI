"""DiaFoot.AI v2 — Data Pipeline Tests (Phase 1, Commit 2)."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.data.cleaning import AuditConfig, AuditReport, DataQualityAuditor


@pytest.fixture
def sample_image_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with sample images."""
    img_dir = tmp_path / "test_images"
    img_dir.mkdir()
    for i in range(5):
        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        img.save(img_dir / f"img_{i:03d}.png")
    return img_dir


class TestAuditConfig:
    def test_default_config(self) -> None:
        cfg = AuditConfig()
        assert cfg.blurry_threshold == 0.3
        assert cfg.dark_threshold == 0.05

    def test_from_yaml(self, tmp_path: Path) -> None:
        yaml_content = "cleanvision:\n  blurry_threshold: 0.5\n  dark_threshold: 0.1\n"
        yaml_path = tmp_path / "test_config.yaml"
        yaml_path.write_text(yaml_content)
        cfg = AuditConfig.from_yaml(yaml_path)
        assert cfg.blurry_threshold == 0.5
        assert cfg.dark_threshold == 0.1


class TestAuditReport:
    def test_to_dict(self) -> None:
        report = AuditReport(dataset_name="test", total_images=10)
        d = report.to_dict()
        assert d["dataset_name"] == "test"
        assert d["total_images"] == 10


class TestDataQualityAuditor:
    def test_find_images(self, sample_image_dir: Path) -> None:
        auditor = DataQualityAuditor(sample_image_dir)
        images = auditor._find_images()
        assert len(images) == 5

    def test_basic_stats(self, sample_image_dir: Path) -> None:
        auditor = DataQualityAuditor(sample_image_dir)
        images = auditor._find_images()
        stats = auditor._collect_basic_stats(images)
        assert stats["count"] == 5
        assert stats["width"]["min"] == 64
        assert stats["height"]["max"] == 64

    def test_run_audit(self, sample_image_dir: Path) -> None:
        auditor = DataQualityAuditor(sample_image_dir, dataset_name="test_ds")
        report = auditor.run_audit()
        assert report.total_images == 5
        assert report.dataset_name == "test_ds"

    def test_save_report(self, sample_image_dir: Path, tmp_path: Path) -> None:
        auditor = DataQualityAuditor(sample_image_dir)
        auditor.run_audit()
        output = tmp_path / "report.json"
        auditor.save_report(output)
        assert output.exists()
        data = json.loads(output.read_text())
        assert data["total_images"] == 5

    def test_missing_directory_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            DataQualityAuditor("/nonexistent/path")

    def test_empty_directory(self, tmp_path: Path) -> None:
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        auditor = DataQualityAuditor(empty_dir)
        report = auditor.run_audit()
        assert report.total_images == 0
