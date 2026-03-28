"""DiaFoot.AI v2 — Fairness Tests (Phase 5, Commit 26)."""

from pathlib import Path

import numpy as np

from src.evaluation.fairness import (
    load_ita_mapping,
    print_fairness_report,
    run_fairness_audit,
    stratified_classification_audit,
    stratified_segmentation_audit,
)


class TestClassificationFairness:
    def test_equal_performance(self) -> None:
        filenames = ["a.png", "b.png", "c.png", "d.png"]
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        ita_map = {
            "a.png": "Light",
            "b.png": "Light",
            "c.png": "Dark",
            "d.png": "Dark",
        }
        result = stratified_classification_audit(filenames, y_true, y_pred, ita_map)
        assert result["fairness_gap_accuracy"] == 0.0
        assert not result["bias_concern"]

    def test_biased_performance(self) -> None:
        filenames = ["a.png", "b.png", "c.png", "d.png"]
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0])  # Wrong for Dark group
        ita_map = {
            "a.png": "Light",
            "b.png": "Light",
            "c.png": "Dark",
            "d.png": "Dark",
        }
        result = stratified_classification_audit(filenames, y_true, y_pred, ita_map)
        assert result["fairness_gap_accuracy"] == 1.0
        assert result["bias_concern"]

    def test_unknown_category_ignored_from_report_groups(self) -> None:
        filenames = ["a.png", "b.png"]
        y_true = np.array([0, 1])
        y_pred = np.array([0, 0])
        ita_map = {"a.png": "Unknown", "b.png": "Unknown"}
        result = stratified_classification_audit(filenames, y_true, y_pred, ita_map)
        assert result["per_ita_group"] == {}
        assert result["fairness_gap_accuracy"] == 0.0


class TestSegmentationFairness:
    def test_stratified_metrics(self) -> None:
        filenames = ["a.png", "b.png"]
        metrics = [
            {"dice": 0.9, "iou": 0.85},
            {"dice": 0.7, "iou": 0.6},
        ]
        ita_map = {"a.png": "Light", "b.png": "Dark"}
        result = stratified_segmentation_audit(filenames, metrics, ita_map)
        assert "Light" in result["per_ita_group"]
        assert "Dark" in result["per_ita_group"]
        assert abs(result["fairness_gaps"]["dice_gap"] - 0.2) < 1e-6
        assert result["bias_concern"]

    def test_no_known_ita_groups(self) -> None:
        filenames = ["x.png"]
        metrics = [{"dice": 0.8, "iou": 0.7}]
        ita_map = {"x.png": "Unknown"}
        result = stratified_segmentation_audit(filenames, metrics, ita_map)
        assert result["per_ita_group"] == {}
        assert result["fairness_gaps"] == {}
        assert result["bias_concern"] is False


class TestFairnessIo:
    def test_load_ita_mapping_missing_file(self, tmp_path: Path) -> None:
        mapping = load_ita_mapping(tmp_path / "missing.csv")
        assert mapping == {}

    def test_load_ita_mapping_reads_csv(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "ita.csv"
        csv_path.write_text("filename,category\na.png,Light\nb.png,Dark\n")
        mapping = load_ita_mapping(csv_path)
        assert mapping == {"a.png": "Light", "b.png": "Dark"}

    def test_run_fairness_audit_and_print(self, tmp_path: Path, capsys) -> None:
        ita_csv = tmp_path / "ita.csv"
        ita_csv.write_text("filename,category\na.png,Light\nb.png,Dark\n")

        cls_results = {
            "filenames": ["a.png", "b.png"],
            "y_true": np.array([0, 1]),
            "y_pred": np.array([0, 0]),
        }
        seg_results = {
            "filenames": ["a.png", "b.png"],
            "metrics_per_image": [
                {"dice": 0.9, "iou": 0.85, "hd95": 1.0, "nsd_2mm": 0.9, "nsd_5mm": 0.95},
                {"dice": 0.7, "iou": 0.60, "hd95": 3.0, "nsd_2mm": 0.7, "nsd_5mm": 0.8},
            ],
        }

        report = run_fairness_audit(
            classification_results=cls_results,
            segmentation_results=seg_results,
            ita_csv=ita_csv,
        )
        assert "classification" in report
        assert "segmentation" in report

        print_fairness_report(report)
        out = capsys.readouterr().out
        assert "ITA-Stratified Fairness Audit" in out
