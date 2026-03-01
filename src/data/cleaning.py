"""DiaFoot.AI v2 — Data Cleaning & Quality Audit Pipeline.

Phase 1, Commit 2: CleanVision-based automated quality audit.
Detects blurry, dark, light, duplicate, and near-duplicate images.

Usage:
    from src.data.cleaning import DataQualityAuditor
    auditor = DataQualityAuditor("data/raw/dfu/fuseg")
    report = auditor.run_audit()
    auditor.save_report("data/metadata/quality_report.json")
"""

from __future__ import annotations

import json
import logging
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class AuditConfig:
    """Configuration for data quality audit thresholds."""

    blurry_threshold: float = 0.3
    dark_threshold: float = 0.05
    light_threshold: float = 0.95
    duplicate_hash_threshold: float = 0.95
    near_duplicate_threshold: float = 0.90
    low_information_threshold: float = 0.15
    odd_aspect_ratio_threshold: float = 3.0
    odd_size_min: tuple[int, int] = (64, 64)

    @classmethod
    def from_yaml(cls, path: str | Path) -> AuditConfig:
        """Load config from YAML file."""
        with open(path) as f:
            raw = yaml.safe_load(f)
        cv_cfg = raw.get("cleanvision", raw)
        return cls(
            blurry_threshold=cv_cfg.get("blurry_threshold", 0.3),
            dark_threshold=cv_cfg.get("dark_threshold", 0.05),
            light_threshold=cv_cfg.get("light_threshold", 0.95),
            duplicate_hash_threshold=cv_cfg.get("duplicate_hash_threshold", 0.95),
            near_duplicate_threshold=cv_cfg.get("near_duplicate_threshold", 0.90),
            low_information_threshold=cv_cfg.get("low_information_threshold", 0.15),
            odd_aspect_ratio_threshold=cv_cfg.get("odd_aspect_ratio_threshold", 3.0),
            odd_size_min=tuple(cv_cfg.get("odd_size_threshold", [64, 64])),
        )


@dataclass
class AuditReport:
    """Results of a data quality audit."""

    dataset_name: str
    total_images: int = 0
    issues: dict[str, list[str]] = field(default_factory=dict)
    summary: dict[str, int] = field(default_factory=dict)
    image_stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary for JSON serialization."""
        return {
            "dataset_name": self.dataset_name,
            "total_images": self.total_images,
            "summary": self.summary,
            "issues": {k: v[:50] for k, v in self.issues.items()},  # Cap at 50 per issue
            "image_stats": self.image_stats,
        }


class DataQualityAuditor:
    """Run CleanVision-based data quality audit on image directories.

    Args:
        image_dir: Path to directory containing images (searched recursively).
        config: Audit configuration thresholds.
        dataset_name: Human-readable name for the dataset being audited.
    """

    IMAGE_EXTENSIONS: typing.ClassVar[set[str]] = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}

    def __init__(
        self,
        image_dir: str | Path,
        config: AuditConfig | None = None,
        dataset_name: str | None = None,
    ) -> None:
        """Initialize the auditor with image directory and config."""
        self.image_dir = Path(image_dir)
        self.config = config or AuditConfig()
        self.dataset_name = dataset_name or self.image_dir.name
        self.report: AuditReport | None = None

        if not self.image_dir.exists():
            msg = f"Image directory not found: {self.image_dir}"
            raise FileNotFoundError(msg)

    def _find_images(self) -> list[Path]:
        """Recursively find all image files in the directory."""
        images: list[Path] = []
        for ext in self.IMAGE_EXTENSIONS:
            images.extend(self.image_dir.rglob(f"*{ext}"))
            images.extend(self.image_dir.rglob(f"*{ext.upper()}"))
        # Deduplicate (case-insensitive extensions may overlap)
        images = list({p.resolve() for p in images})
        images.sort()
        return images

    def _collect_basic_stats(self, image_paths: list[Path]) -> dict[str, Any]:
        """Collect basic image statistics without CleanVision."""
        from PIL import Image

        widths, heights, sizes_kb = [], [], []
        formats: dict[str, int] = {}
        errors: list[str] = []

        for p in image_paths:
            try:
                with Image.open(p) as img:
                    w, h = img.size
                    widths.append(w)
                    heights.append(h)
                    fmt = img.format or p.suffix.lower()
                    formats[fmt] = formats.get(fmt, 0) + 1
                sizes_kb.append(p.stat().st_size / 1024)
            except Exception as e:
                errors.append(f"{p.name}: {e}")

        stats: dict[str, Any] = {
            "count": len(image_paths),
            "corrupt_files": errors,
            "formats": formats,
        }
        if widths:
            stats["width"] = {
                "min": min(widths),
                "max": max(widths),
                "mean": sum(widths) / len(widths),
            }
            stats["height"] = {
                "min": min(heights),
                "max": max(heights),
                "mean": sum(heights) / len(heights),
            }
            stats["size_kb"] = {
                "min": round(min(sizes_kb), 1),
                "max": round(max(sizes_kb), 1),
                "mean": round(sum(sizes_kb) / len(sizes_kb), 1),
            }
        return stats

    def run_audit(self) -> AuditReport:
        """Execute the full CleanVision quality audit.

        Returns:
            AuditReport with detected issues and statistics.
        """
        image_paths = self._find_images()
        logger.info("Found %d images in %s", len(image_paths), self.image_dir)

        if not image_paths:
            logger.warning("No images found in %s", self.image_dir)
            return AuditReport(dataset_name=self.dataset_name, total_images=0)

        # Basic stats (always collected)
        basic_stats = self._collect_basic_stats(image_paths)

        # CleanVision audit
        issues: dict[str, list[str]] = {}
        summary: dict[str, int] = {}

        try:
            from cleanvision import Imagelab

            image_list = [str(p) for p in image_paths]
            lab = Imagelab(filepaths=image_list)

            logger.info("Running CleanVision audit on %d images...", len(image_list))
            lab.find_issues()

            # Extract issue results
            issue_df = lab.issues
            issue_types = [
                col.replace("_issue", "") for col in issue_df.columns if col.endswith("_issue")
            ]

            for issue_type in issue_types:
                col = f"{issue_type}_issue"
                if col in issue_df.columns:
                    flagged_idx = issue_df[issue_df[col]].index.tolist()
                    if flagged_idx:
                        # CleanVision indices may be str paths or int — handle both
                        flagged_names = []
                        for idx in flagged_idx:
                            if isinstance(idx, int) and idx < len(image_list):
                                flagged_names.append(Path(image_list[idx]).name)
                            elif isinstance(idx, str):
                                flagged_names.append(Path(idx).name)
                            else:
                                flagged_names.append(str(idx))
                        issues[issue_type] = flagged_names
                        summary[issue_type] = len(flagged_names)

            logger.info("CleanVision audit complete. Issues found: %s", summary)

        except ImportError:
            logger.warning("CleanVision not installed. Running basic stats only.")
            summary["cleanvision_unavailable"] = 1
        except Exception as e:
            logger.error("CleanVision audit failed: %s", e)
            summary["cleanvision_error"] = 1
            issues["error"] = [str(e)]

        self.report = AuditReport(
            dataset_name=self.dataset_name,
            total_images=len(image_paths),
            issues=issues,
            summary=summary,
            image_stats=basic_stats,
        )
        return self.report

    def save_report(self, output_path: str | Path) -> Path:
        """Save audit report to JSON file.

        Args:
            output_path: Path for the JSON output file.

        Returns:
            Path to the saved report file.
        """
        if self.report is None:
            msg = "No report to save. Run run_audit() first."
            raise RuntimeError(msg)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(self.report.to_dict(), f, indent=2, default=str)

        logger.info("Report saved to %s", output_path)
        return output_path

    def print_summary(self) -> None:
        """Print a human-readable summary of the audit."""
        if self.report is None:
            print("No report available. Run run_audit() first.")  # noqa: T201
            return

        r = self.report
        print(f"\n{'═' * 60}")  # noqa: T201
        print(f"Data Quality Report: {r.dataset_name}")  # noqa: T201
        print(f"{'═' * 60}")  # noqa: T201
        print(f"Total images: {r.total_images}")  # noqa: T201

        if r.image_stats:
            s = r.image_stats
            if "width" in s:
                w_min, w_max = s["width"]["min"], s["width"]["max"]
                h_min, h_max = s["height"]["min"], s["height"]["max"]
                print(f"Image size: {w_min}-{w_max} x {h_min}-{h_max}")  # noqa: T201
            if "size_kb" in s:
                sk = s["size_kb"]
                print(f"File size: {sk['min']}-{sk['max']} KB (avg {sk['mean']} KB)")  # noqa: T201
            if s.get("corrupt_files"):
                print(f"Corrupt files: {len(s['corrupt_files'])}")  # noqa: T201

        if r.summary:
            print("\nIssues detected:")  # noqa: T201
            for issue_type, count in sorted(r.summary.items()):
                pct = (count / r.total_images * 100) if r.total_images > 0 else 0
                print(f"  {issue_type}: {count} ({pct:.1f}%)")  # noqa: T201
        else:
            print("\nNo issues detected.")  # noqa: T201

        print(f"{'═' * 60}\n")  # noqa: T201


def audit_all_dfu_datasets(
    data_root: str | Path = "data/raw/dfu",
    config_path: str | Path = "configs/data/cleaning.yaml",
    output_dir: str | Path = "data/metadata",
) -> dict[str, AuditReport]:
    """Run audit on all DFU dataset directories.

    Args:
        data_root: Root directory containing dataset subdirectories.
        config_path: Path to cleaning config YAML.
        output_dir: Directory for output reports.

    Returns:
        Dictionary mapping dataset names to their audit reports.
    """
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    config_path = Path(config_path)
    config = AuditConfig.from_yaml(config_path) if config_path.exists() else AuditConfig()

    reports: dict[str, AuditReport] = {}
    combined_report: dict[str, Any] = {"datasets": {}}

    # Find all dataset subdirectories
    dataset_dirs = [d for d in data_root.iterdir() if d.is_dir()]

    if not dataset_dirs:
        logger.warning("No dataset directories found in %s", data_root)
        return reports

    for dataset_dir in sorted(dataset_dirs):
        logger.info("Auditing dataset: %s", dataset_dir.name)
        try:
            auditor = DataQualityAuditor(
                image_dir=dataset_dir,
                config=config,
                dataset_name=dataset_dir.name,
            )
            report = auditor.run_audit()
            auditor.print_summary()

            # Save individual report
            individual_path = output_dir / f"quality_report_{dataset_dir.name}.json"
            auditor.save_report(individual_path)

            reports[dataset_dir.name] = report
            combined_report["datasets"][dataset_dir.name] = report.to_dict()

        except Exception as e:
            logger.error("Failed to audit %s: %s", dataset_dir.name, e)

    # Save combined report
    combined_path = output_dir / "quality_report.json"
    with open(combined_path, "w") as f:
        json.dump(combined_report, f, indent=2, default=str)
    logger.info("Combined report saved to %s", combined_path)

    return reports
