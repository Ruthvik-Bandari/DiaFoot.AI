"""Tests for split leakage audit utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from src.data.leakage_audit import audit_samples_for_leakage, canonical_sample_id

if TYPE_CHECKING:
    from pathlib import Path


def _write_image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed=value)
    img = rng.integers(0, 255, size=(32, 32, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)


def test_canonical_sample_id_strips_augmentation_suffixes() -> None:
    assert canonical_sample_id("foo/bar/patient12_aug3.png") == "patient12"
    assert canonical_sample_id("foo/bar/patient12_rot90.png") == "patient12"


def test_audit_detects_path_overlap(tmp_path: Path) -> None:
    p = tmp_path / "img.png"
    _write_image(p, 120)

    train = [{"image": str(p), "class": "dfu"}]
    val = [{"image": str(p), "class": "dfu"}]
    test = []

    report = audit_samples_for_leakage(train, val, test)
    assert report["path_overlap"]["train_x_val"] == 1
    assert report["has_any_leakage"] is True


def test_audit_detects_content_overlap_with_different_paths(tmp_path: Path) -> None:
    a = tmp_path / "a.png"
    b = tmp_path / "nested" / "b.png"
    _write_image(a, 80)
    b.parent.mkdir(parents=True, exist_ok=True)
    b.write_bytes(a.read_bytes())

    train = [{"image": str(a), "class": "dfu"}]
    val = [{"image": str(b), "class": "dfu"}]
    test = []

    report = audit_samples_for_leakage(train, val, test)
    assert report["path_overlap"]["train_x_val"] == 0
    assert report["content_overlap"]["train_x_val"] == 1


def test_audit_no_leakage_signal_for_distinct_images(tmp_path: Path) -> None:
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    c = tmp_path / "c.png"
    _write_image(a, 20)
    _write_image(b, 120)
    _write_image(c, 220)

    train = [{"image": str(a), "class": "dfu"}]
    val = [{"image": str(b), "class": "dfu"}]
    test = [{"image": str(c), "class": "dfu"}]

    report = audit_samples_for_leakage(train, val, test)
    assert report["has_any_leakage"] is False
