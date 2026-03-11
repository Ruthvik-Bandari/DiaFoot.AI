"""Tests for reproducibility bundle helper functions."""

from __future__ import annotations

from pathlib import Path

from scripts.run_repro_bundle import _sha256


def test_sha256(tmp_path: Path) -> None:
    p = tmp_path / "a.txt"
    p.write_text("hello")
    digest = _sha256(p)
    assert len(digest) == 64
