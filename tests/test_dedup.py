"""Tests for the shared dedup helpers in ``src.data.dedup``."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import cv2
import numpy as np

from src.data.dedup import canonical_stem, dhash, hamming, sha256

if TYPE_CHECKING:
    from pathlib import Path


def _write_image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed=value)
    img = rng.integers(0, 255, size=(32, 32, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)


# ── canonical_stem ──────────────────────────────────────────────────────────


def test_canonical_stem_collapses_chained_suffixes() -> None:
    assert canonical_stem("img_aug3_flip.png") == "img"


def test_canonical_stem_replace_dashes_true() -> None:
    assert canonical_stem("A-B_copy2.png", replace_dashes=True) == "a_b"


def test_canonical_stem_replace_dashes_false() -> None:
    assert canonical_stem("A-B_copy2.png", replace_dashes=False) == "a-b"


def test_canonical_stem_is_idempotent() -> None:
    x = "Patient12_aug3_flip.png"
    once = canonical_stem(x)
    twice = canonical_stem(once)
    assert once == twice


def test_canonical_stem_lowercases() -> None:
    assert canonical_stem("PATIENT12.png") == "patient12"


def test_canonical_stem_no_aug_token_unchanged_besides_case_and_dash() -> None:
    assert canonical_stem("Patient-12.png", replace_dashes=True) == "patient_12"
    assert canonical_stem("Patient-12.png", replace_dashes=False) == "patient-12"


# ── hamming ──────────────────────────────────────────────────────────────────


def test_hamming_identical_values_is_zero() -> None:
    assert hamming(0, 0) == 0


def test_hamming_known_distance() -> None:
    assert hamming(1, 2) == 2


def test_hamming_self_is_zero() -> None:
    x = 0xDEADBEEF
    assert hamming(x, x) == 0


# ── dhash ────────────────────────────────────────────────────────────────────


def test_dhash_returns_none_for_unreadable_path(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.png"
    assert dhash(missing) is None


def test_dhash_returns_int_for_valid_image(tmp_path: Path) -> None:
    img_path = tmp_path / "img.png"
    _write_image(img_path, 42)
    result = dhash(img_path)
    assert isinstance(result, int)


def test_dhash_same_image_hashed_twice_is_equal(tmp_path: Path) -> None:
    img_path = tmp_path / "img.png"
    _write_image(img_path, 7)
    h1 = dhash(img_path)
    h2 = dhash(img_path)
    assert h1 == h2
    assert hamming(h1, h2) == 0


# ── sha256 ───────────────────────────────────────────────────────────────────


def test_sha256_matches_hashlib(tmp_path: Path) -> None:
    p = tmp_path / "data.bin"
    p.write_bytes(b"diafoot dedup test payload")
    assert sha256(p) == hashlib.sha256(p.read_bytes()).hexdigest()


# ── back-compat: src.data.leakage_audit still exposes the old names ────────


def test_leakage_audit_back_compat_aliases() -> None:
    from src.data.dedup import dhash as canonical_dhash
    from src.data.dedup import hamming as canonical_hamming
    from src.data.dedup import sha256 as canonical_sha256
    from src.data.leakage_audit import _dhash, _hamming, _sha256, canonical_sample_id

    assert _hamming is canonical_hamming
    assert _dhash is canonical_dhash
    assert _sha256 is canonical_sha256
    assert canonical_sample_id("X_aug1.png") == "x"
