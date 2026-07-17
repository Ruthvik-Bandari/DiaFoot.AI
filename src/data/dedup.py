"""DiaFoot.AI v2 — Shared deduplication helpers.

Canonical home for the filename-normalization, perceptual-hashing, and
content-hashing helpers used by the split-leakage audit
(``src.data.leakage_audit``) and by data-prep scripts. Extracted so a
single implementation exists instead of copy-pasted duplicates.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import cv2

AUGMENT_TOKENS_PATTERN = re.compile(
    r"(_aug\d+|_flip|_vflip|_hflip|_rot\d+|_copy\d+|_dup\d+)$",
    flags=re.IGNORECASE,
)


def canonical_stem(path: str | Path, *, replace_dashes: bool = True) -> str:
    """Normalize a file path's stem to a canonical dedup key.

    Lowercases the stem and repeatedly strips trailing augmentation/copy
    tokens (e.g. ``_aug3``, ``_flip``, ``_rot90``, ``_copy2``) until a
    fixed point is reached, so stacked suffixes such as ``_aug3_flip`` are
    fully removed rather than just the last token.

    Args:
        path: File path whose stem should be normalized.
        replace_dashes: When ``True``, replace ``-`` with ``_`` in the stem
            before stripping augmentation tokens, matching the convention
            used by data-prep scripts. When ``False``, dashes are left
            untouched, matching the split-leakage audit's canonical ID.

    Returns:
        The canonical, lowercased stem with augmentation suffixes removed.
    """
    stem = Path(path).stem.lower()
    if replace_dashes:
        stem = stem.replace("-", "_")
    # Strip repeatedly: offline augmentation can stack suffixes (e.g.
    # "_aug3_flip"), and a single pass would only remove the last token,
    # letting a chained-suffix copy slip past the canonical-overlap check.
    while True:
        stripped = AUGMENT_TOKENS_PATTERN.sub("", stem)
        if stripped == stem or not stripped:
            break
        stem = stripped
    return stem


def sha256(path: Path) -> str:
    """Compute the SHA256 hex digest of a file's contents.

    Args:
        path: Path to the file to hash.

    Returns:
        Hex-encoded SHA256 digest of the file's bytes.
    """
    data = path.read_bytes()
    return hashlib.sha256(data).hexdigest()


def dhash(path: Path, hash_size: int = 8) -> int | None:
    """Compute a 64-bit difference hash (dHash) for an image.

    Args:
        path: Path to the image file.
        hash_size: Hash grid size; the resulting hash has
            ``hash_size * hash_size`` bits.

    Returns:
        The dHash as an integer, or ``None`` if the image cannot be read.
    """
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    resized = cv2.resize(img, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    value = 0
    for bit in diff.flatten():
        value = (value << 1) | int(bool(bit))
    return value


def hamming(a: int, b: int) -> int:
    """Compute the Hamming distance between two integer hashes.

    Args:
        a: First hash value.
        b: Second hash value.

    Returns:
        Number of differing bits between ``a`` and ``b``.
    """
    return (a ^ b).bit_count()
