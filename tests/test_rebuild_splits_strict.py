"""Tests for strict split rebuild.

The rebuild must keep every patch of a single source image in one split.
Patches (`_0`, `_1`, ...) of one wound capture are distinct crops, so they do
NOT collide under dHash; grouping must therefore key on the source-image id
derived from the filename, and must work even when the image files are absent
(the grouping decision is filename-derivable and runs before any pixels load).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from scripts.rebuild_splits_strict import (
    assign_components,
    build_components,
    canonical_stem,
    enrich_rows,
    infer_patient_id,
    infer_source_id,
)


def _azh_rows(n_images: int, patches: int, cls: str) -> list[dict[str, str]]:
    """Synthetic AZH-style rows: `azh_train_<32hex>_<patch>.png`, no files on disk."""
    rows: list[dict[str, str]] = []
    for i in range(n_images):
        case = f"{i:032x}"
        for p in range(patches):
            img = f"data/processed/{cls}/images/azh_train_{case}_{p}.png"
            rows.append(
                {
                    "image": img,
                    "mask": img.replace("/images/", "/masks/"),
                    "class": cls,
                    "ita": "",
                    "ita_group": "unknown",
                    "source_id": "azh",
                    "patient_id": "",
                }
            )
    return rows


def _group_key(row: dict[str, str]) -> tuple[str, str, str]:
    p = Path(row["image"])
    cls = row["class"]
    return (cls, infer_source_id(cls, p), infer_patient_id(cls, p))


def _straddle_count(rebuilt: dict[str, list[dict[str, str]]]) -> int:
    sets = {s: {_group_key(r) for r in rows} for s, rows in rebuilt.items()}
    total = 0
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        total += len(sets[a] & sets[b])
    return total


def test_source_image_patches_never_straddle_splits() -> None:
    rows = _azh_rows(n_images=12, patches=3, cls="dfu")
    rows += _azh_rows(n_images=8, patches=2, cls="non_dfu")
    rows = enrich_rows(rows)

    components = build_components(rows, near_threshold=6)
    rebuilt = assign_components(
        rows=rows,
        class_components=components,
        seed=42,
        train_ratio=0.70,
        val_ratio=0.15,
    )

    assert _straddle_count(rebuilt) == 0


def test_patches_of_one_source_image_share_a_component_without_image_files() -> None:
    rows = enrich_rows(_azh_rows(n_images=6, patches=3, cls="dfu"))
    components = build_components(rows, near_threshold=6)

    index_to_component: dict[int, int] = {}
    for cid, comp in enumerate(components["dfu"]):
        for gi in comp:
            index_to_component[gi] = cid

    by_source_image: dict[tuple[str, str, str], set[int]] = defaultdict(set)
    for gi, row in enumerate(rows):
        by_source_image[_group_key(row)].add(index_to_component[gi])

    for key, comp_ids in by_source_image.items():
        assert len(comp_ids) == 1, f"source image {key} split across components {comp_ids}"


def test_presence_repair_moves_whole_components_never_splitting_one() -> None:
    # Few large components in one class force greedy assignment to leave a split
    # empty, which triggers the class-presence repair. The repair must relocate a
    # WHOLE component; moving a single patch would straddle a source image across
    # splits (the old row-popping repair failed exactly here).
    rows = enrich_rows(_azh_rows(n_images=3, patches=10, cls="dfu"))
    components = build_components(rows, near_threshold=6)
    rebuilt = assign_components(
        rows=rows,
        class_components=components,
        seed=42,
        train_ratio=0.70,
        val_ratio=0.15,
    )

    assert _straddle_count(rebuilt) == 0
    # 3 components -> one whole component per split after repair; none empty.
    assert all(len(rebuilt[s]) > 0 for s in ("train", "val", "test"))


def test_near_duplicate_across_source_images_is_merged(tmp_path: Path) -> None:
    # Two content-identical images filed under DIFFERENT source-image ids:
    # source-image grouping alone cannot merge them, only the dHash near-dup pass
    # can. A third distinct image must stay separate. This exercises the
    # full-pairwise near-dup merge (the half of the fix that replaced the broken
    # top-16-bit bucketing) — previously untested because no image files existed.
    root = tmp_path / "mendeley"
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    base = rng.integers(0, 255, size=(32, 32, 3), dtype=np.uint8)
    a = root / "mendeley_0001.png"
    b = root / "mendeley_0002.png"
    c = root / "mendeley_0003.png"
    cv2.imwrite(str(a), base)
    b.write_bytes(a.read_bytes())  # duplicate content, different id
    cv2.imwrite(str(c), rng.integers(0, 255, size=(32, 32, 3), dtype=np.uint8))

    rows = enrich_rows(
        [
            {"image": str(a), "mask": "", "class": "non_dfu"},
            {"image": str(b), "mask": "", "class": "non_dfu"},
            {"image": str(c), "mask": "", "class": "non_dfu"},
        ]
    )
    components = build_components(rows, near_threshold=6)["non_dfu"]

    comp_of: dict[str, int] = {}
    for cid, comp in enumerate(components):
        for gi in comp:
            comp_of[Path(rows[gi]["image"]).name] = cid

    assert comp_of["mendeley_0001.png"] == comp_of["mendeley_0002.png"]
    assert comp_of["mendeley_0003.png"] != comp_of["mendeley_0001.png"]


def test_infer_patient_id_keeps_per_image_granularity_and_strips_stacked_suffixes() -> None:
    base = "data/processed/healthy/images"
    # distinct photos stay distinct (no over-collapse to a single "male_normal")
    assert infer_patient_id("healthy", Path(f"{base}/Male_Normal-1483.png")) == "male_normal_1483"
    assert infer_patient_id("healthy", Path(f"{base}/Male_Normal-1483.png")) != infer_patient_id(
        "healthy", Path(f"{base}/Male_Normal-1484.png")
    )
    # augmentation copies (even stacked) group back to their source image
    assert (
        infer_patient_id("healthy", Path(f"{base}/male_normal_1483_aug3_flip.png"))
        == "male_normal_1483"
    )
    assert canonical_stem("male_normal_1483_aug3_flip") == "male_normal_1483"
