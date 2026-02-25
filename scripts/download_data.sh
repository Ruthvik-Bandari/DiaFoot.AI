#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# DiaFoot.AI v2 — Dataset Download (Phase 1, Commit 2)
#
# Downloads:
#   1. FUSeg (1,210 images + masks) — from GitHub (public)
#   2. AZH wound segmentation (1,010 images + masks) — from GitHub (public)
#   3. DFUC 2022 (4,000 images) — MANUAL: requires license agreement
#
# Usage: bash scripts/download_data.sh
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

DATA_DIR="data/raw/dfu"
mkdir -p "${DATA_DIR}/fuseg" "${DATA_DIR}/azh" "${DATA_DIR}/dfuc2022"
mkdir -p "data/metadata"

echo "═══════════════════════════════════════════════════════════"
echo "DiaFoot.AI v2 — DFU Dataset Download"
echo "═══════════════════════════════════════════════════════════"

# ── 1. FUSeg Dataset (1,210 images from UWM BigData Lab) ────────────────────
echo ""
echo "[1/3] Downloading FUSeg dataset..."
FUSEG_REPO="https://github.com/uwm-bigdata/wound-segmentation.git"
FUSEG_TMP="/tmp/fuseg_clone"

if [ -d "${DATA_DIR}/fuseg/train" ] && [ "$(find ${DATA_DIR}/fuseg -name '*.png' -o -name '*.jpg' | wc -l)" -gt 100 ]; then
    echo "  FUSeg already downloaded ($(find ${DATA_DIR}/fuseg -name '*.png' -o -name '*.jpg' | wc -l) images found). Skipping."
else
    rm -rf "${FUSEG_TMP}"
    echo "  Cloning uwm-bigdata/wound-segmentation (sparse checkout for data only)..."
    git clone --depth 1 --filter=blob:none --sparse "${FUSEG_REPO}" "${FUSEG_TMP}"
    cd "${FUSEG_TMP}"
    git sparse-checkout set "data/Foot Ulcer Segmentation Challenge"
    cd -

    # Copy into our structure
    FUSEG_SRC="${FUSEG_TMP}/data/Foot Ulcer Segmentation Challenge"
    if [ -d "${FUSEG_SRC}/train" ]; then
        cp -r "${FUSEG_SRC}/train" "${DATA_DIR}/fuseg/train"
        echo "  Copied train split: $(find ${DATA_DIR}/fuseg/train -name '*.png' -o -name '*.jpg' | wc -l) files"
    fi
    if [ -d "${FUSEG_SRC}/validation" ]; then
        cp -r "${FUSEG_SRC}/validation" "${DATA_DIR}/fuseg/validation"
        echo "  Copied validation split: $(find ${DATA_DIR}/fuseg/validation -name '*.png' -o -name '*.jpg' | wc -l) files"
    fi
    if [ -d "${FUSEG_SRC}/test" ]; then
        cp -r "${FUSEG_SRC}/test" "${DATA_DIR}/fuseg/test"
        echo "  Copied test split: $(find ${DATA_DIR}/fuseg/test -name '*.png' -o -name '*.jpg' | wc -l) files"
    fi
    rm -rf "${FUSEG_TMP}"
    FUSEG_COUNT=$(find ${DATA_DIR}/fuseg -name '*.png' -o -name '*.jpg' | wc -l)
    echo "  ✓ FUSeg complete: ${FUSEG_COUNT} total files"
fi

# ── 2. AZH Wound Segmentation Dataset (1,010 images) ───────────────────────
echo ""
echo "[2/3] Downloading AZH wound segmentation dataset..."
AZH_REPO="https://github.com/uwm-bigdata/wound-segmentation.git"
AZH_TMP="/tmp/azh_clone"

if [ -d "${DATA_DIR}/azh/images" ] && [ "$(find ${DATA_DIR}/azh -name '*.png' -o -name '*.jpg' | wc -l)" -gt 100 ]; then
    echo "  AZH already downloaded ($(find ${DATA_DIR}/azh -name '*.png' -o -name '*.jpg' | wc -l) images found). Skipping."
else
    rm -rf "${AZH_TMP}"
    echo "  Cloning uwm-bigdata/wound-segmentation (sparse checkout for wound_dataset)..."
    git clone --depth 1 --filter=blob:none --sparse "${AZH_REPO}" "${AZH_TMP}"
    cd "${AZH_TMP}"
    git sparse-checkout set "data/wound_dataset"
    cd -

    AZH_SRC="${AZH_TMP}/data/wound_dataset"
    if [ -d "${AZH_SRC}" ]; then
        # AZH stores images and masks in subdirectories
        cp -r "${AZH_SRC}" "${DATA_DIR}/azh/raw_download"
        # Reorganize into images/ and masks/ structure
        mkdir -p "${DATA_DIR}/azh/images" "${DATA_DIR}/azh/masks"

        # The wound_dataset typically has train/test with images/labels
        find "${DATA_DIR}/azh/raw_download" -path "*/images/*" -type f -exec cp {} "${DATA_DIR}/azh/images/" \;
        find "${DATA_DIR}/azh/raw_download" -path "*/labels/*" -type f -exec cp {} "${DATA_DIR}/azh/masks/" \;

        AZH_IMG_COUNT=$(find ${DATA_DIR}/azh/images -type f | wc -l)
        AZH_MASK_COUNT=$(find ${DATA_DIR}/azh/masks -type f | wc -l)
        echo "  ✓ AZH complete: ${AZH_IMG_COUNT} images, ${AZH_MASK_COUNT} masks"

        # Cleanup intermediate
        rm -rf "${DATA_DIR}/azh/raw_download"
    else
        echo "  WARNING: wound_dataset not found in repo. Check repo structure."
    fi
    rm -rf "${AZH_TMP}"
fi

# ── 3. DFUC 2022 (4,000 images — requires license agreement) ───────────────
echo ""
echo "[3/3] DFUC 2022 dataset..."
if [ -d "${DATA_DIR}/dfuc2022/train" ] && [ "$(find ${DATA_DIR}/dfuc2022 -name '*.png' -o -name '*.jpg' | wc -l)" -gt 100 ]; then
    echo "  DFUC 2022 already present ($(find ${DATA_DIR}/dfuc2022 -name '*.png' -o -name '*.jpg' | wc -l) images found)."
else
    echo "  ╔════════════════════════════════════════════════════════════╗"
    echo "  ║  DFUC 2022 requires a LICENSE AGREEMENT to download.     ║"
    echo "  ║                                                          ║"
    echo "  ║  Steps:                                                  ║"
    echo "  ║  1. Visit: https://dfu-challenge.github.io/dfuc2022.html ║"
    echo "  ║  2. Or Grand Challenge:                                  ║"
    echo "  ║     https://dfuc2022.grand-challenge.org/dataset/        ║"
    echo "  ║  3. Email m.yap@mmu.ac.uk with signed license agreement  ║"
    echo "  ║  4. Once received, extract to:                           ║"
    echo "  ║     data/raw/dfu/dfuc2022/train/images/                  ║"
    echo "  ║     data/raw/dfu/dfuc2022/train/masks/                   ║"
    echo "  ║     data/raw/dfu/dfuc2022/test/images/                   ║"
    echo "  ║  5. Also check Kaggle: laithjj/diabetic-foot-ulcer-dfu   ║"
    echo "  ╚════════════════════════════════════════════════════════════╝"
    mkdir -p "${DATA_DIR}/dfuc2022/train/images" "${DATA_DIR}/dfuc2022/train/masks"
    mkdir -p "${DATA_DIR}/dfuc2022/test/images"
fi

# ── Summary ─────────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "Download Summary:"
echo "  FUSeg:    $(find ${DATA_DIR}/fuseg -type f 2>/dev/null | wc -l) files"
echo "  AZH:      $(find ${DATA_DIR}/azh -type f 2>/dev/null | wc -l) files"
echo "  DFUC2022: $(find ${DATA_DIR}/dfuc2022 -type f 2>/dev/null | wc -l) files"
echo ""
echo "Next: python scripts/run_cleaning.py --audit-only"
echo "═══════════════════════════════════════════════════════════"
