#!/bin/bash
# DiaFoot.AI v2 — Dataset Download Automation
# Usage: bash scripts/download_data.sh

set -euo pipefail

DATA_DIR="data/raw"
mkdir -p "${DATA_DIR}/dfu/fuseg" "${DATA_DIR}/dfu/dfuc2022" "${DATA_DIR}/dfu/azh"
mkdir -p "${DATA_DIR}/healthy" "${DATA_DIR}/non_dfu"

echo "═══════════════════════════════════════════════════════════"
echo "DiaFoot.AI v2 — Data Download"
echo "═══════════════════════════════════════════════════════════"

# TODO: Phase 1, Commit 2 — Add download logic for:
# 1. FUSeg (1,210 images) — from GitHub/Kaggle
# 2. DFUC 2022 (4,000 images) — from challenge site
# 3. AZH (1,010 images) — from source
echo "Download scripts will be implemented in Phase 1, Commit 2"

echo "Done."
