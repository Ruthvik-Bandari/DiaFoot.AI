#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# DiaFoot.AI — Training-data-composition study (full matrix, SLURM array)
#
# 75 cells = 5 compositions x 3 architectures x 5 CV folds. Each cell trains ONE
# segmentation model under a controlled TRAINING composition on one CV fold and
# evaluates it on the fixed, full, clean TEST set, writing one provenance-stamped
# JSON to results/composition/. Aggregate afterwards with
# scripts/aggregate_composition_results.py (CPU, login node OK).
#
# PREREQ: generate the shared folds first (once):
#     python scripts/make_cv_folds.py --n-folds 5 --seed 42
#
#   Usage:  sbatch slurm/run_composition_matrix.sh
#
# Compositions:  dfu_only, dfu_healthy, dfu_nondfu, all, random_mixed
# Architectures: unetpp (EffB4-scSE), segformer (MiT-B0), dinov2 (ViT-B/14 + UPerNet)
# ═══════════════════════════════════════════════════════════════════════════════
#SBATCH --job-name=diafoot-composition
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --array=0-74%4
#SBATCH --output=logs/slurm/%A_%a_composition.out
#SBATCH --error=logs/slurm/%A_%a_composition.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bandari.ru@northeastern.edu

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_ROOT"

# SLURM batch shells don't source Lmod init — set up the module environment.
set +u
source /etc/profile
set -u
module purge
module load cuda/12.8.0 python/3.13.5
source .venv/bin/activate

export PYTHONPATH="$PROJECT_ROOT"
export NO_ALBUMENTATIONS_UPDATE=1
# Explorer compute nodes have no internet. Pretrained weights must be pre-cached
# on the login node (see docs/COMPOSITION_EXPERIMENT_RUNBOOK.md); offline mode
# makes smp use the local cache immediately instead of retrying an unreachable
# Hugging Face hub for minutes per cell.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mkdir -p logs/slurm results/composition

COMPOSITIONS=(dfu_only dfu_healthy dfu_nondfu all random_mixed)
ARCHS=(unetpp segformer dinov2)
N_FOLDS=5

# Decode array index -> (composition, arch, fold). Layout: fold fastest, then
# arch, then composition (i = ((comp*3)+arch)*5 + fold).
IDX=${SLURM_ARRAY_TASK_ID}
FOLD=$(( IDX % N_FOLDS ))
REM=$(( IDX / N_FOLDS ))
ARCH=${ARCHS[$(( REM % 3 ))]}
COMP=${COMPOSITIONS[$(( REM / 3 ))]}

# Save qualitative masks only on fold 0 (one set of prediction PNGs per
# arch x composition is enough for the comparison figure).
QUAL=""
if [ "$FOLD" -eq 0 ]; then
  QUAL="--save-qualitative --n-qualitative 8"
fi

echo "[$(date)] task ${IDX} | arch=${ARCH} comp=${COMP} fold=${FOLD}"

# shellcheck disable=SC2086  # intentional word-splitting of $QUAL
"$PROJECT_ROOT/.venv/bin/python" scripts/run_composition_experiment.py \
    --arch "${ARCH}" \
    --composition "${COMP}" \
    --fold "${FOLD}" \
    --seed 42 \
    --device cuda \
    --epochs 50 \
    --batch-size 16 \
    --num-workers 8 \
    $QUAL

echo "[$(date)] finished task ${IDX}"
