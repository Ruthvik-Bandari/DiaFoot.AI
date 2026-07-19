#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# DiaFoot.AI — Training-data-composition study (full matrix, SLURM array)
#
# 75 cells = 5 compositions x 3 architectures x 5 CV folds. To stay under the
# cluster's per-user submitted-job (QOS) limit, this is a 15-task array (one task
# per composition x architecture); each task runs its 5 CV folds SEQUENTIALLY.
# Every cell trains ONE segmentation model under a controlled TRAINING composition
# on one CV fold and evaluates it on the fixed, full, clean TEST set, writing one
# provenance-stamped JSON to results/composition/. Aggregate afterwards with
# scripts/aggregate_composition_results.py (CPU, login node OK). Per-cell JSONs are
# written as folds finish, so a task that is interrupted keeps its completed folds.
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
#SBATCH --time=24:00:00
#SBATCH --array=0-14%4
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

# Decode array index -> (composition, arch). 15 tasks = 5 comps x 3 archs.
IDX=${SLURM_ARRAY_TASK_ID}
ARCH=${ARCHS[$(( IDX % 3 ))]}
COMP=${COMPOSITIONS[$(( IDX / 3 ))]}

# Each task runs all 5 CV folds sequentially.
for FOLD in 0 1 2 3 4; do
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
done

echo "[$(date)] finished task ${IDX} (arch=${ARCH} comp=${COMP}, folds 0-4)"
