#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# DiaFoot.AI — Training-data-composition study (full matrix, SLURM array)
#
# 14 cells. Each cell trains ONE segmentation model under a controlled TRAINING
# composition and evaluates it on the fixed, full, clean test set, writing one
# provenance-stamped JSON to results/composition/. After the array finishes,
# aggregate with scripts/aggregate_composition_results.py (CPU, login node OK).
#
#   Usage:  sbatch slurm/run_composition_matrix.sh
#
# Matrix:
#   U-Net++  x {dfu_only, dfu_nondfu, all}  x seeds {41,42,43}   (9, error bars)
#   U-Net++  x negative-ratio {0.25, 0.50}  x seed 42            (2, dose-response)
#   DINOv2   x {dfu_only, dfu_nondfu, all}  x seed 42            (3, arch check)
# ═══════════════════════════════════════════════════════════════════════════════
#SBATCH --job-name=diafoot-composition
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --array=0-13%4
#SBATCH --output=logs/slurm/%A_%a_composition.out
#SBATCH --error=logs/slurm/%A_%a_composition.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bandari.ru@northeastern.edu

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_ROOT"

# SLURM batch shells don't source Lmod init — set up the module environment
# (disable nounset around /etc/profile, which references unset vars).
set +u
source /etc/profile
set -u
module purge
module load cuda/12.8.0 python/3.13.5
source .venv/bin/activate

export PYTHONPATH="$PROJECT_ROOT"
export NO_ALBUMENTATIONS_UPDATE=1
mkdir -p logs/slurm results/composition

# One arg-string per array index. run_composition_experiment.py picks the input
# size per architecture (unetpp=512, dinov2=518) and writes each cell's filtered
# splits + checkpoints to its OWN directory, so concurrent cells never collide.
CELLS=(
  "--arch unetpp --composition dfu_only   --seed 41"
  "--arch unetpp --composition dfu_only   --seed 42"
  "--arch unetpp --composition dfu_only   --seed 43"
  "--arch unetpp --composition dfu_nondfu --seed 41"
  "--arch unetpp --composition dfu_nondfu --seed 42"
  "--arch unetpp --composition dfu_nondfu --seed 43"
  "--arch unetpp --composition all        --seed 41"
  "--arch unetpp --composition all        --seed 42"
  "--arch unetpp --composition all        --seed 43"
  "--arch unetpp --neg-frac 0.25          --seed 42"
  "--arch unetpp --neg-frac 0.50          --seed 42"
  "--arch dinov2 --composition dfu_only   --seed 42"
  "--arch dinov2 --composition dfu_nondfu --seed 42"
  "--arch dinov2 --composition all        --seed 42"
)

CELL="${CELLS[$SLURM_ARRAY_TASK_ID]}"
echo "[$(date)] array task ${SLURM_ARRAY_TASK_ID} | cell: ${CELL}"

# shellcheck disable=SC2086  # intentional word-splitting of the arg string
"$PROJECT_ROOT/.venv/bin/python" scripts/run_composition_experiment.py $CELL \
    --device cuda \
    --epochs 50 \
    --batch-size 16 \
    --num-workers 8

echo "[$(date)] finished task ${SLURM_ARRAY_TASK_ID}"
