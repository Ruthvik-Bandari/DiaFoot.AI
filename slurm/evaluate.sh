#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# DiaFoot.AI v2 — Evaluation Job
# Usage: sbatch slurm/evaluate.sh [classify_ckpt] [segment_ckpt]
# ═══════════════════════════════════════════════════════════════════════════════
#SBATCH --job-name=diafootai-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/slurm/%j_%x.out
#SBATCH --error=logs/slurm/%j_%x.err

set -euo pipefail
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_ROOT"

module purge && module load cuda/12.8.0 python/3.12
source .venv/bin/activate
export PYTHONPATH="$PROJECT_ROOT"

CLASSIFIER_CKPT=${1:-checkpoints/dinov2_classifier/best.pt}
SEGMENTER_CKPT=${2:-checkpoints/dinov2_segmenter/best.pt}
MODEL_TYPE=${3:-dinov2}
BACKBONE=${4:-dinov2_vitb14}
echo "Evaluating classify=${CLASSIFIER_CKPT} segment=${SEGMENTER_CKPT} model=${MODEL_TYPE} | $(date)"

mkdir -p logs/slurm

"$PROJECT_ROOT/.venv/bin/python" scripts/evaluate.py \
    --task classify \
    --checkpoint "${CLASSIFIER_CKPT}" \
    --model "${MODEL_TYPE}" \
    --backbone "${BACKBONE}" \
    --device cuda \
    --verbose

"$PROJECT_ROOT/.venv/bin/python" scripts/evaluate.py \
    --task segment \
    --checkpoint "${SEGMENTER_CKPT}" \
    --model "${MODEL_TYPE}" \
    --backbone "${BACKBONE}" \
    --device cuda \
    --verbose

echo "Finished: $(date)"
