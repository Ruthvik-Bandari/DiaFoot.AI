#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# DiaFoot.AI v2 — Evaluation Job
# Usage: sbatch slurm/evaluate.sh [checkpoint_path] [config_path]
# ═══════════════════════════════════════════════════════════════════════════════
#SBATCH --job-name=diafootai-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/slurm/%j_%x.out
#SBATCH --error=logs/slurm/%j_%x.err

set -euo pipefail
module purge && module load cuda/12.4 python/3.12
source .venv/bin/activate

CHECKPOINT=${1:-checkpoints/best_model.pt}
CONFIG=${2:-configs/training/baseline.yaml}
echo "Evaluating ${CHECKPOINT} with ${CONFIG} | $(date)"

mkdir -p logs/slurm

python scripts/evaluate.py \
    --checkpoint "${CHECKPOINT}" \
    --config "${CONFIG}" \
    --device cuda \
    --tta true \
    --save-predictions true

echo "Finished: $(date)"
