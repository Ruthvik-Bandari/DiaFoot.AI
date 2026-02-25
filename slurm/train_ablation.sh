#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# DiaFoot.AI v2 — Ablation Sweep (SLURM Array Job)
# Usage: sbatch slurm/train_ablation.sh
# Runs 12 ablation configs in parallel on separate H100 GPUs
# ═══════════════════════════════════════════════════════════════════════════════
#SBATCH --job-name=diafootai-ablation
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-11
#SBATCH --output=logs/slurm/%A_%a_ablation.out
#SBATCH --error=logs/slurm/%A_%a_ablation.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bandari.r@northeastern.edu

set -euo pipefail
module purge && module load cuda/12.4 python/3.12
source .venv/bin/activate

# ── Ablation Configurations ─────────────────────────────────────────────────
CONFIGS=(
    "configs/ablation/loss_dice_ce.yaml"
    "configs/ablation/loss_dice_boundary.yaml"
    "configs/ablation/loss_focal_tversky.yaml"
    "configs/ablation/loss_unified_focal.yaml"
    "configs/ablation/encoder_effb4.yaml"
    "configs/ablation/encoder_effb7.yaml"
    "configs/ablation/encoder_convnext.yaml"
    "configs/ablation/encoder_vit.yaml"
    "configs/ablation/data_dfu_only.yaml"
    "configs/ablation/data_dfu_healthy.yaml"
    "configs/ablation/data_dfu_healthy_nondfu.yaml"
    "configs/ablation/attention_pscse.yaml"
)

CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}
echo "Array task ${SLURM_ARRAY_TASK_ID} | Config: ${CONFIG} | $(date)"

export TORCH_COMPILE_BACKEND=inductor
mkdir -p logs/slurm

python scripts/train.py \
    --config "${CONFIG}" \
    --device cuda \
    --precision bf16-mixed \
    --compile true \
    --seed 42 \
    --tags ablation

echo "Finished task ${SLURM_ARRAY_TASK_ID}: $(date)"
