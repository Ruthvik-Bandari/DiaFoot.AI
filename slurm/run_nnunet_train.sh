#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# DiaFoot.AI v2 — nnU-Net v2 Training (Fold 0) on H200
# Usage: sbatch slurm/run_nnunet_train.sh
# ═══════════════════════════════════════════════════════════════════════════════
#SBATCH --job-name=nnunet-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/%j_nnunet_train.out
#SBATCH --error=logs/slurm/%j_nnunet_train.err

set -euo pipefail

cd ~/DiaFoot.AI-v2
module purge
module load cuda/12.8.0 python/3.13.5
source .venv/bin/activate

export nnUNet_raw="$HOME/DiaFoot.AI-v2/nnunet_data/nnUNet_raw"
export nnUNet_preprocessed="$HOME/DiaFoot.AI-v2/nnunet_data/nnUNet_preprocessed"
export nnUNet_results="$HOME/DiaFoot.AI-v2/nnunet_data/nnUNet_results"

mkdir -p logs/slurm

echo "=== nnU-Net v2 Training — Fold 0 ==="
echo "Job ${SLURM_JOB_ID} | Node $(hostname) | GPU ${CUDA_VISIBLE_DEVICES:-?} | $(date)"

python -c "
import torch
assert torch.cuda.is_available(), 'No GPU'
print(f'PyTorch {torch.__version__} | {torch.cuda.get_device_name(0)}')
"

# Dataset 1, 2D config, Fold 0, save softmax predictions for ensembling.
# --c resumes from latest checkpoint if present (useful after timeout).
nnUNetv2_train 1 2d 0 --npz --c

echo "=== DONE: $(date) ==="
