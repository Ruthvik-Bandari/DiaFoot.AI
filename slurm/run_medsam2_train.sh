#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# DiaFoot.AI v2 — MedSAM2 LoRA Fine-Tuning on H200
# Usage: sbatch slurm/run_medsam2_train.sh
# ═══════════════════════════════════════════════════════════════════════════════
#SBATCH --job-name=medsam2-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/%j_medsam2_train.out
#SBATCH --error=logs/slurm/%j_medsam2_train.err

set -euo pipefail

cd ~/DiaFoot.AI-v2
module purge
module load cuda/12.8.0 python/3.13.5
source .venv/bin/activate

export PYTHONPATH="$PWD"
export NO_ALBUMENTATIONS_UPDATE=1
mkdir -p logs/slurm checkpoints/medsam2_lora

echo "=== MedSAM2 LoRA Training ==="
echo "Job ${SLURM_JOB_ID} | Node $(hostname) | GPU ${CUDA_VISIBLE_DEVICES:-?} | $(date)"

# Verify GPU and SAM2 availability
python -c "
import torch
assert torch.cuda.is_available(), 'No GPU'
print(f'PyTorch {torch.__version__} | {torch.cuda.get_device_name(0)}')
from sam2.build_sam import build_sam2
print('SAM2 import OK')
"

# Find SAM2 checkpoint if available
SAM2_CKPT=""
if [ -f "models/sam2_weights/sam2.1_hiera_base_plus.pt" ]; then
    SAM2_CKPT="--sam2-checkpoint models/sam2_weights/sam2.1_hiera_base_plus.pt"
    echo "Using checkpoint: models/sam2_weights/sam2.1_hiera_base_plus.pt"
elif [ -f "models/sam2_weights/sam2_hiera_base_plus.pt" ]; then
    SAM2_CKPT="--sam2-checkpoint models/sam2_weights/sam2_hiera_base_plus.pt"
    echo "Using checkpoint: models/sam2_weights/sam2_hiera_base_plus.pt"
elif [ -f "checkpoints/sam2.1_hiera_base_plus.pt" ]; then
    SAM2_CKPT="--sam2-checkpoint checkpoints/sam2.1_hiera_base_plus.pt"
    echo "Using checkpoint: checkpoints/sam2.1_hiera_base_plus.pt"
elif [ -f "checkpoints/sam2_hiera_base_plus.pt" ]; then
    SAM2_CKPT="--sam2-checkpoint checkpoints/sam2_hiera_base_plus.pt"
    echo "Using checkpoint: checkpoints/sam2_hiera_base_plus.pt"
else
    echo "No SAM2 checkpoint found — using random init (download weights first for best results)"
fi

python scripts/train_medsam2.py \
    --splits-dir data/splits \
    --sam2-config "configs/sam2.1/sam2.1_hiera_b+" \
    --epochs 50 \
    --batch-size 2 \
    --lr 1e-4 \
    --lora-rank 8 \
    --lora-alpha 16 \
    --num-workers 4 \
    --device cuda \
    --checkpoint-dir checkpoints/medsam2_lora \
    $SAM2_CKPT

echo "=== DONE: $(date) ==="
