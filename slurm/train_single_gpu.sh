#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# DiaFoot.AI v2 — Single H100 GPU Training
# Usage: sbatch slurm/train_single_gpu.sh [config_path]
# ═══════════════════════════════════════════════════════════════════════════════
#SBATCH --job-name=diafootai-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/%j_%x.out
#SBATCH --error=logs/slurm/%j_%x.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bandari.r@northeastern.edu

set -euo pipefail
module purge && module load cuda/12.4 python/3.12
source .venv/bin/activate

echo "Job ${SLURM_JOB_ID} | Node $(hostname) | GPU ${CUDA_VISIBLE_DEVICES:-?} | $(date)"

python -c "
import torch; assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(), 'GPU/BF16 check failed'
print(f'PyTorch {torch.__version__} | {torch.cuda.get_device_name(0)} | BF16 OK')
"

export TORCH_COMPILE_BACKEND=inductor
export TORCHINDUCTOR_MODE=reduce-overhead
mkdir -p logs/slurm

python scripts/train.py \
    --config "${1:-configs/training/baseline.yaml}" \
    --device cuda \
    --precision bf16-mixed \
    --compile true \
    --seed 42

echo "Finished: $(date)"
