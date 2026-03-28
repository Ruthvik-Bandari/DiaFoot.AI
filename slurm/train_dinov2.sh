#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# DiaFoot.AI v2 — DINOv2 Training (Single GPU)
#
# Three-phase training strategy:
#   Phase 1: Linear Probe (frozen backbone, train head/decoder only)
#   Phase 2: LoRA Fine-tuning (add LoRA adapters to attention layers)
#   Phase 3: Partial Unfreeze (unfreeze last N blocks, very low LR)
#
# Usage:
#   sbatch slurm/train_dinov2.sh                          # Full pipeline
#   sbatch slurm/train_dinov2.sh classify                 # Classifier only
#   sbatch slurm/train_dinov2.sh segment                  # Segmenter only
#   BACKBONE=dinov2_vitl14 sbatch slurm/train_dinov2.sh   # Use ViT-L backbone
# ═══════════════════════════════════════════════════════════════════════════════
#SBATCH --job-name=diafoot-dinov2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/%j_%x.out
#SBATCH --error=logs/slurm/%j_%x.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bandari.r@northeastern.edu

set -euo pipefail
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_ROOT"

module purge && module load cuda/12.8.0 python/3.12
source .venv/bin/activate
export PYTHONPATH="$PROJECT_ROOT"

TASK=${1:-all}
BACKBONE=${BACKBONE:-dinov2_vitb14}
EPOCHS_P1=${EPOCHS_P1:-20}
EPOCHS_P2=${EPOCHS_P2:-50}
BATCH_SIZE=${BATCH_SIZE:-16}

echo "Job ${SLURM_JOB_ID} | Node $(hostname) | GPU ${CUDA_VISIBLE_DEVICES:-?} | $(date)"
echo "Task: ${TASK} | Backbone: ${BACKBONE} | Batch: ${BATCH_SIZE}"

python -c "
import torch; assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(), 'GPU/BF16 check failed'
print(f'PyTorch {torch.__version__} | {torch.cuda.get_device_name(0)} | BF16 OK')
"

mkdir -p logs/slurm checkpoints/dinov2_classifier checkpoints/dinov2_segmenter

# ── Phase 1: Linear Probe (frozen backbone) ──────────────────────────────────
if [[ "$TASK" == "all" || "$TASK" == "classify" ]]; then
    echo ""
    echo "═══ Phase 1: DINOv2 Classifier (linear probe) ═══"
    python scripts/train.py \
        --task classify \
        --backbone "$BACKBONE" \
        --epochs "$EPOCHS_P1" \
        --batch-size "$BATCH_SIZE" \
        --lr 1e-3 \
        --device cuda \
        --verbose
fi

if [[ "$TASK" == "all" || "$TASK" == "segment" ]]; then
    echo ""
    echo "═══ Phase 1: DINOv2 Segmenter (linear probe) ═══"
    python scripts/train.py \
        --task segment \
        --backbone "$BACKBONE" \
        --epochs "$EPOCHS_P1" \
        --batch-size "$BATCH_SIZE" \
        --lr 1e-3 \
        --device cuda \
        --verbose
fi

# ── Phase 2: LoRA Fine-tuning ────────────────────────────────────────────────
if [[ "$TASK" == "all" || "$TASK" == "classify" ]]; then
    echo ""
    echo "═══ Phase 2: DINOv2 Classifier (LoRA fine-tune) ═══"
    python scripts/train.py \
        --task classify \
        --backbone "$BACKBONE" \
        --epochs "$EPOCHS_P2" \
        --batch-size "$BATCH_SIZE" \
        --lr 5e-5 \
        --use-lora \
        --lora-rank 8 \
        --lora-alpha 16 \
        --device cuda \
        --verbose
fi

if [[ "$TASK" == "all" || "$TASK" == "segment" ]]; then
    echo ""
    echo "═══ Phase 2: DINOv2 Segmenter (LoRA fine-tune) ═══"
    python scripts/train.py \
        --task segment \
        --backbone "$BACKBONE" \
        --epochs "$EPOCHS_P2" \
        --batch-size "$BATCH_SIZE" \
        --lr 5e-5 \
        --use-lora \
        --lora-rank 8 \
        --lora-alpha 16 \
        --device cuda \
        --verbose
fi

echo ""
echo "═══ Training Complete ═══ $(date)"
