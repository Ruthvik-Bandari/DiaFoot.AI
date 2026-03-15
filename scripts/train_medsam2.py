"""DiaFoot.AI v2 — MedSAM2 LoRA Fine-Tuning Script.

Trains SAM2.1 (Hiera-B+) with LoRA adapters on DFU segmentation data.
Freezes the image encoder, applies LoRA to q_proj/v_proj, fine-tunes mask decoder.
Uses bounding box prompts auto-generated from ground truth masks.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.medsam2_finetune import LoRAConfig, apply_lora_to_model, mask_to_bbox

logger = logging.getLogger("train_medsam2")


class DFUSegDataset(Dataset):
    """Dataset for MedSAM2 fine-tuning — wound images only.

    Returns images at 1024x1024 (SAM2 native resolution) with
    binary masks and auto-generated bounding box prompts.
    """

    def __init__(
        self,
        split_csv: str | Path,
        image_size: int = 1024,
        include_classes: list[str] | None = None,
    ) -> None:
        self.image_size = image_size
        self.samples: list[dict] = []

        with open(split_csv) as f:
            for row in csv.DictReader(f):
                cls = row.get("class", "")
                if include_classes and cls not in include_classes:
                    continue
                self.samples.append(row)

        logger.info("Loaded %d samples from %s", len(self.samples), split_csv)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        image_path = sample.get("image_path") or sample.get("image", "")
        mask_path = sample.get("mask_path") or sample.get("mask", "")

        # Load image (BGR -> RGB)
        image = cv2.imread(image_path)
        if image is None:
            raise RuntimeError(f"Failed to load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize to SAM2 native resolution
        image = cv2.resize(image, (self.image_size, self.image_size))

        # Load and resize mask
        if mask_path and Path(mask_path).exists():
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
            else:
                mask = cv2.resize(mask, (self.image_size, self.image_size))
                mask = (mask > 127).astype(np.uint8)
        else:
            mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)

        # Convert to tensors
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)  # (1, H, W)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "filename": Path(image_path).name,
        }


def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """Differentiable Dice loss."""
    pred_flat = pred.sigmoid().flatten(1)
    target_flat = target.flatten(1)
    intersection = (pred_flat * target_flat).sum(1)
    return 1 - (2 * intersection + smooth) / (pred_flat.sum(1) + target_flat.sum(1) + smooth)


def combined_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Dice + BCE combined loss."""
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction="mean")
    dl = dice_loss(pred, target).mean()
    return bce + dl


def _tokens_to_bchw(feat: torch.Tensor, feat_size: tuple[int, int]) -> torch.Tensor:
    """Convert SAM2 tokenized feature map (HW, B, C) to (B, C, H, W)."""
    h, w = int(feat_size[0]), int(feat_size[1])
    return feat.permute(1, 2, 0).reshape(feat.shape[1], feat.shape[2], h, w)


def _discover_sam2_configs() -> list[str]:
    """Find all available SAM2 config names from the installed package."""
    import os

    import sam2

    configs = []
    cfg_root = os.path.join(os.path.dirname(sam2.__file__), "configs")
    if os.path.isdir(cfg_root):
        for root, _dirs, files in os.walk(cfg_root):
            for f in files:
                if f.endswith(".yaml"):
                    rel = os.path.relpath(os.path.join(root, f), cfg_root)
                    # Both with and without .yaml
                    configs.append(rel)
                    configs.append(rel.removesuffix(".yaml"))
                    # Some SAM2 builds expose Hydra root as pkg://sam2,
                    # which requires the explicit "configs/" prefix.
                    configs.append(f"configs/{rel}")
                    configs.append(f"configs/{rel.removesuffix('.yaml')}")
    return configs


def build_sam2_model(
    config_name: str = "sam2.1/sam2.1_hiera_b+",
    checkpoint_path: str | None = None,
    device: str = "cuda",
) -> torch.nn.Module:
    """Build SAM2 model, trying multiple config name formats."""
    from sam2.build_sam import build_sam2

    # Build candidate list: user-specified variants + fallbacks
    candidates = [
        config_name,
        config_name + ".yaml",
        "configs/" + config_name,
        "configs/" + config_name + ".yaml",
        # Fallback: sam2 (not sam2.1) variants
        config_name.replace("sam2.1/sam2.1_", "sam2/sam2_"),
        config_name.replace("sam2.1/sam2.1_", "sam2/sam2_") + ".yaml",
        "configs/" + config_name.replace("sam2.1/sam2.1_", "sam2/sam2_"),
        "configs/" + config_name.replace("sam2.1/sam2.1_", "sam2/sam2_") + ".yaml",
    ]

    # De-duplicate while preserving order
    seen = set()
    candidates = [c for c in candidates if not (c in seen or seen.add(c))]

    # Discover what's actually installed and try those too
    available = _discover_sam2_configs()
    if available:
        logger.info("Available SAM2 configs: %s", [c for c in available if c.endswith(".yaml")])
        # Find best match from available configs (prefer hiera_b+)
        for avail in available:
            if "hiera_b+" in avail and avail not in candidates:
                candidates.append(avail)

    errors = []
    for cfg in candidates:
        try:
            model = build_sam2(cfg, ckpt_path=checkpoint_path, device=device)
            logger.info("Loaded SAM2 with config: %s", cfg)
            return model
        except Exception as e:
            errors.append(f"  {cfg}: {e}")
            continue

    raise RuntimeError(
        f"Could not load SAM2. Tried configs:\n" + "\n".join(errors)
    )


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int,
    scaler: torch.amp.GradScaler | None = None,
) -> dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    n_batches = 0

    for batch in dataloader:
        images = batch["image"].to(device)  # (B, 3, 1024, 1024)
        masks = batch["mask"].to(device)    # (B, 1, 1024, 1024)

        # Generate bbox prompts from GT masks
        bboxes = mask_to_bbox(masks.squeeze(1), padding=20).to(device)  # (B, 4)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            # SAM2 image predictor interface
            model.eval()  # Encoder in eval mode for BN
            # Set image batch
            backbone_out = model.forward_image(images)
            # Get features from backbone
            _, vision_feats, vision_pos_embeds, feat_sizes = model._prepare_backbone_features(backbone_out)

            # Prepare prompts (bbox)
            # SAM2 expects (B, num_points, 2) for points, (B, 4) for boxes
            model.train()  # Back to train for decoder

            # Predict masks using the SAM2 forward path
            # We use the low-level API for training
            B = images.shape[0]
            pred_masks_list = []
            for i in range(B):
                # Extract per-image features and convert token layout (HW,B,C) -> (B,C,H,W)
                feats_per_img = [f[:, i:i+1] for f in vision_feats]
                image_embed = _tokens_to_bchw(feats_per_img[-1], feat_sizes[-1])
                high_res_feats = [
                    _tokens_to_bchw(feats_per_img[0], feat_sizes[0]),
                    _tokens_to_bchw(feats_per_img[1], feat_sizes[1]),
                ]

                sparse_embeddings, dense_embeddings = model.sam_prompt_encoder(
                    points=None,
                    boxes=bboxes[i:i+1],
                    masks=None,
                )

                low_res_masks, iou_pred, _, _ = model.sam_mask_decoder(
                    image_embeddings=image_embed,
                    image_pe=model.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                    repeat_image=False,
                    high_res_features=high_res_feats,
                )

                # Upscale to original resolution
                pred_mask = F.interpolate(
                    low_res_masks,
                    size=(images.shape[2], images.shape[3]),
                    mode="bilinear",
                    align_corners=False,
                )
                pred_masks_list.append(pred_mask.squeeze(0))

            pred_masks = torch.stack(pred_masks_list)  # (B, 1, H, W)
            loss = combined_loss(pred_masks, masks)

        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()

        # Metrics
        with torch.no_grad():
            pred_binary = (pred_masks.sigmoid() > 0.5).float()
            intersection = (pred_binary * masks).sum()
            union = pred_binary.sum() + masks.sum()
            batch_dice = (2 * intersection / (union + 1e-6)).item()

        total_loss += loss.item()
        total_dice += batch_dice
        n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "dice": total_dice / max(n_batches, 1),
    }


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str,
) -> dict[str, float]:
    """Validate model."""
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    n_batches = 0

    for batch in dataloader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        bboxes = mask_to_bbox(masks.squeeze(1), padding=20).to(device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            backbone_out = model.forward_image(images)
            _, vision_feats, vision_pos_embeds, feat_sizes = model._prepare_backbone_features(backbone_out)

            B = images.shape[0]
            pred_masks_list = []
            for i in range(B):
                feats_per_img = [f[:, i:i+1] for f in vision_feats]
                image_embed = _tokens_to_bchw(feats_per_img[-1], feat_sizes[-1])
                high_res_feats = [
                    _tokens_to_bchw(feats_per_img[0], feat_sizes[0]),
                    _tokens_to_bchw(feats_per_img[1], feat_sizes[1]),
                ]

                sparse_embeddings, dense_embeddings = model.sam_prompt_encoder(
                    points=None,
                    boxes=bboxes[i:i+1],
                    masks=None,
                )

                low_res_masks, iou_pred, _, _ = model.sam_mask_decoder(
                    image_embeddings=image_embed,
                    image_pe=model.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                    repeat_image=False,
                    high_res_features=high_res_feats,
                )

                pred_mask = F.interpolate(
                    low_res_masks,
                    size=(images.shape[2], images.shape[3]),
                    mode="bilinear",
                    align_corners=False,
                )
                pred_masks_list.append(pred_mask.squeeze(0))

            pred_masks = torch.stack(pred_masks_list)
            loss = combined_loss(pred_masks, masks)

        pred_binary = (pred_masks.sigmoid() > 0.5).float()
        intersection = (pred_binary * masks).sum()
        union = pred_binary.sum() + masks.sum()
        batch_dice = (2 * intersection / (union + 1e-6)).item()

        iou_union = pred_binary.sum() + masks.sum() - intersection
        batch_iou = (intersection / (iou_union + 1e-6)).item()

        total_loss += loss.item()
        total_dice += batch_dice
        total_iou += batch_iou
        n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "dice": total_dice / max(n_batches, 1),
        "iou": total_iou / max(n_batches, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="MedSAM2 LoRA Fine-Tuning")
    parser.add_argument("--splits-dir", type=str, default="data/splits")
    parser.add_argument("--sam2-config", type=str, default="sam2.1/sam2.1_hiera_b+")
    parser.add_argument("--sam2-checkpoint", type=str, default=None,
                        help="Path to SAM2 pretrained checkpoint (.pt)")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/medsam2_lora")
    parser.add_argument("--image-size", type=int, default=1024)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    torch.manual_seed(42)
    device = args.device

    # Build SAM2 model
    logger.info("Loading SAM2 model...")
    model = build_sam2_model(
        config_name=args.sam2_config,
        checkpoint_path=args.sam2_checkpoint,
        device=device,
    )

    # Freeze entire model first
    for param in model.parameters():
        param.requires_grad = False

    # Apply LoRA to image encoder attention
    lora_config = LoRAConfig(
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=0.05,
        target_modules=("q_proj", "v_proj"),
    )
    # Apply in-place to the encoder; do not overwrite the top-level SAM2 model.
    _, n_lora_params = apply_lora_to_model(model.image_encoder, lora_config)
    if n_lora_params == 0:
        logger.warning(
            "No LoRA target modules matched in image encoder; continuing with decoder fine-tuning only."
        )
    logger.info("LoRA params added to image encoder: %d", n_lora_params)

    # Unfreeze mask decoder
    for param in model.sam_mask_decoder.parameters():
        param.requires_grad = True
    decoder_params = sum(p.numel() for p in model.sam_mask_decoder.parameters())
    logger.info("Mask decoder params (unfrozen): %d", decoder_params)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info("Trainable: %d / %d (%.2f%%)", trainable, total, trainable / total * 100)

    model = model.to(device)

    # Data — wound images only (DFU + non-DFU)
    train_ds = DFUSegDataset(
        Path(args.splits_dir) / "train.csv",
        image_size=args.image_size,
        include_classes=["dfu", "non_dfu"],
    )
    val_ds = DFUSegDataset(
        Path(args.splits_dir) / "val.csv",
        image_size=args.image_size,
        include_classes=["dfu", "non_dfu"],
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # Optimizer — only trainable params
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6,
    )
    scaler = torch.amp.GradScaler("cuda")

    # Checkpointing
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_dice = 0.0

    logger.info("Starting training: %d epochs, %d train / %d val batches",
                args.epochs, len(train_loader), len(val_loader))

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(model, train_loader, optimizer, device, epoch, scaler)
        val_metrics = validate(model, val_loader, device)
        scheduler.step()

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        logger.info(
            "Epoch %03d/%03d [%.0fs] lr=%.2e | "
            "train loss=%.4f dice=%.4f | "
            "val loss=%.4f dice=%.4f iou=%.4f",
            epoch, args.epochs, elapsed, lr,
            train_metrics["loss"], train_metrics["dice"],
            val_metrics["loss"], val_metrics["dice"], val_metrics["iou"],
        )

        # Save best
        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            ckpt_path = ckpt_dir / f"best_epoch{epoch:03d}_{best_dice:.4f}.pt"
            # Save only trainable weights (LoRA + decoder)
            trainable_state = {
                k: v for k, v in model.state_dict().items()
                if any(p.data_ptr() == v.data_ptr()
                       for p in model.parameters() if p.requires_grad)
            }
            torch.save({
                "epoch": epoch,
                "model_state_dict": trainable_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "best_dice": best_dice,
                "val_metrics": val_metrics,
                "lora_config": {
                    "rank": args.lora_rank,
                    "alpha": args.lora_alpha,
                },
            }, ckpt_path)
            logger.info("New best! Dice=%.4f saved to %s", best_dice, ckpt_path)

        # Save latest every 10 epochs
        if epoch % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_dice": best_dice,
            }, ckpt_dir / "latest.pt")

    logger.info("Training complete. Best val Dice: %.4f", best_dice)


if __name__ == "__main__":
    main()
