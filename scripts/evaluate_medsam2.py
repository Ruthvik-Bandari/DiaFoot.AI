#!/usr/bin/env python3
"""Evaluate MedSAM2 checkpoint on a split CSV.

This uses the same model construction + validation path as train_medsam2.py,
so MedSAM2 checkpoints are loaded with matching architecture.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import scripts.train_medsam2 as tm
from src.models.medsam2_finetune import LoRAConfig, apply_lora_to_model

logger = logging.getLogger("evaluate_medsam2")


def _extract_state_dict(ckpt_obj: dict) -> tuple[dict, str]:
    """Extract model weights from common checkpoint formats."""
    if isinstance(ckpt_obj, dict):
        for key in ("lora_state_dict", "model_state_dict", "state_dict", "model"):
            value = ckpt_obj.get(key)
            if isinstance(value, dict):
                return value, key
        return ckpt_obj, "root"
    raise RuntimeError("Unsupported checkpoint format: expected dict-like object")


def _normalize_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Strip common wrappers (DDP/module/model) from checkpoint parameter keys."""
    normalized: dict[str, torch.Tensor] = {}
    prefixes = ("module.", "model.", "sam_model.")
    for key, value in state_dict.items():
        nk = key
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p):]
        normalized[nk] = value
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MedSAM2 checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--splits-dir", type=str, default="data/splits")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--sam2-config", type=str, default="configs/sam2.1/sam2.1_hiera_b+")
    parser.add_argument(
        "--sam2-checkpoint",
        type=str,
        default="models/sam2_weights/sam2.1_hiera_base_plus.pt",
        help="Base SAM2 pretrained checkpoint (.pt)",
    )
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument(
        "--include-classes",
        type=str,
        default="dfu,non_dfu",
        help="Comma-separated classes to evaluate (e.g. 'dfu,non_dfu'). Use 'all' for no filtering.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out-json", type=str, default="results/segmentation_metrics_medsam2.json")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    device = args.device if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    # Build model exactly like training.
    model = tm.build_sam2_model(
        config_name=args.sam2_config,
        checkpoint_path=args.sam2_checkpoint,
        device=device,
    )

    for p in model.parameters():
        p.requires_grad = False

    lora_cfg = LoRAConfig(
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=0.05,
        target_modules=("q_proj", "v_proj"),
    )
    _, n_lora_params = apply_lora_to_model(model.image_encoder, lora_cfg)
    for p in model.sam_mask_decoder.parameters():
        p.requires_grad = True

    logger.info("LoRA params matched: %d", n_lora_params)

    # Load fine-tuned checkpoint (often only trainable subset, so strict=False).
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    raw_state, source_key = _extract_state_dict(ckpt)
    state = _normalize_state_dict_keys(raw_state)

    model_state = model.state_dict()
    filtered = {k: v for k, v in state.items() if k in model_state and model_state[k].shape == v.shape}
    matched = len(filtered)
    if matched == 0:
        raise RuntimeError(
            "No checkpoint weights matched model keys. "
            "Check sam2 config/checkpoint and training checkpoint compatibility."
        )

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    logger.info("Loaded checkpoint: %s", args.checkpoint)
    logger.info("Checkpoint state source: %s", source_key)
    logger.info("Matched keys: %d | Missing keys: %d | Unexpected keys: %d", matched, len(missing), len(unexpected))

    model = model.to(device)

    # Dataset/loader
    include_classes = None
    if args.include_classes.strip().lower() != "all":
        include_classes = [c.strip() for c in args.include_classes.split(",") if c.strip()]

    split_csv = Path(args.splits_dir) / f"{args.split}.csv"
    ds = tm.DFUSegDataset(
        split_csv=split_csv,
        image_size=args.image_size,
        include_classes=include_classes,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    metrics = tm.validate(model, loader, device)
    logger.info("Metrics: %s", metrics)

    output = {
        "checkpoint": args.checkpoint,
        "sam2_config": args.sam2_config,
        "sam2_checkpoint": args.sam2_checkpoint,
        "split": args.split,
        "include_classes": include_classes if include_classes is not None else "all",
        "num_samples": len(ds),
        "metrics": {k: float(v) for k, v in metrics.items()},
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    logger.info("Saved to %s", out_path)

    print("\n=== MedSAM2 Evaluation ===")
    for k, v in output["metrics"].items():
        print(f"{k:>8}: {v:.4f}")


if __name__ == "__main__":
    main()
