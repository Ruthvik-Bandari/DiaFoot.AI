"""DiaFoot.AI — one cell of the training-data-composition study.

Trains a single segmentation model under a controlled TRAINING composition,
then evaluates it on the fixed, full, clean test set and writes one
provenance-stamped JSON. Running this once per (architecture, composition, seed)
cell and aggregating the JSONs produces the paper's composition table.

Why a dedicated driver (vs scripts/train.py + evaluate_all.py):
  * concurrency-safe — each cell writes its filtered split CSVs to its OWN
    directory, so a SLURM array of cells cannot clobber a shared ``filtered/``
    dir (the existing build_dataloaders bug);
  * both architectures — U-Net++ AND the DINOv2 segmenter can be swept across
    compositions (train.py hardcodes DFU-only for DINOv2);
  * negative-ratio sweep — a deterministic dose of negatives, not just the
    3 categorical presets;
  * every number is stamped with the split hash, checkpoint hash, seed, and git
    commit, so the composition table is reproducible and auditable.

The TEST set is ALWAYS the full clean test split (all classes); the DFU-only
numbers are the DFU slice of that identical set, so every cell is judged on the
same images.

Usage (one cell):
    python scripts/run_composition_experiment.py --arch unetpp --composition dfu_only --seed 42
    python scripts/run_composition_experiment.py --arch unetpp --neg-frac 0.5 --seed 42
    python scripts/run_composition_experiment.py --arch dinov2 --composition all --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.augmentation import get_train_transforms, get_val_transforms
from src.data.composition import (
    class_counts,
    random_mixed,
    read_split_csv,
    select_composition,
    subsample_negatives,
    write_split_csv,
)
from src.data.torch_dataset import DFUDataset
from src.evaluation.composition_report import build_provenance, summarize_run
from src.evaluation.segmentation_eval import run_segmentation_eval
from src.training.losses import DiceCELoss
from src.training.schedulers import CosineAnnealingWithWarmup
from src.training.trainer import TrainConfig, Trainer

logger = logging.getLogger("composition")

# Per-architecture input size: U-Net++ and SegFormer (smp) require H,W divisible
# by 32 (512); DINOv2 requires divisibility by the patch size 14 (518). Using the
# wrong size crashes the model, so it is chosen from the architecture, not left to
# a shared default.
ARCH_IMAGE_SIZE = {"unetpp": 512, "segformer": 512, "dinov2": 518}


def _set_seed(seed: int) -> None:
    """Seed all RNGs that affect training/data selection."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(arch: str, encoder_weights: str | None = "imagenet") -> torch.nn.Module:
    """Build the segmentation model for the requested architecture."""
    if arch == "unetpp":
        from src.models.unetpp import build_unetpp

        return build_unetpp(
            encoder_name="efficientnet-b4",
            encoder_weights=encoder_weights,
            classes=1,
            decoder_attention_type="scse",
        )
    if arch == "segformer":
        import segmentation_models_pytorch as smp

        # SegFormer-B0 = MiT-B0 encoder + the SegFormer all-MLP decoder.
        return smp.Segformer(
            encoder_name="mit_b0",
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1,
        )
    if arch == "dinov2":
        from src.models.dinov2_segmenter import DINOv2Segmenter

        return DINOv2Segmenter(backbone="dinov2_vitb14", num_classes=1, freeze_backbone=True)
    msg = f"unknown arch {arch!r}; expected 'unetpp', 'segformer', or 'dinov2'"
    raise ValueError(msg)


def make_composition_rows(
    rows: list[dict[str, str]],
    composition: str | None,
    neg_frac: float | None,
    seed: int,
) -> list[dict[str, str]]:
    """Apply a categorical composition, the size-matched random mix, OR the negative-ratio."""
    if composition == "random_mixed":
        return random_mixed(rows, seed=seed)
    if composition is not None:
        return select_composition(rows, composition)
    if neg_frac is not None:
        return subsample_negatives(rows, neg_frac, seed=seed)
    msg = "exactly one of composition / neg_frac must be provided"
    raise ValueError(msg)


def pick_best_checkpoint(ckpt_dir: Path, mode: str = "min") -> Path:
    """Pick the best ``best_epochNNN_<metric>.pt`` by the value in its filename.

    The trainer only writes a checkpoint when the monitored metric improves, so
    for ``val/loss`` (min) the best is the lowest-valued file. Returns its path.
    """
    ckpts = sorted(ckpt_dir.glob("best_epoch*_*.pt"))
    if not ckpts:
        msg = f"no checkpoints found in {ckpt_dir}"
        raise FileNotFoundError(msg)

    def metric_of(p: Path) -> float:
        return float(p.stem.rsplit("_", 1)[1])

    return min(ckpts, key=metric_of) if mode == "min" else max(ckpts, key=metric_of)


def dfu_test_indices(test_ds: DFUDataset, n: int) -> list[int]:
    """First ``n`` DFU-labelled indices of the test set (stable across cells).

    Reads labels from the dataset's CSV rows (no image loading), so every
    architecture/composition/fold saves predictions for the *same* wounds — a
    prerequisite for the side-by-side qualitative comparison figure.
    """
    dfu = [i for i, s in enumerate(test_ds.samples) if s.get("class") == "dfu"]
    return dfu[:n]


def save_qualitative(
    model: torch.nn.Module,
    test_ds: DFUDataset,
    indices: list[int],
    device: str,
    out_dir: Path,
) -> None:
    """Save predicted binary masks (PNG) for fixed test indices for the figure."""
    import cv2
    import numpy as np

    out_dir.mkdir(parents=True, exist_ok=True)
    model = model.to(device).eval()
    with torch.no_grad():
        for idx in indices:
            sample = test_ds[idx]
            image = sample["image"].unsqueeze(0).to(device)
            logits = model(image)
            if isinstance(logits, dict):
                logits = logits.get("seg_logits", logits)
            pred = (torch.sigmoid(logits) > 0.5).squeeze().cpu().numpy().astype(np.uint8) * 255
            cv2.imwrite(str(out_dir / f"pred_idx{idx:05d}.png"), pred)


def main() -> None:
    """Train + evaluate one composition cell and write its result JSON."""
    parser = argparse.ArgumentParser(description="One cell of the composition study")
    parser.add_argument("--arch", choices=["unetpp", "segformer", "dinov2"], required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--composition",
        choices=["dfu_only", "dfu_healthy", "dfu_nondfu", "all", "random_mixed"],
    )
    group.add_argument("--neg-frac", type=float, help="Fraction of the negative pool (0..1)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="CV fold index; reads {cv-dir}/fold{N}/{train,val}.csv instead of the global pool",
    )
    parser.add_argument("--cv-dir", default="data/splits/cv")
    parser.add_argument(
        "--save-qualitative",
        action="store_true",
        help="Save predicted masks for a fixed set of test images (for the comparison figure)",
    )
    parser.add_argument("--n-qualitative", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Override input size (default: 512 for unetpp, 518 for dinov2)",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument(
        "--encoder-weights",
        default="imagenet",
        help="U-Net++ encoder init: 'imagenet' (default) or 'none' (random, e.g. smoke tests)",
    )
    parser.add_argument("--work-dir", default="checkpoints/composition")
    parser.add_argument("--results-dir", default="results/composition")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # Per-fold seed offset (matches the repo's CV convention) so each fold gets a
    # distinct training seed AND a distinct random-mixed / subsample draw.
    cell_seed = args.seed + (args.fold or 0)
    _set_seed(cell_seed)

    comp_label = args.composition if args.composition else f"negfrac{args.neg_frac:.2f}"
    fold_tag = "" if args.fold is None else f"_fold{args.fold}"
    run_tag = f"{args.arch}_{comp_label}_seed{args.seed}{fold_tag}"
    logger.info("=== composition cell: %s ===", run_tag)

    splits = Path(args.splits_dir)
    # Pool source: a shared CV fold (train+val partition) when --fold is set,
    # otherwise the global train/val split.
    if args.fold is not None:
        pool_dir = Path(args.cv_dir) / f"fold{args.fold}"
        train_src, val_src = pool_dir / "train.csv", pool_dir / "val.csv"
    else:
        train_src, val_src = splits / "train.csv", splits / "val.csv"

    # Each cell gets its OWN filtered-split dir (concurrency-safe for array jobs).
    filtered_dir = splits / "_composition" / run_tag
    train_rows = make_composition_rows(
        read_split_csv(train_src), args.composition, args.neg_frac, cell_seed
    )
    val_rows = make_composition_rows(
        read_split_csv(val_src), args.composition, args.neg_frac, cell_seed
    )
    train_csv = filtered_dir / "train.csv"
    val_csv = filtered_dir / "val.csv"
    write_split_csv(train_rows, train_csv)
    write_split_csv(val_rows, val_csv)
    n_train_by_class = class_counts(train_rows)
    n_val_by_class = class_counts(val_rows)
    logger.info("train composition: %s | val: %s", n_train_by_class, n_val_by_class)

    # Data loaders. TRAIN/VAL are composition-filtered; TEST is the full clean set.
    image_size = args.image_size or ARCH_IMAGE_SIZE[args.arch]
    logger.info("input size: %d (arch=%s)", image_size, args.arch)
    train_ds = DFUDataset(str(train_csv), transform=get_train_transforms(image_size))
    val_ds = DFUDataset(str(val_csv), transform=get_val_transforms(image_size))
    test_csv = splits / "test.csv"
    test_ds = DFUDataset(
        str(test_csv), transform=get_val_transforms(image_size), return_metadata=True
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Model + training.
    encoder_weights = None if args.encoder_weights.lower() == "none" else args.encoder_weights
    model = build_model(args.arch, encoder_weights=encoder_weights)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingWithWarmup(optimizer, warmup_epochs=5, max_epochs=args.epochs)
    loss_fn = DiceCELoss()

    ckpt_dir = Path(args.work_dir) / run_tag
    train_config = TrainConfig(
        epochs=args.epochs,
        precision="bf16-mixed",
        compile_model=False,
        gradient_clip=1.0,
        checkpoint_dir=str(ckpt_dir),
        monitor_metric="val/loss",
        monitor_mode="min",
        device=args.device,
        early_stopping_patience=15,
    )
    trainer = Trainer(model=model, config=train_config)
    # Learning curves (saved for the appendix, per the manuscript instructions).
    history = trainer.fit(train_loader, val_loader, loss_fn, optimizer, scheduler)

    # Evaluate the best checkpoint on the FULL clean test set.
    best_ckpt = pick_best_checkpoint(ckpt_dir, mode="min")
    logger.info("best checkpoint: %s", best_ckpt)
    state = torch.load(str(best_ckpt), map_location="cpu", weights_only=True)
    model.load_state_dict(state["model_state_dict"])

    device = args.device if torch.cuda.is_available() else "cpu"
    per_image, labels = run_segmentation_eval(model, test_loader, device)
    summary = summarize_run(per_image, labels, seed=cell_seed)

    qualitative_dir = None
    if args.save_qualitative:
        qualitative_dir = Path(args.results_dir) / "qualitative" / run_tag
        indices = dfu_test_indices(test_ds, args.n_qualitative)
        save_qualitative(model, test_ds, indices, device, qualitative_dir)
        logger.info("saved %d qualitative masks -> %s", len(indices), qualitative_dir)

    provenance = build_provenance(
        split_csv=test_csv,
        checkpoint=best_ckpt,
        arch=args.arch,
        composition=comp_label,
        seed=args.seed,
        extra={
            "fold": args.fold,
            "cell_seed": cell_seed,
            "image_size": image_size,
            "lr": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "n_train_by_class": n_train_by_class,
            "n_val_by_class": n_val_by_class,
            "n_test": len(labels),
            "train_csv": str(train_csv),
        },
    )

    result = {
        "run_tag": run_tag,
        "arch": args.arch,
        "composition": comp_label,
        "seed": args.seed,
        "fold": args.fold,
        "summary": summary,
        "learning_curve": history,
        "qualitative_dir": str(qualitative_dir) if qualitative_dir else None,
        "provenance": provenance,
    }
    out_path = Path(args.results_dir) / f"{run_tag}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    dfu = summary["dfu_only"]["dice_ci"]
    fp = summary["false_positive_on_empty"]
    logger.info(
        "DONE %s | DFU Dice %.4f [%.4f, %.4f] (n=%d) | FP-on-empty %d/%d | -> %s",
        run_tag,
        dfu["mean"],
        dfu["ci_low"],
        dfu["ci_high"],
        dfu["n"],
        fp["n_false_positive"],
        fp["n_empty"],
        out_path,
    )


if __name__ == "__main__":
    main()
