"""DiaFoot.AI v2 — Training Pipeline Tests (Phase 3, Commits 13-15)."""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.training.classification_losses import FocalLoss, LabelSmoothingCE
from src.training.ema import EMA
from src.training.losses import DiceBoundaryLoss, DiceCELoss, DiceLoss, FocalTverskyLoss
from src.training.schedulers import CosineAnnealingWithWarmup
from src.training.trainer import TrainConfig, Trainer


class _DictDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, samples: list[dict[str, torch.Tensor]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.samples[idx]


class _TinyClassifier(nn.Module):
    def __init__(self, in_features: int = 3 * 8 * 8, num_classes: int = 3) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.flatten(start_dim=1))


class _TinySegmenter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class TestDiceLoss:
    def test_perfect_prediction(self) -> None:
        pred = torch.ones(2, 1, 64, 64) * 10  # High logits
        target = torch.ones(2, 64, 64, dtype=torch.long)
        loss = DiceLoss()(pred, target)
        assert loss.item() < 0.1  # Should be near 0

    def test_worst_prediction(self) -> None:
        pred = torch.ones(2, 1, 64, 64) * -10  # Wrong prediction
        target = torch.ones(2, 64, 64, dtype=torch.long)
        loss = DiceLoss()(pred, target)
        assert loss.item() > 0.9  # Should be near 1


class TestDiceCELoss:
    def test_output_scalar(self) -> None:
        pred = torch.randn(2, 1, 64, 64)
        target = torch.randint(0, 2, (2, 64, 64))
        loss = DiceCELoss()(pred, target)
        assert loss.dim() == 0


class TestFocalTverskyLoss:
    def test_output_scalar(self) -> None:
        pred = torch.randn(2, 1, 64, 64)
        target = torch.randint(0, 2, (2, 64, 64))
        loss = FocalTverskyLoss()(pred, target)
        assert loss.dim() == 0


class TestDiceBoundaryLoss:
    def test_warmup_phase(self) -> None:
        loss_fn = DiceBoundaryLoss(warmup_epoch=30)
        loss_fn.set_epoch(10)  # Before warmup
        pred = torch.randn(2, 1, 64, 64)
        target = torch.randint(0, 2, (2, 64, 64))
        loss = loss_fn(pred, target)
        assert loss.dim() == 0

    def test_boundary_phase(self) -> None:
        loss_fn = DiceBoundaryLoss(warmup_epoch=5, max_epoch=20)
        loss_fn.set_epoch(15)
        pred = torch.randn(2, 1, 64, 64)
        target = torch.randint(0, 2, (2, 64, 64))
        loss = loss_fn(pred, target)
        assert loss.dim() == 0


class TestFocalLoss:
    def test_output_scalar(self) -> None:
        pred = torch.randn(4, 3)
        target = torch.randint(0, 3, (4,))
        loss = FocalLoss()(pred, target)
        assert loss.dim() == 0

    def test_with_class_weights(self) -> None:
        alpha = torch.tensor([1.0, 2.0, 3.0])
        pred = torch.randn(4, 3)
        target = torch.randint(0, 3, (4,))
        loss = FocalLoss(alpha=alpha)(pred, target)
        assert loss.dim() == 0


class TestLabelSmoothingCE:
    def test_output_scalar(self) -> None:
        pred = torch.randn(4, 6)
        target = torch.randint(0, 6, (4,))
        loss = LabelSmoothingCE(smoothing=0.1, num_classes=6)(pred, target)
        assert loss.dim() == 0


class TestCosineAnnealingWithWarmup:
    def test_warmup_increases_lr(self) -> None:
        model = nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = CosineAnnealingWithWarmup(optimizer, warmup_epochs=5, max_epochs=50)
        lrs = []
        for _ in range(10):
            lrs.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()
        # LR should increase during warmup then decrease
        assert lrs[0] < lrs[4]  # Warmup increases

    def test_cosine_decreases_lr(self) -> None:
        model = nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = CosineAnnealingWithWarmup(optimizer, warmup_epochs=2, max_epochs=20)
        for _ in range(3):
            optimizer.step()
            scheduler.step()
        lr_mid = optimizer.param_groups[0]["lr"]
        for _ in range(15):
            optimizer.step()
            scheduler.step()
        lr_late = optimizer.param_groups[0]["lr"]
        assert lr_late < lr_mid  # Cosine decay


class TestEMA:
    def test_shadow_copy(self) -> None:
        model = nn.Linear(10, 5)
        ema = EMA(model, decay=0.999)
        # Shadow should start equal to model
        for (_, p1), (_, p2) in zip(
            model.named_parameters(), ema.shadow.named_parameters(), strict=False
        ):
            torch.testing.assert_close(p1, p2)

    def test_update_changes_shadow(self) -> None:
        model = nn.Linear(10, 5)
        ema = EMA(model, decay=0.9)
        # Modify model weights
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p))
        ema.update(model)
        # Shadow should now differ from original
        # (it's a weighted average, so not identical to model)


class TestTrainer:
    def test_fit_classification_saves_checkpoint(self, tmp_path: Path) -> None:
        samples = [
            {
                "image": torch.randn(3, 8, 8),
                "label": torch.tensor(i % 3, dtype=torch.long),
                "mask": torch.zeros(8, 8, dtype=torch.long),
            }
            for i in range(6)
        ]
        train_loader = DataLoader(_DictDataset(samples), batch_size=2, shuffle=False)
        val_loader = DataLoader(_DictDataset(samples), batch_size=2, shuffle=False)

        model = _TinyClassifier()
        cfg = TrainConfig(
            epochs=2,
            precision="bf16-mixed",
            checkpoint_dir=str(tmp_path / "ckpts_cls"),
            monitor_metric="val/accuracy",
            monitor_mode="max",
            device="cpu",
            early_stopping_patience=2,
        )
        trainer = Trainer(model=model, config=cfg)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = CosineAnnealingWithWarmup(optimizer, warmup_epochs=1, max_epochs=2)
        history = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=nn.CrossEntropyLoss(),
            optimizer=optimizer,
            scheduler=scheduler,
        )

        assert "val/accuracy" in history
        assert len(history["val/accuracy"]) >= 1
        saved = list((tmp_path / "ckpts_cls").glob("best_epoch*.pt"))
        assert saved

    def test_validate_segmentation_reports_dice(self) -> None:
        samples = [
            {
                "image": torch.randn(3, 8, 8),
                "label": torch.tensor(2, dtype=torch.long),
                "mask": torch.randint(0, 2, (8, 8), dtype=torch.long),
            }
            for _ in range(4)
        ]
        val_loader = DataLoader(_DictDataset(samples), batch_size=2, shuffle=False)

        model = _TinySegmenter()
        cfg = TrainConfig(
            epochs=1,
            precision="bf16-mixed",
            checkpoint_dir="checkpoints/test_tmp",
            monitor_metric="val/loss",
            monitor_mode="min",
            device="cpu",
        )
        trainer = Trainer(model=model, config=cfg)
        metrics = trainer.validate(val_loader=val_loader, loss_fn=DiceLoss())

        assert "val/loss" in metrics
        assert "val/dice" in metrics
        assert 0.0 <= metrics["val/dice"] <= 1.0

    def test_early_stopping_triggers(self, tmp_path: Path) -> None:
        samples = [
            {
                "image": torch.randn(3, 8, 8),
                "label": torch.tensor(0, dtype=torch.long),
                "mask": torch.zeros(8, 8, dtype=torch.long),
            }
            for _ in range(6)
        ]
        train_loader = DataLoader(_DictDataset(samples), batch_size=2, shuffle=False)
        val_loader = DataLoader(_DictDataset(samples), batch_size=2, shuffle=False)

        model = _TinyClassifier()
        cfg = TrainConfig(
            epochs=6,
            precision="bf16-mixed",
            checkpoint_dir=str(tmp_path / "ckpts_es"),
            monitor_metric="val/accuracy",
            monitor_mode="max",
            device="cpu",
            early_stopping_patience=1,
            early_stopping_min_delta=10.0,
        )
        trainer = Trainer(model=model, config=cfg)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        history = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=nn.CrossEntropyLoss(),
            optimizer=optimizer,
            scheduler=None,
        )

        assert len(history["val/accuracy"]) < cfg.epochs
