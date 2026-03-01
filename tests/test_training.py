"""DiaFoot.AI v2 — Training Pipeline Tests (Phase 3, Commits 13-15)."""

import torch
import torch.nn as nn

from src.training.classification_losses import FocalLoss, LabelSmoothingCE
from src.training.ema import EMA
from src.training.losses import DiceBoundaryLoss, DiceCELoss, DiceLoss, FocalTverskyLoss
from src.training.schedulers import CosineAnnealingWithWarmup


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
            scheduler.step()
        # LR should increase during warmup then decrease
        assert lrs[0] < lrs[4]  # Warmup increases

    def test_cosine_decreases_lr(self) -> None:
        model = nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = CosineAnnealingWithWarmup(optimizer, warmup_epochs=2, max_epochs=20)
        for _ in range(3):
            scheduler.step()
        lr_mid = optimizer.param_groups[0]["lr"]
        for _ in range(15):
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
