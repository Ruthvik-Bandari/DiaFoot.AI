"""DiaFoot.AI v2 — Model Architecture Tests (Phase 2, Commit 9)."""

import torch

from src.models.attention import ChannelSE, ParallelScSE, SpatialSE
from src.models.staging_head import WagnerStagingHead
from src.models.unetpp import build_unetpp


class TestAttentionModules:
    """Test P-scSE attention components."""

    def test_channel_se_shape(self) -> None:
        x = torch.randn(2, 64, 32, 32)
        module = ChannelSE(64)
        out = module(x)
        assert out.shape == x.shape

    def test_spatial_se_shape(self) -> None:
        x = torch.randn(2, 64, 32, 32)
        module = SpatialSE(64)
        out = module(x)
        assert out.shape == x.shape

    def test_pscse_max_fusion(self) -> None:
        x = torch.randn(2, 64, 32, 32)
        module = ParallelScSE(64, use_max=True)
        out = module(x)
        assert out.shape == x.shape

    def test_pscse_additive_fusion(self) -> None:
        x = torch.randn(2, 64, 32, 32)
        module = ParallelScSE(64, use_max=False)
        out = module(x)
        assert out.shape == x.shape


class TestUNetPP:
    """Test U-Net++ segmentation model."""

    def test_output_shape(self) -> None:
        model = build_unetpp(
            encoder_name="resnet18",
            encoder_weights=None,
            classes=1,
            decoder_attention_type="scse",
        )
        x = torch.randn(1, 3, 512, 512)
        out = model(x)
        assert out.shape == (1, 1, 512, 512)

    def test_multiclass_output(self) -> None:
        model = build_unetpp(
            encoder_name="resnet18",
            encoder_weights=None,
            classes=4,
        )
        x = torch.randn(1, 3, 256, 256)
        out = model(x)
        assert out.shape == (1, 4, 256, 256)


class TestWagnerStagingHead:
    """Test Wagner grade staging head."""

    def test_output_shape(self) -> None:
        head = WagnerStagingHead(in_features=512, num_grades=6)
        x = torch.randn(4, 512)
        out = head(x)
        assert out.shape == (4, 6)

    def test_default_grades(self) -> None:
        head = WagnerStagingHead(in_features=256)
        x = torch.randn(2, 256)
        out = head(x)
        assert out.shape == (2, 6)
