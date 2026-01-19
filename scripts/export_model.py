#!/usr/bin/env python3
"""
DiaFootAI Model Export
======================

Export trained models to various formats for deployment:
1. ONNX - Universal format (web, mobile, edge devices)
2. TorchScript - PyTorch production deployment
3. CoreML - iOS/macOS native apps

Author: Ruthvik
Date: January 2025
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import argparse
import json
import segmentation_models_pytorch as smp


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model(
    checkpoint_path: str,
    encoder: str = 'efficientnet-b4',
    architecture: str = 'unetplusplus'
) -> nn.Module:
    """Load trained model from checkpoint."""
    
    if architecture == 'unetplusplus':
        model = smp.UnetPlusPlus(
            encoder_name=encoder,
            encoder_weights=None,
            in_channels=3,
            classes=1
        )
    elif architecture == 'unet':
        model = smp.Unet(
            encoder_name=encoder,
            encoder_weights=None,
            in_channels=3,
            classes=1
        )
    elif architecture == 'deeplabv3plus':
        model = smp.DeepLabV3Plus(
            encoder_name=encoder,
            encoder_weights=None,
            in_channels=3,
            classes=1
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # Handle 'model.' prefix
    if any(k.startswith('model.') for k in state_dict.keys()):
        state_dict = {k.replace('model.', '', 1): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model.eval()
    
    return model, checkpoint


# =============================================================================
# WRAPPER FOR EXPORT (includes sigmoid)
# =============================================================================

class SegmentationModelForExport(nn.Module):
    """
    Wrapper that includes sigmoid activation for easier deployment.
    Output is probability map in range [0, 1].
    """
    
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x)
        return torch.sigmoid(logits)


# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================

def export_onnx(
    model: nn.Module,
    output_path: Path,
    image_size: int = 512,
    opset_version: int = 14,
    dynamic_batch: bool = True
) -> None:
    """Export model to ONNX format."""
    
    print(f"\n📦 Exporting to ONNX...")
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, image_size, image_size)
    
    # Dynamic axes for flexible batch size
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    
    # Export
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes
    )
    
    print(f"   ✅ Saved: {output_path}")
    print(f"   Input shape: (batch, 3, {image_size}, {image_size})")
    print(f"   Output shape: (batch, 1, {image_size}, {image_size})")
    print(f"   Output range: [0, 1] (probability)")
    
    # Verify
    try:
        import onnx
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        print(f"   ✅ ONNX model verified!")
    except ImportError:
        print(f"   ⚠️  Install 'onnx' package to verify model")
    except Exception as e:
        print(f"   ⚠️  Verification warning: {e}")


def export_torchscript(
    model: nn.Module,
    output_path: Path,
    image_size: int = 512,
    method: str = 'trace'
) -> None:
    """Export model to TorchScript format."""
    
    print(f"\n📦 Exporting to TorchScript ({method})...")
    
    model.eval()
    
    if method == 'trace':
        # Tracing (recommended for most cases)
        dummy_input = torch.randn(1, 3, image_size, image_size)
        traced_model = torch.jit.trace(model, dummy_input)
    else:
        # Scripting (for models with control flow)
        traced_model = torch.jit.script(model)
    
    # Optimize
    traced_model = torch.jit.optimize_for_inference(traced_model)
    
    # Save
    traced_model.save(str(output_path))
    
    print(f"   ✅ Saved: {output_path}")
    print(f"   Input shape: (batch, 3, {image_size}, {image_size})")
    
    # Verify
    loaded = torch.jit.load(str(output_path))
    test_input = torch.randn(1, 3, image_size, image_size)
    with torch.no_grad():
        output = loaded(test_input)
    print(f"   ✅ Verified! Output shape: {tuple(output.shape)}")


def export_coreml(
    model: nn.Module,
    output_path: Path,
    image_size: int = 512
) -> None:
    """Export model to CoreML format for iOS/macOS."""
    
    print(f"\n📦 Exporting to CoreML...")
    
    try:
        import coremltools as ct
    except ImportError:
        print("   ❌ CoreML Tools not installed. Install with: pip install coremltools")
        return
    
    model.eval()
    
    # Trace model first
    dummy_input = torch.randn(1, 3, image_size, image_size)
    traced_model = torch.jit.trace(model, dummy_input)
    
    # Convert to CoreML
    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.ImageType(
            name="input",
            shape=(1, 3, image_size, image_size),
            scale=1/255.0,
            bias=[-0.485/0.229, -0.456/0.224, -0.406/0.225]  # Normalization
        )],
        outputs=[ct.TensorType(name="output")],
        minimum_deployment_target=ct.target.iOS15
    )
    
    # Add metadata
    mlmodel.author = "DiaFootAI"
    mlmodel.short_description = "Diabetic Foot Ulcer Segmentation Model"
    mlmodel.version = "1.0.0"
    
    # Save
    mlmodel.save(str(output_path))
    
    print(f"   ✅ Saved: {output_path}")
    print(f"   Target: iOS 15+, macOS 12+")


# =============================================================================
# MAIN EXPORT FUNCTION
# =============================================================================

def export_model(
    checkpoint_path: str,
    encoder: str,
    architecture: str,
    output_dir: str,
    image_size: int = 512,
    formats: list = ['onnx', 'torchscript']
) -> dict:
    """Export model to specified formats."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("DiaFootAI Model Export")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Encoder: {encoder}")
    print(f"Architecture: {architecture}")
    print(f"Image Size: {image_size}")
    print(f"Formats: {formats}")
    print("=" * 60)
    
    # Load model
    print("\n📦 Loading model...")
    model, checkpoint_data = load_model(checkpoint_path, encoder, architecture)
    
    # Wrap with sigmoid for easier deployment
    export_model = SegmentationModelForExport(model)
    export_model.eval()
    
    # Get model info
    val_iou = checkpoint_data.get('val_iou', 'N/A')
    val_dice = checkpoint_data.get('val_dice', 'N/A')
    epoch = checkpoint_data.get('epoch', 'N/A')
    
    if isinstance(val_iou, float):
        print(f"   Validation IoU: {val_iou:.4f}")
        print(f"   Validation Dice: {val_dice:.4f}")
    print(f"   Trained epochs: {epoch}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {total_params / 1e6:.1f}M")
    
    exported_paths = {}
    
    # Export to each format
    if 'onnx' in formats:
        onnx_path = output_dir / "model.onnx"
        export_onnx(export_model, onnx_path, image_size)
        exported_paths['onnx'] = str(onnx_path)
    
    if 'torchscript' in formats:
        ts_path = output_dir / "model.pt"
        export_torchscript(export_model, ts_path, image_size)
        exported_paths['torchscript'] = str(ts_path)
    
    if 'coreml' in formats:
        coreml_path = output_dir / "model.mlmodel"
        export_coreml(export_model, coreml_path, image_size)
        exported_paths['coreml'] = str(coreml_path)
    
    # Save metadata
    metadata = {
        'encoder': encoder,
        'architecture': architecture,
        'image_size': image_size,
        'parameters': total_params,
        'val_iou': float(val_iou) if isinstance(val_iou, float) else None,
        'val_dice': float(val_dice) if isinstance(val_dice, float) else None,
        'exported_formats': list(exported_paths.keys()),
        'exported_paths': exported_paths,
        'preprocessing': {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'input_range': [0, 1],  # After normalization
        },
        'postprocessing': {
            'output_range': [0, 1],  # Probability
            'threshold': 0.5,  # For binary mask
        }
    }
    
    metadata_path = output_dir / "model_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n📋 Metadata saved: {metadata_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("EXPORT COMPLETE")
    print("=" * 60)
    print(f"📁 Output directory: {output_dir}")
    for fmt, path in exported_paths.items():
        size_mb = Path(path).stat().st_size / (1024 * 1024)
        print(f"   {fmt}: {Path(path).name} ({size_mb:.1f} MB)")
    print("=" * 60)
    
    return exported_paths


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Export DiaFootAI models for deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export to ONNX and TorchScript
  python export_model.py --checkpoint outputs/best_model.pt --encoder efficientnet-b5

  # Export to all formats including CoreML
  python export_model.py --checkpoint outputs/best_model.pt --encoder efficientnet-b5 --formats onnx torchscript coreml

  # Export with custom image size
  python export_model.py --checkpoint outputs/best_model.pt --encoder efficientnet-b4 --image-size 384
        """
    )
    
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--encoder', default='efficientnet-b4', help='Encoder name')
    parser.add_argument('--architecture', default='unetplusplus', help='Architecture name')
    parser.add_argument('--output-dir', default='outputs/exported', help='Output directory')
    parser.add_argument('--image-size', type=int, default=512, help='Input image size')
    parser.add_argument('--formats', nargs='+', default=['onnx', 'torchscript'],
                        choices=['onnx', 'torchscript', 'coreml'],
                        help='Export formats')
    
    args = parser.parse_args()
    
    export_model(
        checkpoint_path=args.checkpoint,
        encoder=args.encoder,
        architecture=args.architecture,
        output_dir=args.output_dir,
        image_size=args.image_size,
        formats=args.formats
    )


if __name__ == "__main__":
    main()
