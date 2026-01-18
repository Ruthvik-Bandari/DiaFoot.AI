# DiaFootAI: Complete Training Guide

## 🎯 Project Overview

DiaFootAI is a production-grade deep learning system for diabetic foot ulcer (DFU) detection and segmentation. This guide walks you through every step from setup to deployment.

## 📋 Prerequisites

- Python 3.10+
- CUDA 11.8+ (for GPU training) or Apple Silicon Mac
- 16GB RAM minimum (32GB recommended)
- 50GB free disk space for datasets

## 🚀 Step 1: Environment Setup

### Option A: Using Conda (Recommended)
```bash
# Create environment
conda create -n diafootai python=3.11 -y
conda activate diafootai

# Install PyTorch (CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# OR for Apple Silicon
pip install torch torchvision torchaudio

# Install project dependencies
cd DiaFootAI
pip install -r requirements.txt
```

### Option B: Using venv
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

pip install torch torchvision torchaudio
pip install -r requirements.txt
```

### Verify Installation
```bash
python scripts/verify_setup.py
```

## 📊 Step 2: Download Datasets

### Automatic Download (Open-Access Datasets)
```bash
# Download all open-access datasets
python scripts/download_datasets.py --all

# Or download specific datasets
python scripts/download_datasets.py --dataset fuseg
python scripts/download_datasets.py --dataset azh_segmentation
python scripts/download_datasets.py --dataset wound_tissue
```

### Manual Downloads Required

#### 1. DFUC Challenge Dataset (HIGHEST QUALITY - RECOMMENDED)
This is the gold standard dataset with 15,683 annotated images.

1. Visit: https://dfu-challenge.github.io/
2. Click "Apply for Datasets": http://www2.docm.mmu.ac.uk/STAFF/M.Yap/dataset.php
3. Fill the license agreement form
4. You'll receive download links via email (1-2 business days)
5. Extract to: `data/raw/dfuc/`

#### 2. Kaggle DFU Dataset
```bash
# Install Kaggle CLI
pip install kaggle

# Setup API key (download from kaggle.com/account)
mkdir -p ~/.kaggle
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Download
kaggle datasets download -d laithjj/diabetic-foot-ulcer-dfu
unzip diabetic-foot-ulcer-dfu.zip -d data/raw/kaggle_dfu/
```

### Dataset Structure After Download
```
data/
├── raw/
│   ├── fuseg/
│   │   └── wound-segmentation/
│   │       └── data/
│   │           └── Foot Ulcer Segmentation Challenge/
│   │               ├── train/
│   │               │   ├── images/
│   │               │   └── labels/
│   │               └── validation/
│   │                   ├── images/
│   │                   └── labels/
│   ├── azh_segmentation/
│   ├── dfuc/  (if you got access)
│   └── kaggle_dfu/
└── processed/
```

## 🔧 Step 3: Configuration

Edit `configs/config.yaml` to customize training:

```yaml
# Key settings to modify:

# Dataset
dataset:
  name: "fuseg"  # Change based on your dataset
  image:
    input_size: 512  # Increase for better accuracy (at cost of memory)

# Model
models:
  segmentation:
    architecture: "unetplusplus"  # Options: unet, unetplusplus, deeplabv3plus
    encoder: "efficientnet-b4"    # Options: efficientnet-b0 to b7, resnet50/101

# Training
training:
  batch_size: 8           # Reduce if OOM, increase if you have more VRAM
  epochs: 150
  device: "cuda"          # "cuda", "mps" (Mac), or "cpu"
  
  mixed_precision:
    enabled: true         # Faster training, less memory
  
  early_stopping:
    patience: 20          # Stop if no improvement for 20 epochs
```

## 🏃 Step 4: Training

### Basic Training
```bash
python scripts/train.py --config configs/config.yaml
```

### Training with Options
```bash
# Custom experiment name
python scripts/train.py --config configs/config.yaml --experiment-name my_first_run

# Override epochs and batch size
python scripts/train.py --config configs/config.yaml --epochs 100 --batch-size 4

# Enable Weights & Biases logging
python scripts/train.py --config configs/config.yaml --wandb

# Resume from checkpoint
python scripts/train.py --config configs/config.yaml --resume models/checkpoints/last_model.pt

# Disable mixed precision (if issues on some GPUs)
python scripts/train.py --config configs/config.yaml --no-amp
```

### Expected Training Output
```
============================================================
Starting Training
============================================================
Device: cuda
Epochs: 150
Batch size: 8
Accumulation steps: 2
Effective batch size: 16
Mixed precision: True
EMA: True
============================================================

Epoch 1/150 | train_loss: 0.6234 | val_loss: 0.5123 | val_iou: 0.4521 | val_dice: 0.6234
  EarlyStopping: val_iou improved to 0.452100
  Checkpoint: Saved best model (val_iou=0.4521)
  Learning rate: 1.00e-04
...
```

## 📈 Step 5: Monitor Training

### Using TensorBoard
```bash
tensorboard --logdir outputs/
```
Then open http://localhost:6006

### Using Weights & Biases
```bash
# Login first
wandb login

# Train with logging
python scripts/train.py --config configs/config.yaml --wandb
```

## 🎯 Step 6: Evaluation

### Evaluate Best Model
```bash
python scripts/evaluate.py \
    --model models/checkpoints/best_model.pt \
    --data data/raw/fuseg/wound-segmentation/data/Foot\ Ulcer\ Segmentation\ Challenge \
    --split test
```

### Target Metrics
| Metric | Target | Excellent |
|--------|--------|-----------|
| IoU | > 0.80 | > 0.85 |
| Dice | > 0.85 | > 0.90 |
| Precision | > 0.85 | > 0.90 |
| Recall (Sensitivity) | > 0.85 | > 0.90 |
| Hausdorff Distance | < 15 | < 10 |

## 🔍 Step 7: Inference on New Images

```python
import torch
from PIL import Image
from src.models.segmentation import SegmentationModel
from src.data.preprocessing import preprocess_image
import numpy as np

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SegmentationModel(
    architecture="unetplusplus",
    encoder_name="efficientnet-b4",
    num_classes=1,
)
checkpoint = torch.load("models/checkpoints/best_model.pt", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

# Process image
image = Image.open("path/to/wound_image.jpg").convert("RGB")
image_np = np.array(image)
processed = preprocess_image(image_np, target_size=512)
input_tensor = torch.from_numpy(processed).permute(2, 0, 1).unsqueeze(0).to(device)

# Predict
with torch.no_grad():
    output = model(input_tensor)
    mask = torch.sigmoid(output).squeeze().cpu().numpy()
    binary_mask = (mask > 0.5).astype(np.uint8)

# Visualize
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(image)
axes[0].set_title("Original")
axes[1].imshow(mask, cmap="jet")
axes[1].set_title("Probability Map")
axes[2].imshow(image)
axes[2].imshow(binary_mask, alpha=0.5, cmap="Reds")
axes[2].set_title("Overlay")
plt.savefig("prediction.png")
```

## 📱 Step 8: Export for Mobile

### Export to ONNX
```python
import torch

model.eval()
dummy_input = torch.randn(1, 3, 512, 512).to(device)

torch.onnx.export(
    model,
    dummy_input,
    "models/exported/wound_segmentation.onnx",
    opset_version=12,
    input_names=["image"],
    output_names=["mask"],
    dynamic_axes={
        "image": {0: "batch_size"},
        "mask": {0: "batch_size"},
    },
)
```

### Export to TFLite (for Android)
```python
# Install tensorflow
pip install tensorflow

# Convert ONNX to TFLite
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf

onnx_model = onnx.load("models/exported/wound_segmentation.onnx")
tf_rep = prepare(onnx_model)
tf_rep.export_graph("models/exported/tf_model")

converter = tf.lite.TFLiteConverter.from_saved_model("models/exported/tf_model")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open("models/exported/wound_segmentation.tflite", "wb") as f:
    f.write(tflite_model)
```

### Export to CoreML (for iOS)
```python
import coremltools as ct

model_ct = ct.convert(
    model,
    inputs=[ct.ImageType(shape=(1, 3, 512, 512))],
)
model_ct.save("models/exported/wound_segmentation.mlmodel")
```

## 🚨 Troubleshooting

### Out of Memory (OOM)
```bash
# Reduce batch size
python scripts/train.py --batch-size 4

# Or reduce image size in config.yaml
# image.input_size: 384 instead of 512
```

### Training Not Converging
1. Check learning rate (try 1e-4 to 3e-4)
2. Verify data augmentation isn't too aggressive
3. Ensure masks are properly binarized (0 or 1)
4. Try different loss function (Focal Tversky for imbalanced data)

### Low Validation Metrics
1. Add more data augmentation for skin tone diversity
2. Increase model capacity (efficientnet-b5 or b6)
3. Use longer training with smaller learning rate
4. Enable Test Time Augmentation (TTA) for inference

## 📚 Model Architecture Comparison

| Model | Params | IoU | Speed | Use Case |
|-------|--------|-----|-------|----------|
| UNet + MobileNetV2 | 3.5M | 0.78 | Fast | Mobile deployment |
| UNet++ + EfficientNet-B4 | 25M | 0.86 | Medium | **Recommended** |
| DeepLabV3+ + ResNet101 | 60M | 0.87 | Slow | Maximum accuracy |

## 🎓 Academic Citations

If using this project, please cite:

```bibtex
@article{dfu_segmentation_2025,
  title={DiaFootAI: Deep Learning for Diabetic Foot Ulcer Detection},
  author={Your Name},
  year={2025}
}
```

## 📞 Support

- Issues: GitHub Issues
- Questions: ruthvik@example.com
- Documentation: docs/

---

**Happy Training! 🚀**
