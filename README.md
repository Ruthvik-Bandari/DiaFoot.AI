# DiaFoot.AI v2

> Production-grade Diabetic Foot Ulcer (DFU) Detection, Segmentation & Wagner Staging

[![CI](https://github.com/Ruthvik-Bandari/DiaFoot.AI/actions/workflows/ci.yml/badge.svg)](https://github.com/Ruthvik-Bandari/DiaFoot.AI/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://python.org)
[![PyTorch 2.10](https://img.shields.io/badge/PyTorch-2.10-ee4c2c.svg)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## What's New in v2

DiaFoot.AI v2 is a complete rebuild addressing critical issues from v1:

| Issue | v1 | v2 |
|-------|----|----|
| Data quality | Raw, unaudited | CleanVision + Cleanlab pipeline |
| Training data | Ulcer-only (zero specificity) | Healthy + non-DFU + DFU (Wagner 0-5) |
| Architecture | Single-task binary segmentation | Multi-task: Classify → Segment → Stage |
| Fairness | Not analyzed | ITA skin-tone stratified evaluation |
| Evaluation | Dice + IoU only | HD95, NSD, AUROC, uncertainty, calibration |

## Multi-Task Pipeline

```
Input Image → Triage Classifier → [Healthy | Non-DFU | DFU]
                                          ↓ (if DFU)
                                   Segmentation → Wound Mask + Uncertainty
                                          ↓
                                   Wagner Staging → Grade 0-5 + Confidence
```

## Tech Stack (Feb 2026)

| Component | Version | Role |
|-----------|---------|------|
| PyTorch | 2.10.0 | Core framework, BFloat16 native on H100 |
| MONAI | 1.5.2 | Medical transforms, losses, metrics |
| SMP | 0.5.0 | 12 architectures, 800+ encoders |
| FastAPI | 0.133.0 | Production REST API |
| Ruff | 0.15.2 | Linting + formatting (replaces black/flake8/isort) |

## Quick Start

```bash
git clone https://github.com/Ruthvik-Bandari/DiaFoot.AI.git
cd DiaFoot.AI && git checkout v2-rebuild

pip install uv
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"

pre-commit install
pytest tests/ -m "not gpu and not slow"
```

## HPC Training (H100)

```bash
sbatch slurm/train_single_gpu.sh configs/training/baseline.yaml
sbatch slurm/train_multi_gpu.sh configs/training/advanced.yaml
sbatch slurm/train_ablation.sh   # 12 ablation configs in parallel
```

## Project Structure

```
DiaFoot.AI-v2/
├── configs/          # YAML configs for data, models, training, deploy
├── src/
│   ├── data/         # Cleaning, preprocessing, augmentation, ITA analysis
│   ├── models/       # Classifier, U-Net++, nnU-Net v2, MedSAM2, FUSegNet
│   ├── training/     # Trainer, losses, schedulers, EMA, pseudo-labeling
│   ├── evaluation/   # Metrics, uncertainty, calibration, fairness
│   ├── inference/    # Pipeline, TTA, ONNX export, post-processing
│   └── deploy/       # FastAPI app, schemas, middleware
├── slurm/            # HPC job templates (H100)
├── docker/           # Training + inference containers
├── tests/            # Pytest suite
└── scripts/          # CLI entry points
```

## Development Phases

| Phase | Focus | Commits |
|-------|-------|---------|
| 1 | Data Foundation | 1-7 |
| 2 | Model Architecture | 8-12 |
| 3 | Training Pipeline | 13-18 |
| 4 | Evaluation & Clinical Metrics | 19-23 |
| 5 | Uncertainty & Fairness | 24-28 |
| 6 | Deployment & API | 29-34 |
| 7 | Documentation & Submission | 35-38 |

## Course

AAI6620 Computer Vision — Northeastern University

## Author

**Ruthvik Bandari** — [GitHub](https://github.com/Ruthvik-Bandari)

## License

[MIT](LICENSE)
