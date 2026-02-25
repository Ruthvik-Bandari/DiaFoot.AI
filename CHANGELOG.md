# Changelog

All notable changes to DiaFoot.AI will be documented in this file.

## [Unreleased] — v2.0.0

### Added
- Complete project scaffold with production-grade tooling
- Multi-task pipeline architecture (Classify → Segment → Stage)
- 7-phase development plan with 38 commits
- H100 HPC SLURM job templates
- CI/CD via GitHub Actions (lint, type-check, test)
- Pre-commit hooks (Ruff 0.15.2, mypy, nbstripout, torch.load safety)
- DVC data versioning setup
- Docker containers for training and inference

### Changed
- Full rebuild from v1 to address critical data and architecture issues

### v1.0 (Previous)
- U-Net++ / EfficientNet-B4 single-task binary segmentation
- 84.93% IoU, 91.73% Dice on FUSeg
- Known issues: raw data, ulcer-only training, zero real-world specificity
