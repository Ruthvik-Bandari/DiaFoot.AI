# Changelog

All notable changes to DiaFoot.AI are documented in this file.

## [2.0.0] — 2026-03-02

### Complete rebuild addressing v1 clinical limitations

#### Why Rebuild
- v1 trained only on ulcer images (zero specificity on healthy feet)
- v1 had no data cleaning pipeline (raw scraped data)
- v1 reported only Dice/IoU (no boundary or clinical metrics)
- v1 had no skin tone fairness analysis

#### Added
- **Multi-task pipeline**: Classify -> Segment -> Stage (three cascaded tasks)
- **Three data categories**: Healthy feet (3,300), Non-DFU conditions (2,686), DFU (1,010)
- **Data cleaning**: CleanVision quality audit + Cleanlab label audit
- **ITA skin tone analysis**: Objective skin tone measurement across all categories
- **Preprocessing pipeline**: CLAHE, resize to 512x512, mask binarization
- **Doubly-stratified splits**: By class AND ITA skin tone category
- **Triage classifier**: EfficientNet-V2-M, 3-class (Healthy/Non-DFU/DFU)
- **U-Net++ segmentation**: EfficientNet-B4 encoder with scSE attention
- **Multi-task U-Net++**: Shared encoder with classification + segmentation + staging heads
- **FUSegNet**: EfficientNet-B7 + P-scSE attention (top DFU architecture)
- **MedSAM2 LoRA**: Adapter-based fine-tuning with automatic bbox prompts
- **nnU-Net v2 wrapper**: Self-configuring segmentation
- **Compound losses**: Dice+CE, Dice+Boundary (warm-start), Focal Tversky
- **Focal Loss**: For classification with class imbalance
- **Cosine annealing + warmup**: LR scheduling
- **EMA**: Exponential moving average for stable evaluation
- **BFloat16 training**: Native on H200/H100 GPUs
- **Comprehensive metrics**: Dice, IoU, HD95, NSD, ASSD, wound area
- **Uncertainty quantification**: MC Dropout, ensemble, conformal prediction
- **Calibration analysis**: ECE, temperature scaling
- **Fairness audit**: ITA-stratified performance reporting
- **Robustness testing**: 5 degradation types x 5 severity levels
- **TTA**: Test-time augmentation with uncertainty proxy
- **Inter-annotator agreement**: STAPLE consensus, Fleiss' kappa
- **ONNX export**: Production inference format
- **Inference pipeline**: Full classify -> segment -> post-process
- **FastAPI REST service**: POST /predict endpoint
- **Boundary refinement**: Morphological smoothing + connected component filtering

#### Results
- Classification: 100% accuracy (3-class triage)
- Segmentation: 83.4% Dice, 76.9% IoU (overall), 81.2% Dice on DFU
- NSD@5mm: 92.0% on DFU (clinically accurate boundaries)
- Wound area estimation: Mean error < 60 mm²

## [1.0.0] — 2026-01-17

### Initial release
- U-Net++ with EfficientNet-B4 encoder
- Binary segmentation on FUSeg dataset only
- 84.93% IoU, 91.73% Dice
- No healthy foot or non-DFU training data
- No clinical specificity (predicts wounds on all inputs)
