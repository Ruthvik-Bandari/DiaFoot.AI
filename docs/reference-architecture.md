# Architecture reference

This is a map of the DiaFoot.AI source tree. It describes every package under
`src/` and its key modules so you can find the right file fast. Each section has
a short intro and a table of modules, key classes or functions, and their
purpose. Stub modules (files that still carry a `# TODO: Implementation in
Phase N` marker) are marked clearly.

> Note on untrained components: some model code ships in the repo but has not
> been trained yet. The README lists these as MedSAM2 LoRA and nnU-Net v2. The
> deployed path is the DINOv2 classifier plus DINOv2 segmenter. For the trained
> models and their measured results, see [../README.md](../README.md).

The package name is `diafootai` (v2.0.0). All modules live under the `src`
import root, for example `from src.inference.pipeline import InferencePipeline`.

## Package overview

| Package | Role |
| --- | --- |
| `src.data` | Collect, clean, label, preprocess, split, and load the dataset. |
| `src.models` | Model architectures for classification, segmentation, and staging. |
| `src.training` | Trainers, loss functions, schedulers, and EMA. |
| `src.evaluation` | Metrics, calibration, uncertainty, robustness, fairness, and audits. |
| `src.inference` | End-to-end prediction pipeline, TTA, and ONNX export. |
| `src.deploy` | FastAPI service, request or response schemas, and middleware. |

## src.data

This package turns raw images into clean, split, and loadable training data. It
handles quality auditing, skin-tone (ITA) analysis for fairness, Wagner grade
labeling, standardization, stratified splitting, and leakage detection.

| Module | Key classes / functions | Purpose |
| --- | --- | --- |
| `cleaning.py` | `DataQualityAuditor`, `AuditConfig`, `AuditReport`, `audit_all_dfu_datasets()` | CleanVision-based automated quality audit over the DFU datasets. |
| `healthy_feet.py` | `organize_kaggle_dfu_normal()`, `organize_mendeley_normal()`, `validate_healthy_images()`, `create_empty_masks()`, `run_quality_audit_on_healthy()` | Collect and curate healthy foot images as the negative class. |
| `wagner_labeling.py` | `check_mask_quality()`, `audit_masks()`, `create_wagner_grade_csv()`, `run_label_audit()`, `MaskQualityResult` | Label quality checks plus Wagner grade annotation infrastructure. |
| `ita_analysis.py` | `compute_ita()`, `rgb_to_lab()`, `classify_ita()`, `analyze_dataset_ita()`, `run_ita_analysis()` | Compute the Individual Typology Angle per image for skin-tone fairness. |
| `preprocessing.py` | `resize_with_padding()`, `apply_clahe()`, `binarize_mask()`, `preprocess_image()`, `preprocess_mask()`, `preprocess_dataset()` | Standardize images and masks (CLAHE, padded resize, mask binarize). |
| `dataset.py` | `create_stratified_splits()`, `load_ita_scores()` | Create doubly-stratified train, val, and test splits by ITA plus class label. |
| `leakage_audit.py` | `audit_samples_for_leakage()`, `canonical_sample_id()`, `load_split_csv()`, `Sample` | Detect train/val/test leakage: path overlap, SHA256 content, canonical filename ID, and dHash near-duplicates. |
| `augmentation.py` | `get_train_transforms()`, `get_val_transforms()` | Mask-aware Albumentations pipelines (default image size 518). |
| `torch_dataset.py` | `DFUDataset` | PyTorch `Dataset` that loads samples from the split CSVs. |
| `synthetic.py` | (none) | STUB, `# TODO: Implementation in Phase 1`. |

## src.models

This package holds every model architecture. The deployed path uses the DINOv2
classifier and DINOv2 segmenter. The other architectures are alternatives or
baselines. LoRA fine-tuning support is built into the DINOv2 models. The nnU-Net
v2 wrapper prepares data and config but its trained weights are not included
(see the README note above).

| Module | Key classes / functions | Purpose |
| --- | --- | --- |
| `dinov2_classifier.py` | `DINOv2Classifier` (`forward()`, `predict_with_confidence()`, `_apply_lora()`) | DINOv2 ViT backbone (frozen or LoRA) plus an MLP head for 3-class triage. |
| `dinov2_segmenter.py` | `DINOv2Segmenter`, `UPerNetDecoder`, `PPM`, `ConvBNReLU`, `build_dinov2_segmenter()` | DINOv2 encoder with multi-scale patch tokens plus a UPerNet-style decoder, optional LoRA. |
| `unetpp.py` | `build_unetpp()` | Baseline single-task U-Net++ segmenter (segmentation-models-pytorch). |
| `unetpp_multitask.py` | `MultiTaskUNetPP` | Shared encoder with three task heads (segmentation, classification, staging). |
| `fusegnet.py` | `FUSegNet` | EfficientNet-B7 encoder plus P-scSE attention segmenter. |
| `nnunet_wrapper.py` | `NNUNetConfig`, `generate_dataset_json()`, `prepare_nnunet_directory()` | Self-configuring nnU-Net v2 setup for multi-class segmentation (implemented, not trained). |
| `classifier.py` | `TriageClassifier` (`predict_with_confidence()`) | timm-based EfficientNet gateway classifier for the multi-task pipeline. |
| `staging_head.py` | `WagnerStagingHead` | Classifies wound severity into Wagner grades 0 to 5. |
| `attention.py` | `ParallelScSE`, `ChannelSE`, `SpatialSE` | P-scSE (Parallel Spatial-Channel Squeeze and Excitation) attention blocks. |
| `boundary_refine.py` | `morphological_smooth()`, `connected_component_filter()`, `refine_prediction()` | Morphological and connected-component cleanup of predicted masks. |

## src.training

This package trains the models. It provides a single-task trainer and a
multi-task trainer, plus compound loss functions, a warmup scheduler, and EMA.
Three modules are stubs reserved for later phases.

| Module | Key classes / functions | Purpose |
| --- | --- | --- |
| `multitask_trainer.py` | `MultiTaskTrainer` (`compute_multitask_loss()`, `train_epoch()`, `fit()`), `MultiTaskConfig` | Joint training for classification plus segmentation plus staging. |
| `trainer.py` | `Trainer` (`train_epoch()`, `validate()`, `save_checkpoint()`, `fit()`), `TrainConfig` | Single-task trainer with BFloat16, gradient clipping, checkpointing, and early stopping. |
| `losses.py` | `DiceLoss`, `DiceCELoss`, `FocalTverskyLoss`, `DiceBoundaryLoss` | Compound loss functions for wound segmentation. |
| `classification_losses.py` | `FocalLoss`, `LabelSmoothingCE` | Losses for the triage classifier and Wagner staging head. |
| `schedulers.py` | `CosineAnnealingWithWarmup` | Cosine annealing learning rate with a linear warmup phase. |
| `ema.py` | `EMA` (`update()`, `apply_shadow()`, `get_shadow_model()`) | Exponential moving average of weights for training stabilization. |
| `callbacks.py` | (none) | STUB, `# TODO: Implementation in Phase 3`. |
| `distributed.py` | (none) | STUB, `# TODO: Implementation in Phase 3`. |
| `pseudo_label.py` | (none) | STUB, `# TODO: Implementation in Phase 3`. |

## src.evaluation

This package scores model outputs against labels and runs trustworthiness
audits. It covers segmentation and classification metrics, calibration,
uncertainty, robustness to degradations, inter-annotator agreement, skin-tone
fairness, and shortcut-learning checks. For the actual measured numbers, see
[../README.md](../README.md).

| Module | Key classes / functions | Purpose |
| --- | --- | --- |
| `metrics.py` | `dice_score()`, `iou_score()`, `hausdorff_distance_95()`, `surface_dice()`, `wound_area_mm2()`, `compute_segmentation_metrics()`, `aggregate_metrics()` | Segmentation metrics: Dice, IoU, HD95, surface Dice or NSD, and clinical wound area. |
| `classification_metrics.py` | `compute_classification_metrics()`, `print_classification_report()` | Comprehensive triage classification evaluation. |
| `calibration.py` | `expected_calibration_error()`, `multiclass_brier_score()`, `temperature_scaling()`, `tune_defer_threshold()`, `compute_calibration_report()` | ECE, reliability diagrams, temperature scaling, and defer-threshold tuning. |
| `uncertainty.py` | `enable_mc_dropout()`, `mc_dropout_predict()`, `ensemble_predict()`, `conformal_wound_area()`, `compute_uncertainty_metrics()` | Pixel-wise uncertainty via MC Dropout and ensemble variance, plus conformal area. |
| `robustness.py` | `apply_gaussian_blur()`, `apply_gaussian_noise()`, `apply_brightness_shift()`, `apply_contrast_reduction()`, `apply_jpeg_compression()`, `run_robustness_test()` | Test model performance under synthetic image degradations. |
| `annotator_agreement.py` | `compute_pairwise_dice()`, `compute_majority_vote()`, `staple_consensus()`, `model_vs_human_ceiling()`, `fleiss_kappa()` | Compare model predictions against human annotations and estimate a human ceiling. |
| `area_agreement.py` | `compute_area_agreement()` | Agreement between predicted and manual wound areas. |
| `external_validation.py` | `bootstrap_ci()`, `compute_drop_report()` | External validation helpers: bootstrap confidence intervals and performance drop. |
| `failure_atlas.py` | `classify_segmentation_failure()`, `summarize_failure_types()` | Categorize and summarize segmentation failure modes. |
| `shortcut_audit.py` | `perturb_border_noise()`, `keep_center_only()`, `blur_background()`, `summarize_shortcut_shift()` | Audit the classifier for shortcut learning on backgrounds and borders. |
| `fairness.py` | `load_ita_mapping()`, `stratified_classification_audit()`, `stratified_segmentation_audit()`, `run_fairness_audit()` | Evaluate performance across skin-tone (ITA) groups. |
| `subgroup_analysis.py` | `classification_subgroup_report()`, `segmentation_subgroup_report()` | Subgroup reports with bootstrap confidence intervals. |
| `clinical_metrics.py` | (none) | STUB, `# TODO: Implementation in Phase 4-5`. |

## src.inference

This package runs the trained models end to end. The pipeline classifies an
image, and if the image is a DFU it segments the wound and post-processes the
mask. TTA and ONNX export sit alongside for accuracy and deployment.

| Module | Key classes / functions | Purpose |
| --- | --- | --- |
| `pipeline.py` | `InferencePipeline` (`__init__()`, `preprocess()`, `predict()`, `predict_batch()`), `PipelineResult` | End-to-end flow: image, preprocess, classify (DINOv2), then if DFU segment (DINOv2), then post-process. |
| `tta.py` | `tta_predict_segmentation()`, `tta_predict_classification()`, `compute_tta_improvement()`, `_d4_transforms()` | D4 test-time augmentation for better predictions and an uncertainty proxy. |
| `onnx_export.py` | `export_to_onnx()`, `validate_onnx()` | Export trained DINOv2 or legacy models to ONNX and check parity. |
| `postprocess.py` | (none) | STUB, `# TODO: Implementation in Phase 6`. |
| `predictor.py` | (none) | STUB, `# TODO: Implementation in Phase 6`. |

## src.deploy

This package serves the pipeline over HTTP with FastAPI. It exposes the predict,
health, and model info endpoints, validates and rate-limits uploads, gates on
image quality, and defers low-quality or low-confidence cases to a clinician.

| Module | Key classes / functions | Purpose |
| --- | --- | --- |
| `app.py` | `health_check()`, `model_info()`, `assess_image_quality()`, `resolve_runtime_thresholds()`, `_build_pipeline_from_checkpoints()`, `PredictionResponse`, `HealthResponse`, `ModelInfoResponse` | FastAPI application. Endpoints: `POST /predict`, `GET /health`, `GET /model/info`. |
| `schemas.py` | `DriftFeatures`, `PredictionLogEvent` | Pydantic schemas for monitoring and drift hooks. |
| `middleware.py` | `MaxContentLengthMiddleware`, `RateLimitMiddleware`, `is_allowed_content_type()`, `validate_upload_metadata()` | Request payload size limiting, in-memory per-client rate limiting, and content-type validation. |

## How the packages depend on each other

Data flows left to right and top to bottom. `src.data` prepares batches,
`src.models` and `src.training` produce trained checkpoints, `src.inference`
runs those checkpoints, and `src.deploy` serves the pipeline. `src.evaluation`
sits to the side and consumes model or pipeline outputs against labels.

```
   +-------------------+
   |     src.data      |  cleaning, preprocessing, ITA,
   |                   |  stratified splits, leakage audit
   +---------+---------+
             |  torch_dataset feeds batches
             v
   +-------------------+        +-------------------+
   |    src.models     | <----> |   src.training    |
   | DINOv2, U-Net++,  | builds | trainers, losses, |
   | FUSegNet, nnU-Net |        | schedulers, EMA   |
   +---------+---------+        +-------------------+
             |  trained weights (checkpoints)
             v
   +-------------------+
   |   src.inference   |  InferencePipeline, TTA, ONNX export
   +---------+---------+
             |  pipeline result
             v
   +-------------------+
   |    src.deploy     |  FastAPI app, schemas, middleware
   +-------------------+

   +---------------------------------------------------+
   |                  src.evaluation                   |
   | metrics, calibration, uncertainty, robustness,    |
   | fairness, agreement, failure atlas, shortcuts     |
   +---------------------------------------------------+
        ^  consumes model and pipeline outputs
        |  (predictions compared against labels)
```

## Related

- [explanation-pipeline-design.md](explanation-pipeline-design.md) - why the pipeline is designed this way.
- [reference-cli.md](reference-cli.md) - the command line scripts that drive these packages.
- [../README.md](../README.md) - project overview, trained models, and measured results.
