# How to serve and deploy the API

This guide gets the DiaFoot.AI REST service running so you can send an image to
`POST /predict` and receive a triage classification, wound mask statistics, and
defer flags. By the end you will have a live service answering `/health`,
`/model/info`, and `/predict` on port 8000, either from your shell with uvicorn
or from a container.

The deployed service uses the DINOv2 path (classifier plus segmenter). For the
model results behind that path, see [../README.md](../README.md).

## Prerequisites

Before you start, make sure you have one of these ready:

- **The installed package.** Follow the Install section in
  [../README.md](../README.md) so `src.deploy.app` is importable and the
  runtime dependencies (FastAPI, uvicorn, PyTorch, OpenCV, Pillow) are present.
- **Trained checkpoints** for the PyTorch serving path: a classifier and,
  optionally, a segmenter. The app looks for them at these default paths:
  - `checkpoints/dinov2_classifier/best_epoch009_0.9785.pt`
  - `checkpoints/dinov2_segmenter/best_epoch009_0.1062.pt`

  The classifier is required. If its checkpoint is missing, the service still
  starts but reports `model_loaded: false` and every `/predict` call defers. The
  segmenter is optional. If it is missing, the service runs classification only.
- **Or an exported ONNX model** in `exports/` if you plan to build the container
  image, which ships that directory. See
  [Export and serve ONNX](#export-and-serve-onnx) below.

To train checkpoints first, see [howto-train.md](howto-train.md).

## Run locally with uvicorn

Start the service from the repository root:

```bash
uvicorn src.deploy.app:app --host 0.0.0.0 --port 8000
```

### How the app loads the model at startup

The FastAPI app uses a lifespan hook. When the process starts, it calls
`_build_pipeline_from_checkpoints()`, which:

1. Resolves the classifier and segmenter checkpoint paths from environment
   variables (or their defaults).
2. Builds a `DINOv2Classifier` (3 classes) and loads the classifier weights.
3. Builds a `DINOv2Segmenter` and loads the segmenter weights if that checkpoint
   exists.
4. Constructs an `InferencePipeline` and stores it in a module global.

If the classifier checkpoint is not found, or any load step raises, the pipeline
stays `None` and `/health` reports `model_loaded: false`.

### Environment variables

The running app reads its settings from `DIAFOOT_*` environment variables, not
from a config file. Set them before you launch the process. All names below are
verified against `src/deploy/app.py`.

| Variable | Default | Purpose |
|---|---|---|
| `DIAFOOT_BACKBONE` | `dinov2_vitb14` | DINOv2 backbone for classifier and segmenter |
| `DIAFOOT_CLASSIFIER_CKPT` (alias `DIAFOOT_CLASSIFIER_CHECKPOINT`) | `checkpoints/dinov2_classifier/best_epoch009_0.9785.pt` | Classifier weights |
| `DIAFOOT_SEGMENTER_CKPT` (alias `DIAFOOT_SEGMENTER_CHECKPOINT`) | `checkpoints/dinov2_segmenter/best_epoch009_0.1062.pt` | Segmenter weights (optional) |
| `DIAFOOT_DEVICE` | `cuda` | Inference device. Falls back to `cpu` if CUDA is unavailable |
| `DIAFOOT_CONFIDENCE_THRESHOLD` | `0.70` | Classification confidence threshold |
| `DIAFOOT_DEFER_THRESHOLD` | `0.50` | Defer-to-clinician threshold |
| `DIAFOOT_CALIBRATION_PATH` | `results/classification_calibration.json` | Calibration artifact used to source the defer threshold |
| `DIAFOOT_DFU_SEG_FALLBACK_PROB` | `0.10` | DFU segmentation fallback probability |
| `DIAFOOT_DFU_PROMOTION_THRESHOLD` | `0.04` | DFU promotion threshold |
| `DIAFOOT_MAX_IMAGE_SIZE_MB` | `20.0` | Upload size limit in megabytes |
| `DIAFOOT_RATE_LIMIT_RPM` | `100` | Requests per minute per client on `/predict` |
| `DIAFOOT_MIN_IMAGE_SIDE` | `256` | Minimum image side in pixels before flagging `low_resolution` |
| `DIAFOOT_BLUR_VARIANCE_THRESHOLD` | `10.0` | Laplacian variance below which an image is `blurry` |
| `DIAFOOT_BRIGHTNESS_MIN` | `20.0` | Mean brightness below which an image is `too_dark` |
| `DIAFOOT_BRIGHTNESS_MAX` | `235.0` | Mean brightness above which an image is `too_bright` |
| `DIAFOOT_CORS_ORIGINS` (alias `CORS_ORIGINS`) | `*` | Comma-separated allowed CORS origins |
| `DIAFOOT_PREDICTION_LOG` | empty (disabled) | Path to a JSONL file for structured prediction logging |

You can now run on CPU with explicit checkpoint paths like this:

```bash
DIAFOOT_DEVICE=cpu \
DIAFOOT_CLASSIFIER_CKPT=checkpoints/dinov2_classifier/best_epoch009_0.9785.pt \
DIAFOOT_SEGMENTER_CKPT=checkpoints/dinov2_segmenter/best_epoch009_0.1062.pt \
uvicorn src.deploy.app:app --host 0.0.0.0 --port 8000
```

Note that the confidence and defer thresholds resolve once, when the module is
imported. Set those variables before the process starts so they take effect.

## Configure

`configs/deploy/api.yaml` records the recommended deployment values in one place:

```yaml
api:
  host: 0.0.0.0
  port: 8000
  workers: 4
  cors_origins: ["*"]
  max_image_size_mb: 20
  rate_limit: 100             # requests per minute
  backbone: dinov2_vitb14
  classifier_checkpoint: checkpoints/dinov2_classifier/best_epoch009_0.9785.pt
  segmenter_checkpoint: checkpoints/dinov2_segmenter/best_epoch009_0.1062.pt
  confidence_threshold: 0.95
  defer_threshold: 0.67       # preferred from calibration when available
  calibration_report: results/classification_calibration.json
  min_image_side: 256
  blur_variance_threshold: 30.0
  brightness_min: 20.0
  brightness_max: 235.0
```

The app does not read this file directly. To apply these values to the running
service, set the matching `DIAFOOT_*` environment variables from the table
above. For example, to run with the recommended thresholds:

```bash
DIAFOOT_CONFIDENCE_THRESHOLD=0.95 \
DIAFOOT_DEFER_THRESHOLD=0.67 \
DIAFOOT_MAX_IMAGE_SIZE_MB=20 \
DIAFOOT_RATE_LIMIT_RPM=100 \
uvicorn src.deploy.app:app --host 0.0.0.0 --port 8000
```

### How confidence and defer thresholds are set

The confidence threshold comes straight from `DIAFOOT_CONFIDENCE_THRESHOLD`, or
`0.70` by default.

The defer threshold resolves in priority order:

1. `DIAFOOT_DEFER_THRESHOLD` if it is set. The source reports as `env`.
2. Otherwise the `recommended_threshold` under
   `classification.defer_tuning` in the calibration JSON at
   `DIAFOOT_CALIBRATION_PATH`. The source reports as
   `calibration:<path>`.
3. Otherwise the built-in default `0.50`. The source reports as `default`.

The resolved value and its source appear in `GET /model/info` as
`defer_threshold` and `defer_threshold_source`, so you can confirm which rule
won at runtime.

## Run with Docker

The `docker/` directory holds a slim inference image and a Compose file. Run
these from inside `docker/`:

```bash
cd docker
docker compose up --build
```

Compose builds `docker/Dockerfile.serve` with the repository root as the build
context, publishes port 8000, mounts `../exports` read-only into `/app/exports`,
restarts unless stopped, and health-checks the container with
`curl -f http://localhost:8000/health`.

The serve image is built on `python:3.12-slim` with ONNX Runtime and ships the
`exports/` directory, so it targets the ONNX serving path. The build copies
`exports/` into the image, so make sure that directory holds your exported
`.onnx` models before you build. See
[Export and serve ONNX](#export-and-serve-onnx).

To build the image directly instead of using Compose, run this from the
repository root so the build context is correct:

```bash
docker build -f docker/Dockerfile.serve -t diafoot-api .
```

Both paths run the same startup command as the local case:
`uvicorn src.deploy.app:app --host 0.0.0.0 --port 8000`.

## Export and serve ONNX

Export a trained checkpoint to ONNX, then validate that the ONNX output matches
PyTorch and benchmark its speed:

```bash
python scripts/export_onnx.py \
  --checkpoint checkpoints/dinov2_segmenter/best_epoch009_0.1062.pt \
  --model dinov2 \
  --output exports/segmenter.onnx \
  --validate --benchmark --verbose
```

Export the classifier the same way with the classifier model type:

```bash
python scripts/export_onnx.py \
  --checkpoint checkpoints/dinov2_classifier/best_epoch009_0.9785.pt \
  --model dinov2-classifier \
  --output exports/classifier.onnx \
  --validate --benchmark
```

`--validate` compares PyTorch and ONNX Runtime outputs over several random
inputs and passes when the maximum difference stays within tolerance.
`--benchmark` times ONNX Runtime inference and writes `onnx_benchmark.json` next
to the exported model. The export uses the DINOv2 native input size of 518x518.

`configs/deploy/onnx_export.yaml` documents the export defaults (opset 17, input
shape `[1, 3, 518, 518]`, validation tolerance `1.0e-5`) and lists the two model
targets `exports/classifier.onnx` and `exports/segmenter.onnx`.

The ONNX runtime path runs on the `onnxruntime` package. The export script's
`--validate` and `--benchmark` steps load the model with
`onnxruntime.InferenceSession`, and the `docker/Dockerfile.serve` image installs
`onnxruntime` and ships the exported models from `exports/`. For the measured
ONNX vs PyTorch parity and speedup numbers, see [../README.md](../README.md).

## Verification

With the service running on port 8000, check liveness:

```bash
curl http://localhost:8000/health
```

A healthy response with a loaded model looks like this:

```json
{"status": "healthy", "model_loaded": true, "version": "2.1.0"}
```

If `model_loaded` is `false`, the classifier checkpoint did not load. See
[Troubleshooting](#troubleshooting).

Inspect the model metadata, thresholds, and limits:

```bash
curl http://localhost:8000/model/info
```

This returns the classifier and segmenter names, `input_size` `[518, 518]`,
`num_classes` `3`, the resolved `confidence_threshold`, `defer_threshold`,
`defer_threshold_source`, `max_image_size_mb`, `rate_limit_rpm`, and `version`.

Send an image for a full prediction:

```bash
curl -F "file=@img.jpg" http://localhost:8000/predict
```

A successful prediction returns `classification` (one of `Healthy`, `Non-DFU`,
or `DFU`), `classification_confidence`, `classification_probs`,
`defer_to_clinician`, `defer_reason`, `quality_flags`, `has_wound`,
`wound_area_mm2`, `wound_coverage_pct`, and `inference_time_ms`. When a wound is
present, the response also includes a base64 PNG mask in
`segmentation_mask_base64`.

## Troubleshooting

- **`model_loaded: false` on `/health`.** The classifier checkpoint was not
  found or failed to load. Confirm the file exists at the path in
  `DIAFOOT_CLASSIFIER_CKPT` (default
  `checkpoints/dinov2_classifier/best_epoch009_0.9785.pt`), then restart. Until
  it loads, `/predict` returns `classification: "Model not loaded"` with
  `defer_reason: "model_not_loaded"`.

- **`413` payload too large.** The request `Content-Length` exceeds the limit,
  which returns `{"detail": "Request payload too large", ...}`. Send a smaller
  file or raise `DIAFOOT_MAX_IMAGE_SIZE_MB` (default 20) and restart. Note that
  a file within the size limit but with a bad content type returns
  `classification: "Manual Review Required"` with
  `defer_reason: "payload_too_large"` or `defer_reason: "invalid_content_type"`.

- **`429` rate limited.** A client exceeded `DIAFOOT_RATE_LIMIT_RPM` requests in
  the 60-second window on `/predict`. The response includes a `Retry-After`
  header and `{"detail": "Rate limit exceeded", "retry_after_seconds": N}`. Wait
  and retry, or raise `DIAFOOT_RATE_LIMIT_RPM` and restart.

- **Low-quality image deferral.** If the image fails the quality gate,
  `/predict` returns `classification: "Manual Review Required"`,
  `defer_to_clinician: true`, and `defer_reason: "low_image_quality"`. The
  `quality_flags` list names the reason: `low_resolution`, `blurry`, `too_dark`,
  or `too_bright`. Send a clearer image, or tune `DIAFOOT_MIN_IMAGE_SIDE`,
  `DIAFOOT_BLUR_VARIANCE_THRESHOLD`, `DIAFOOT_BRIGHTNESS_MIN`, and
  `DIAFOOT_BRIGHTNESS_MAX`.

- **Invalid image.** If the bytes cannot be decoded as an image, `/predict`
  returns `classification: "Error"` with `defer_reason: "invalid_image"`. Check
  that the file is a valid JPEG, PNG, or WebP and is not corrupted. Allowed
  content types are `image/jpeg`, `image/jpg`, `image/png`, and `image/webp`.

## Related

- [reference-api.md](reference-api.md) — full endpoint and schema reference.
- [explanation-pipeline-design.md](explanation-pipeline-design.md) — why the
  pipeline triages, defers, and gates the way it does.
- [../README.md](../README.md) — project overview, install, and results.
