# REST API reference

DiaFoot.AI serves its inference pipeline through a FastAPI application defined in
`src/deploy/app.py`. You upload a foot image and get back a triage classification (Healthy,
Non-DFU, or DFU), a wound segmentation summary, a defer-to-clinician decision, and image-quality
flags. The service loads the deployed DINOv2 classifier and DINOv2 + UPerNet segmenter at startup
and runs the same cascade documented in
[explanation-pipeline-design.md](explanation-pipeline-design.md).

This page documents every endpoint, request and response schema, guard, and configuration knob.
For end-to-end serving instructions (Docker, checkpoints, health checks) see
[howto-serve-api.md](howto-serve-api.md). For honest model results, see [../README.md](../README.md).

## Running the server

Start the API with uvicorn from the project root:

```bash
uvicorn src.deploy.app:app --host 0.0.0.0 --port 8000
```

The app builds the inference pipeline during startup (FastAPI `lifespan`). If the classifier
checkpoint is missing, the pipeline stays uninitialized and `/predict` responds with
`classification: "Model not loaded"` while `/health` reports `model_loaded: false`. Point the app
at your checkpoints with the environment variables in [Configuration](#configuration), or rely on
the defaults under `checkpoints/`.

Once the server is up, interactive docs are available at `http://localhost:8000/docs` (Swagger UI)
and `http://localhost:8000/redoc`.

## Endpoints

The API exposes three endpoints. Rate limiting and payload-size limits apply only to `/predict`.

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness and model-loaded status |
| GET | `/model/info` | Model metadata and active thresholds |
| POST | `/predict` | Run the full inference pipeline on an uploaded image |

### GET /health

Returns service liveness and whether the model pipeline loaded successfully. This endpoint never
runs inference and has no request body.

**Request:** no parameters, no body.

**Response** (`HealthResponse`):

| Field | Type | Description |
|---|---|---|
| `status` | string | Always `"healthy"` when the process is serving. |
| `model_loaded` | boolean | `true` if the inference pipeline initialized, `false` otherwise. |
| `version` | string | API version, currently `"2.1.0"`. |

**Example:**

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "2.1.0"
}
```

### GET /model/info

Returns metadata about the loaded models and the runtime thresholds and limits the service is
using. Use this to confirm which backbone, checkpoints, and defer threshold are active before you
send predictions. No request body.

**Request:** no parameters, no body.

**Response** (`ModelInfoResponse`):

| Field | Type | Description |
|---|---|---|
| `classifier` | string | Classifier description, for example `"DINOv2 (dinov2_vitb14) 3-class triage"`. |
| `segmenter` | string | Segmenter description, for example `"DINOv2 (dinov2_vitb14) + UPerNet"`, or `"Unavailable"` if the pipeline did not load. |
| `input_size` | array of integers | Model input resolution, `[518, 518]`. |
| `num_classes` | integer | Number of triage classes, `3` (Healthy, Non-DFU, DFU). |
| `confidence_threshold` | float | Minimum confidence used to trust a classification. |
| `defer_threshold` | float | Confidence below which a case is deferred to a clinician. |
| `defer_threshold_source` | string | Where the defer threshold came from: `"env"`, `"calibration:<path>"`, or `"default"`. |
| `max_image_size_mb` | float | Maximum accepted upload size in megabytes. |
| `rate_limit_rpm` | integer | Requests per minute allowed per client on `/predict`. |
| `version` | string | API version, currently `"2.1.0"`. |

**Example:**

```bash
curl http://localhost:8000/model/info
```

```json
{
  "classifier": "DINOv2 (dinov2_vitb14) 3-class triage",
  "segmenter": "DINOv2 (dinov2_vitb14) + UPerNet",
  "input_size": [518, 518],
  "num_classes": 3,
  "confidence_threshold": 0.95,
  "defer_threshold": 0.67,
  "defer_threshold_source": "calibration:results/classification_calibration.json",
  "max_image_size_mb": 20.0,
  "rate_limit_rpm": 100,
  "version": "2.1.0"
}
```

### POST /predict

Runs the full inference pipeline on an uploaded image: upload validation, image-quality gate,
triage classification, and (when a wound is plausible) segmentation and area measurement.

**Request:** `multipart/form-data` with a single file part.

| Part | Type | Required | Description |
|---|---|---|---|
| `file` | file upload | yes | Foot image. Allowed content types: `image/jpeg`, `image/jpg`, `image/png`, `image/webp`. |

`/predict` returns HTTP `200` with a `PredictionResponse` body in almost every case, including
validation failures. Guard failures are reported inside the body through `classification` and
`defer_reason` rather than through an HTTP error status. The middleware layer is the exception: it
can return `413` (payload too large), `400` (invalid `Content-Length`), or `429` (rate limit
exceeded) before the request reaches the endpoint. See [Guards and limits](#guards-and-limits).

**Response** (`PredictionResponse`):

| Field | Type | Description |
|---|---|---|
| `classification` | string | Triage label: `"Healthy"`, `"Non-DFU"`, `"DFU"`, `"Manual Review Required"`, `"Error"`, or `"Model not loaded"`. |
| `classification_confidence` | float | Confidence for the predicted class, `0.0` for guard failures. |
| `classification_probs` | object (string to float) | Per-class probabilities, for example `{"Healthy": ..., "Non-DFU": ..., "DFU": ...}`. Empty `{}` on guard failures. |
| `defer_to_clinician` | boolean | `true` when the case should be reviewed by a human. |
| `defer_reason` | string | Why the case was deferred, empty string when not deferred. See [defer reasons](#defer-to-clinician-behavior). |
| `quality_flags` | array of strings | Image-quality issues detected, for example `["blurry"]`. Empty when quality passed. |
| `has_wound` | boolean | `true` if the segmenter found a wound. |
| `wound_area_mm2` | float | Estimated wound area in square millimeters. |
| `wound_coverage_pct` | float | Wound pixels as a percentage of image pixels. |
| `inference_time_ms` | float | Server-side processing time for the request in milliseconds. |
| `segmentation_mask_base64` | string or null | Base64-encoded PNG of the binary wound mask, present only when a wound was found. |
| `diagnostics` | object or null | Optional pipeline diagnostics, `null` when empty. |

**Example:**

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@foot.jpg;type=image/jpeg"
```

A realistic DFU response (placeholder values, not benchmark results):

```json
{
  "classification": "DFU",
  "classification_confidence": 0.972,
  "classification_probs": {
    "Healthy": 0.011,
    "Non-DFU": 0.017,
    "DFU": 0.972
  },
  "defer_to_clinician": false,
  "defer_reason": "",
  "quality_flags": [],
  "has_wound": true,
  "wound_area_mm2": 412.5,
  "wound_coverage_pct": 3.8,
  "inference_time_ms": 184.3,
  "segmentation_mask_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
  "diagnostics": null
}
```

To decode and save the returned mask:

```bash
curl -s -X POST http://localhost:8000/predict -F "file=@foot.jpg;type=image/jpeg" \
  | python -c "import sys,json,base64; m=json.load(sys.stdin)['segmentation_mask_base64']; open('mask.png','wb').write(base64.b64decode(m)) if m else print('no mask')"
```

## Response schemas

The endpoint response models live in `src/deploy/app.py`. The monitoring schemas live in
`src/deploy/schemas.py` and describe events written to the optional prediction log.

### PredictionResponse

Returned by `POST /predict`. See the field table in the [POST /predict](#post-predict) section
above for full descriptions. Fields: `classification`, `classification_confidence`,
`classification_probs`, `defer_to_clinician`, `defer_reason`, `quality_flags`, `has_wound`,
`wound_area_mm2`, `wound_coverage_pct`, `inference_time_ms`, `segmentation_mask_base64`
(nullable), `diagnostics` (nullable).

### HealthResponse

Returned by `GET /health`.

| Field | Type | Description |
|---|---|---|
| `status` | string | Service status string. |
| `model_loaded` | boolean | Whether the pipeline initialized. |
| `version` | string | API version. |

### ModelInfoResponse

Returned by `GET /model/info`. See the field table in the [GET /model/info](#get-modelinfo)
section above. Fields: `classifier`, `segmenter`, `input_size`, `num_classes`,
`confidence_threshold`, `defer_threshold`, `defer_threshold_source`, `max_image_size_mb`,
`rate_limit_rpm`, `version`.

### DriftFeatures

Lightweight input-distribution features captured for drift monitoring. Embedded inside each
`PredictionLogEvent`.

| Field | Type | Description |
|---|---|---|
| `brightness_mean` | float | Mean grayscale brightness of the input image. |
| `brightness_std` | float | Standard deviation of grayscale brightness. |
| `blur_variance` | float | Variance of the Laplacian, a sharpness proxy. |
| `width` | integer | Input image width in pixels. |
| `height` | integer | Input image height in pixels. |

### PredictionLogEvent

Structured event written as one JSON line per prediction when logging is enabled (see
`DIAFOOT_PREDICTION_LOG` in [Configuration](#configuration)). Intended for offline monitoring and
drift analysis, not returned to API clients.

| Field | Type | Description |
|---|---|---|
| `timestamp_utc` | string | UTC timestamp in ISO 8601 form, for example `"2026-07-17T12:34:56Z"`. |
| `classification` | string | The classification returned to the client. |
| `classification_confidence` | float | Confidence for the predicted class. |
| `defer_to_clinician` | boolean | Whether the case was deferred. |
| `defer_reason` | string | Reason for deferral, empty when not deferred. |
| `quality_flags` | array of strings | Image-quality flags detected. |
| `has_wound` | boolean | Whether a wound was found. |
| `wound_area_mm2` | float | Estimated wound area in square millimeters. |
| `drift` | `DriftFeatures` | Input-distribution features for the request. |

`PredictionLogEvent.now_timestamp()` produces the `timestamp_utc` value. Each logged line is the
`model_dump_json()` output of the event, appended to the configured log path.

## Guards and limits

`/predict` applies four guards in order: upload validation, image decode, the image-quality gate,
and the pipeline availability check. Rate limiting and the payload-size ceiling run as middleware
before the endpoint.

### Upload validation

Two layers protect the upload:

- **Content-Length middleware** (`MaxContentLengthMiddleware`) inspects the `Content-Length`
  header on requests to `/predict`. If it exceeds the configured maximum, the request is rejected
  with HTTP `413` and body `{"detail": "Request payload too large", "max_bytes": <n>}`. A
  non-numeric `Content-Length` returns HTTP `400`.
- **In-endpoint validation** (`validate_upload_metadata`) checks the actual upload after reading
  it. The content type must be one of `image/jpeg`, `image/jpg`, `image/png`, or `image/webp`; an
  unsupported type returns a `200` response with `classification: "Manual Review Required"` and
  `defer_reason: "invalid_content_type"`. If the read payload exceeds the maximum, the
  `defer_reason` is `"payload_too_large"`.

The maximum upload size defaults to 20 MB and is set by `DIAFOOT_MAX_IMAGE_SIZE_MB`.

### Image-quality gate

After a valid upload is decoded, `assess_image_quality` scores the image and can defer it before
any model runs. If OpenCV cannot decode the bytes into an image, the response is
`classification: "Error"` with `defer_reason: "invalid_image"`.

The gate produces `quality_flags` and defers the case (`classification: "Manual Review Required"`,
`defer_reason: "low_image_quality"`) if any flag is raised:

| Flag | Condition | Env override (default) |
|---|---|---|
| `low_resolution` | Shorter image side is below the minimum. | `DIAFOOT_MIN_IMAGE_SIDE` (256) |
| `blurry` | Laplacian variance is below the blur threshold. | `DIAFOOT_BLUR_VARIANCE_THRESHOLD` (10.0) |
| `too_dark` | Mean brightness is below the minimum. | `DIAFOOT_BRIGHTNESS_MIN` (20.0) |
| `too_bright` | Mean brightness is above the maximum. | `DIAFOOT_BRIGHTNESS_MAX` (235.0) |

When the gate passes, `quality_flags` is empty and inference proceeds.

### Rate limiting

`RateLimitMiddleware` applies an in-memory sliding-window limit per client IP address to
`/predict`. The default is 100 requests per 60-second window, set by `DIAFOOT_RATE_LIMIT_RPM`.
When a client exceeds the limit, the request is rejected with HTTP `429`, a `Retry-After` header,
and body `{"detail": "Rate limit exceeded", "retry_after_seconds": <n>}`. The limiter state is
per-process and resets on restart, so it does not coordinate across multiple workers or replicas.

### Defer-to-clinician behavior

The pipeline abstains rather than guessing whenever it cannot make a reliable call. When
`defer_to_clinician` is `true`, treat the case as one for human review. The `defer_reason` field
tells you why:

| `defer_reason` | Meaning | Resulting `classification` |
|---|---|---|
| `invalid_content_type` | Upload was not an allowed image type. | `"Manual Review Required"` |
| `payload_too_large` | Upload exceeded the size limit. | `"Manual Review Required"` |
| `invalid_image` | Bytes could not be decoded into an image. | `"Error"` |
| `low_image_quality` | Image failed the brightness, blur, or size gate. | `"Manual Review Required"` |
| `model_not_loaded` | Pipeline did not initialize (missing checkpoint). | `"Model not loaded"` |
| `low_classification_confidence` | Calibrated confidence fell below the defer threshold. | `"Healthy"`, `"Non-DFU"`, or `"DFU"` |
| `segmenter_unavailable` | Segmenter checkpoint was not loaded. | triage label |
| `segmentation_classifier_disagreement` | Segmenter found a wound the classifier did not call DFU. | `"DFU"` |

The possible `classification` values are therefore the three triage labels (`"Healthy"`,
`"Non-DFU"`, `"DFU"`) plus three guard states: `"Manual Review Required"` (upload or quality
failure), `"Error"` (undecodable image), and `"Model not loaded"` (pipeline unavailable).

The rationale behind the two abstention mechanisms, the quality gate and the confidence defer, is
explained in [explanation-pipeline-design.md](explanation-pipeline-design.md).

## Configuration

The running app reads its settings from environment variables, falling back to built-in defaults.
`configs/deploy/api.yaml` documents the intended deployment values for a serving stack; wire those
values through to the process environment (or your container spec) to apply them.

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `DIAFOOT_BACKBONE` | `dinov2_vitb14` | DINOv2 backbone for both classifier and segmenter. |
| `DIAFOOT_DEVICE` | `cuda` | Inference device. Falls back to `cpu` if CUDA is unavailable. |
| `DIAFOOT_CLASSIFIER_CKPT` (or `DIAFOOT_CLASSIFIER_CHECKPOINT`) | `checkpoints/dinov2_classifier/best_epoch009_0.9785.pt` | Classifier checkpoint path. |
| `DIAFOOT_SEGMENTER_CKPT` (or `DIAFOOT_SEGMENTER_CHECKPOINT`) | `checkpoints/dinov2_segmenter/best_epoch009_0.1062.pt` | Segmenter checkpoint path. |
| `DIAFOOT_CONFIDENCE_THRESHOLD` | `0.70` | Minimum confidence to trust a classification. |
| `DIAFOOT_DEFER_THRESHOLD` | `0.50` | Confidence below which a case defers. Highest priority for the defer threshold. |
| `DIAFOOT_CALIBRATION_PATH` | `results/classification_calibration.json` | Calibration artifact read for the recommended defer threshold when `DIAFOOT_DEFER_THRESHOLD` is unset. |
| `DIAFOOT_MAX_IMAGE_SIZE_MB` | `20.0` | Maximum upload size in megabytes. |
| `DIAFOOT_RATE_LIMIT_RPM` | `100` | Requests per minute per client on `/predict`. |
| `DIAFOOT_MIN_IMAGE_SIDE` | `256` | Minimum shorter-side length before `low_resolution` is flagged. |
| `DIAFOOT_BLUR_VARIANCE_THRESHOLD` | `10.0` | Laplacian variance below which `blurry` is flagged. |
| `DIAFOOT_BRIGHTNESS_MIN` | `20.0` | Mean brightness below which `too_dark` is flagged. |
| `DIAFOOT_BRIGHTNESS_MAX` | `235.0` | Mean brightness above which `too_bright` is flagged. |
| `DIAFOOT_DFU_SEG_FALLBACK_PROB` | `0.10` | DFU probability above which the segmenter runs as a fallback on uncertain triage. |
| `DIAFOOT_DFU_PROMOTION_THRESHOLD` | `0.04` | DFU probability above which a detected wound can promote a case to DFU. |
| `DIAFOOT_PREDICTION_LOG` | empty (disabled) | JSONL path for `PredictionLogEvent` records. |
| `DIAFOOT_CORS_ORIGINS` (or `CORS_ORIGINS`) | `*` | Comma-separated list of allowed CORS origins. |

The defer threshold resolves in priority order: `DIAFOOT_DEFER_THRESHOLD` if set, then the
recommended threshold from the calibration JSON if present, then the built-in default. The
`defer_threshold_source` field in `/model/info` reports which one won.

Example, serving on CPU with a custom threshold and prediction logging:

```bash
DIAFOOT_DEVICE=cpu \
DIAFOOT_DEFER_THRESHOLD=0.67 \
DIAFOOT_PREDICTION_LOG=logs/predictions.jsonl \
uvicorn src.deploy.app:app --host 0.0.0.0 --port 8000
```

### configs/deploy/api.yaml

The deployment config file mirrors these settings under an `api:` block:

| Key | Value in repo | Corresponds to |
|---|---|---|
| `host` | `0.0.0.0` | uvicorn bind address. |
| `port` | `8000` | uvicorn port. |
| `workers` | `4` | Number of server workers. |
| `cors_origins` | `["*"]` | `DIAFOOT_CORS_ORIGINS`. |
| `max_image_size_mb` | `20` | `DIAFOOT_MAX_IMAGE_SIZE_MB`. |
| `rate_limit` | `100` | `DIAFOOT_RATE_LIMIT_RPM` (requests per minute). |
| `model_dir` | `exports/` | Model artifact directory. |
| `log_level` | `info` | Server log level. |
| `backbone` | `dinov2_vitb14` | `DIAFOOT_BACKBONE`. |
| `classifier_checkpoint` | `checkpoints/dinov2_classifier/best_epoch009_0.9785.pt` | `DIAFOOT_CLASSIFIER_CKPT`. |
| `segmenter_checkpoint` | `checkpoints/dinov2_segmenter/best_epoch009_0.1062.pt` | `DIAFOOT_SEGMENTER_CKPT`. |
| `confidence_threshold` | `0.95` | `DIAFOOT_CONFIDENCE_THRESHOLD`. |
| `defer_threshold` | `0.67` | `DIAFOOT_DEFER_THRESHOLD` (preferred from calibration when available). |
| `calibration_report` | `results/classification_calibration.json` | `DIAFOOT_CALIBRATION_PATH`. |
| `min_image_side` | `256` | `DIAFOOT_MIN_IMAGE_SIDE`. |
| `blur_variance_threshold` | `30.0` | `DIAFOOT_BLUR_VARIANCE_THRESHOLD`. |
| `brightness_min` | `20.0` | `DIAFOOT_BRIGHTNESS_MIN`. |
| `brightness_max` | `235.0` | `DIAFOOT_BRIGHTNESS_MAX`. |

## Related

- [howto-serve-api.md](howto-serve-api.md) — deploy and run the service, including Docker.
- [reference-cli.md](reference-cli.md) — command-line tools for training, evaluation, and prediction.
- [explanation-pipeline-design.md](explanation-pipeline-design.md) — why the pipeline classifies, then segments, then defers.
- [../README.md](../README.md) — project overview and the honest results tables.
