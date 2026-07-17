# Getting started with DiaFoot.AI

You'll install DiaFoot.AI, run a prediction on a foot image, and call the REST API, end to end. By the end you will have the package importing in a fresh virtual environment, a first prediction printed to your terminal, and a running HTTP service you can `curl`. Every step below produces something you can see on screen.

DiaFoot.AI is a multi-task pipeline for diabetic foot images: it triages an image (Healthy / Non-DFU / DFU), and when a wound is likely it segments the wound and estimates its area. For what the models actually score on the held-out test set, see the results section in [../README.md](../README.md). This tutorial does not repeat those numbers, and the example values shown after each command are illustrative placeholders, not benchmarks.

## What you'll need

- Python 3.12 or 3.13 (the project pins `requires-python = ">=3.12,<3.14"`).
- `git` and a terminal.
- One foot image in JPEG or PNG (any photo works for a first run; a real diabetic-foot photo gives a meaningful classification).
- A GPU is optional. Inference runs fine on CPU, and every command here uses CPU by default.

## Step 1: Get the code and install

Clone the repository, create an isolated virtual environment, and install the package in editable mode with the developer extras.

```bash
git clone https://github.com/Ruthvik-Bandari/DiaFoot.AI.git
cd DiaFoot.AI
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

The install pulls PyTorch, FastAPI, and the imaging stack, so the first run takes a few minutes. When it finishes, confirm the package imports:

```bash
python -c "import src; print('DiaFoot.AI import OK')"
```

Expected output:

```
DiaFoot.AI import OK
```

If you want a fuller check, run the test suite:

```bash
pytest -q
```

You should see a run of passing tests (some are marked `slow` or `gpu` and may be skipped without a GPU). Either result means the environment is ready.

## Step 2: Run your first prediction

This is your first result. The prediction CLI takes a single image and prints the classification and, when a wound is found, the segmentation summary.

```bash
python scripts/predict.py --image path/to/foot.jpg --device cpu
```

The real flags are: `--image` (required), `--device` (default `cpu`), `--backbone` (default `dinov2_vitb14`, also accepts `dinov2_vits14` or `dinov2_vitl14`), `--classifier-checkpoint` (default `checkpoints/dinov2_classifier/best.pt`), `--segmenter-checkpoint` (default `checkpoints/dinov2_segmenter/best.pt`), and `--save-mask` to write the wound mask to a PNG.

### About the checkpoints (read this before you expect a diagnosis)

Trained model weights are **not shipped in the repository**. The `.gitignore` excludes `checkpoints/` and every `*.pt` file, so a clean clone has an empty checkpoints tree. You have two ways to get weights:

- Train them yourself. See [howto-train.md](howto-train.md).
- Obtain a checkpoint out of band and place it at the paths above: `checkpoints/dinov2_classifier/best.pt` and `checkpoints/dinov2_segmenter/best.pt` (or pass your own paths with `--classifier-checkpoint` / `--segmenter-checkpoint`).

The script is written to run either way, so you still get a visible result right now.

**On a fresh clone (no checkpoints yet)** the script loads and preprocesses your image, reports that the weights are missing, and exits cleanly:

```
==================================================
DiaFoot.AI v2 — Inference (DINOv2)
Image: path/to/foot.jpg
Backbone: dinov2_vitb14
==================================================

  Classifier checkpoint not found: checkpoints/dinov2_classifier/best.pt

  Segmenter checkpoint not found: checkpoints/dinov2_segmenter/best.pt

==================================================
```

Seeing this confirms your install works: the image was read, resized to 518x518, and the pipeline ran up to the point where weights are required.

**Once a checkpoint is in place**, the same command prints a full prediction:

```
==================================================
DiaFoot.AI v2 — Inference (DINOv2)
Image: path/to/foot.jpg
Backbone: dinov2_vitb14
==================================================

  Classification: DFU (Diabetic Foot Ulcer)
  Confidence: 91.2%
    Healthy: 3.1%
    Non-DFU Wound: 5.7%
    DFU (Diabetic Foot Ulcer): 91.2%

  Segmentation:
    Wound detected: Yes
    Wound pixels: 12,345
    Coverage: 4.6%
    Estimated area: 3086.2 mm2

==================================================
```

The numbers above are placeholders to show the output shape. Here is how to read the real thing:

- **Classification** is the triage verdict: `Healthy`, `Non-DFU Wound`, or `DFU (Diabetic Foot Ulcer)`.
- **Confidence** is the model's probability for the winning class, followed by the full per-class breakdown.
- **Segmentation** runs only when a wound is plausible. `Coverage` is the share of the image covered by wound pixels, and `Estimated area` converts wound pixels to mm2 using an assumed 0.5 mm per pixel.

The CLI does not print a defer flag. That "defer to a clinician" decision lives in the REST API pipeline, which you'll see next. To also save the wound mask as an image, add `--save-mask wound_mask.png`.

## Step 3: Start the API and call it

The same models are served over HTTP by a FastAPI app. Start it with uvicorn:

```bash
uvicorn src.deploy.app:app --host 0.0.0.0 --port 8000
```

Leave that running and open a second terminal. First, check health:

```bash
curl http://localhost:8000/health
```

On a fresh clone with no checkpoints, the service starts but reports that the model did not load:

```json
{"status": "healthy", "model_loaded": false, "version": "2.1.0"}
```

That is expected. The service is up (`status` is `healthy`) but `model_loaded` is `false` because no weights were found. The API looks for `checkpoints/dinov2_classifier/best_epoch009_0.9785.pt` and `checkpoints/dinov2_segmenter/best_epoch009_0.1062.pt` by default. Point it at whatever checkpoint you have with environment variables, then restart uvicorn:

```bash
export DIAFOOT_CLASSIFIER_CKPT=checkpoints/dinov2_classifier/best.pt
export DIAFOOT_SEGMENTER_CKPT=checkpoints/dinov2_segmenter/best.pt
uvicorn src.deploy.app:app --host 0.0.0.0 --port 8000
```

Now send an image to the prediction endpoint. It accepts a multipart file upload:

```bash
curl -F "file=@path/to/foot.jpg" http://localhost:8000/predict
```

**When the model is not loaded**, `/predict` still responds and tells you why, routing the case to a human:

```json
{
  "classification": "Model not loaded",
  "classification_confidence": 0.0,
  "classification_probs": {},
  "defer_to_clinician": true,
  "defer_reason": "model_not_loaded",
  "quality_flags": [],
  "has_wound": false,
  "wound_area_mm2": 0.0,
  "wound_coverage_pct": 0.0,
  "inference_time_ms": 0.0
}
```

**When the model is loaded**, a successful prediction looks like this (placeholder values shown for shape, not benchmarks):

```json
{
  "classification": "DFU",
  "classification_confidence": 0.91,
  "classification_probs": {"Healthy": 0.03, "Non-DFU": 0.06, "DFU": 0.91},
  "defer_to_clinician": false,
  "defer_reason": "",
  "quality_flags": [],
  "has_wound": true,
  "wound_area_mm2": 3086.2,
  "wound_coverage_pct": 4.6,
  "inference_time_ms": 812.4,
  "segmentation_mask_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
  "diagnostics": null
}
```

Reading the JSON:

- `classification` is `Healthy`, `Non-DFU`, or `DFU` (the API uses shorter labels than the CLI), with `classification_probs` giving all three.
- `defer_to_clinician` is the safety flag. It flips to `true` for low-confidence calls, low image quality (see `quality_flags` such as `blurry` or `too_dark`), or when the classifier and segmenter disagree, with the cause in `defer_reason`.
- `has_wound`, `wound_area_mm2`, and `wound_coverage_pct` summarize the segmentation, and `segmentation_mask_base64` is the mask as a base64 PNG when a wound is found.

You can also open `http://localhost:8000/docs` in a browser for the interactive API explorer, and `GET /model/info` for the loaded model metadata and active thresholds. Full endpoint and field details are in [reference-api.md](reference-api.md).

## What you built

You now have DiaFoot.AI installed in a clean virtual environment, a first prediction printed from the CLI, and the REST API running and answering `/health` and `/predict`. You also know the one thing a clean checkout is missing: trained checkpoints, and where they go.

Where to go next:

- Train your own checkpoints: [howto-train.md](howto-train.md).
- Build the dataset the models train on: [howto-run-data-pipeline.md](howto-run-data-pipeline.md).
- Full API reference (every endpoint, field, and threshold): [reference-api.md](reference-api.md).
- Why the pipeline is designed this way (triage first, segment second, defer to a human): [explanation-pipeline-design.md](explanation-pipeline-design.md).
- Project overview and the honest test-set results: [../README.md](../README.md).
