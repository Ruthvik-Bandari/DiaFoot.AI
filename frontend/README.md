# DiaFoot.AI Dashboard

Production-grade frontend for the DiaFoot.AI v2 diabetic foot ulcer detection system.

## Tech Stack

- **Runtime:** Bun
- **Framework:** Next.js 16 (App Router) + TypeScript
- **Compiler:** Turbopack
- **UI:** Material UI 7 + MUI Icons
- **Animations:** GSAP + @gsap/react
- **Data Fetching:** TanStack Query
- **Forms:** React Hook Form + Zod
- **Linting:** OxLint

## Setup

```bash
# Install dependencies
bun install

# Start development server (Turbopack)
bun dev

# Build for production
bun run build

# Start production server
bun start
```

## Backend

The dashboard connects to the DiaFoot.AI FastAPI backend at `http://127.0.0.1:8000` by default.

### Start the backend:
```bash
cd ~/Desktop/"Diafoot CV"
source .venv/bin/activate
export DIAFOOT_CLASSIFIER_CKPT="checkpoints/dinov2_classifier/best_epoch009_0.9785.pt"
export DIAFOOT_SEGMENTER_CKPT="checkpoints/dinov2_segmenter/best_epoch009_0.1062.pt"
export DIAFOOT_CALIBRATION_PATH="results/classification_calibration.json"
export DIAFOOT_DEFER_THRESHOLD="0.60"
export DIAFOOT_MIN_IMAGE_SIDE=64
uvicorn src.deploy.app:app --host 0.0.0.0 --port 8000
```

### Patch API for mask overlay (one-time):
```bash
cd ~/Desktop/"Diafoot CV"
python scripts/patch_api_mask.py
```

### Demo Mode
Demo mode is **opt-in**.

To enable simulated responses (when backend is unavailable), set:

```bash
NEXT_PUBLIC_ENABLE_DEMO_MODE=true
```

For actual model inference, keep:

```bash
NEXT_PUBLIC_ENABLE_DEMO_MODE=false
```

## Configuration

Edit `.env.local`:
```
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
NEXT_PUBLIC_ENABLE_DEMO_MODE=false
```

## Disclaimer

⚠️ This is an academic project. NOT a medical device. NOT for clinical use.
