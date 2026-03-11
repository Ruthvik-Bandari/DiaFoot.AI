"""Patch to add segmentation_mask_base64 to the FastAPI /predict response.

Apply this patch to src/deploy/app.py:

1. Add to imports at top:
   import base64
   from io import BytesIO
   from PIL import Image as PILImage

2. Add field to PredictionResponse:
   segmentation_mask_base64: str | None = None

3. Before the return PredictionResponse(...), add:
   mask_b64 = None
   if result.segmentation_mask is not None and result.has_wound:
       mask_img = PILImage.fromarray((result.segmentation_mask * 255).astype("uint8"))
       buf = BytesIO()
       mask_img.save(buf, format="PNG")
       mask_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

4. Add to the PredictionResponse return:
   segmentation_mask_base64=mask_b64,

---
Run this script to auto-patch:
    python scripts/patch_api_mask.py
"""

from pathlib import Path
import re

def patch():
    app_path = Path("src/deploy/app.py")
    content = app_path.read_text()

    # 1. Add imports if not present
    if "import base64" not in content:
        content = content.replace(
            "import time",
            "import base64\nimport time",
        )
    if "from io import BytesIO" not in content:
        content = content.replace(
            "import base64",
            "import base64\nfrom io import BytesIO",
        )
    if "from PIL import Image as PILImage" not in content:
        content = content.replace(
            "from io import BytesIO",
            "from io import BytesIO\n\nfrom PIL import Image as PILImage",
        )

    # 2. Add field to PredictionResponse
    if "segmentation_mask_base64" not in content:
        content = content.replace(
            "    inference_time_ms: float",
            "    inference_time_ms: float\n    segmentation_mask_base64: str | None = None",
        )

    # 3. Add mask encoding before return
    if "mask_b64" not in content:
        content = content.replace(
            "    return PredictionResponse(\n        classification=result.classification,",
            '''    # Encode segmentation mask as base64 PNG
    mask_b64: str | None = None
    if result.segmentation_mask is not None and result.has_wound:
        mask_img = PILImage.fromarray(
            (result.segmentation_mask * 255).astype("uint8")
        )
        buf = BytesIO()
        mask_img.save(buf, format="PNG")
        mask_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return PredictionResponse(
        classification=result.classification,''',
        )

    # 4. Add mask_b64 to return
    if "segmentation_mask_base64=mask_b64" not in content:
        content = content.replace(
            "        inference_time_ms=elapsed,\n    )",
            "        inference_time_ms=elapsed,\n        segmentation_mask_base64=mask_b64,\n    )",
        )

    app_path.write_text(content)
    print(f"Patched: {app_path}")
    print("Added: segmentation_mask_base64 field to PredictionResponse")


if __name__ == "__main__":
    patch()
