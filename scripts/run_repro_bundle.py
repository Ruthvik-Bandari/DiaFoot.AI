"""DiaFoot.AI v2 — Reproducibility bundle generator."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_info(repo_dir: Path) -> dict[str, str]:
    def _run(args: list[str]) -> str:
        try:
            out = subprocess.check_output(args, cwd=repo_dir, stderr=subprocess.DEVNULL)
            return out.decode().strip()
        except Exception:
            return ""

    return {
        "commit": _run(["git", "rev-parse", "HEAD"]),
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "status": _run(["git", "status", "--short"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate reproducibility bundle manifest")
    parser.add_argument("--output", type=str, default="results/repro/repro_bundle.json")
    parser.add_argument("--include", type=str, nargs="*", default=[])
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    ts = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")

    pkg_versions = {}
    for mod in ("torch", "numpy", "scipy", "sklearn", "cv2"):
        try:
            m = __import__(mod)
            pkg_versions[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            pkg_versions[mod] = "not_installed"

    files = []
    for rel in args.include:
        p = (root / rel).resolve()
        if p.exists() and p.is_file():
            files.append(
                {
                    "path": str(p.relative_to(root)),
                    "sha256": _sha256(p),
                    "size_bytes": p.stat().st_size,
                }
            )

    manifest = {
        "generated_at_utc": ts,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "python": sys.version,
        },
        "environment": {
            "cwd": str(root),
            "venv": os.getenv("VIRTUAL_ENV", ""),
            "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", ""),
        },
        "git": _git_info(root),
        "package_versions": pkg_versions,
        "included_files": files,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Repro bundle written to: {out}")


if __name__ == "__main__":
    main()
