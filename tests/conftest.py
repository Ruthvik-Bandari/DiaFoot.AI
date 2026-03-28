"""Pytest configuration — ensure src is importable."""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

warnings.filterwarnings(
    "ignore",
    message=r".*deprecated - use 'one_of'.*",
    module=r"matplotlib\._fontconfig_pattern",
)
warnings.filterwarnings(
    "ignore",
    message=r".*deprecated - use 'parse_string'.*",
    module=r"matplotlib\._fontconfig_pattern",
)
warnings.filterwarnings(
    "ignore",
    message=r".*deprecated - use 'reset_cache'.*",
    module=r"matplotlib\._fontconfig_pattern",
)
warnings.filterwarnings(
    "ignore",
    message=r".*deprecated - use 'enable_packrat'.*",
    module=r"matplotlib\._mathtext",
)
