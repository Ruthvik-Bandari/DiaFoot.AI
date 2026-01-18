"""
DiaFootAI - Diabetic Foot Wound Assessment System
==================================================

A deep learning system for automated assessment of diabetic foot wounds.

Modules:
    - data: Data loading and preprocessing
    - models: Neural network architectures
    - training: Training loops and utilities
    - inference: Inference pipelines
    - evaluation: Metrics and evaluation
    - utils: Helper functions
"""

__version__ = "0.1.0"
__author__ = "Ruthvik"
__email__ = ""

from . import data
from . import models
from . import training
from . import inference
from . import evaluation
from . import utils

__all__ = [
    "data",
    "models",
    "training",
    "inference",
    "evaluation",
    "utils",
    "__version__",
]
