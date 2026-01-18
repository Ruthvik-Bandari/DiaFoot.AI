"""
Inference Module
=================

Production inference utilities for wound analysis.
"""

from .predictor import WoundPredictor, load_predictor

__all__ = [
    "WoundPredictor",
    "load_predictor",
]
