"""
Training Module
================

This module contains training utilities:
    - Training loops
    - Loss functions
    - Optimizers and schedulers
    - Callbacks and checkpointing
"""

from .trainer import Trainer, SegmentationTrainer, ClassificationTrainer
from .losses import (
    DiceLoss, 
    DiceBCELoss, 
    FocalLoss, 
    TverskyLoss,
    FocalTverskyLoss,
    BoundaryLoss,
    LovaszHingeLoss,
    CombinedLoss,
    LabelSmoothingCrossEntropy,
    create_segmentation_loss,
    create_classification_loss,
)
from .callbacks import (
    Callback,
    CallbackList,
    EarlyStopping, 
    ModelCheckpoint, 
    LearningRateLogger,
    MetricsLogger,
    WandBLogger,
    ProgressCallback,
)
from .optimizers import (
    get_optimizer, 
    get_scheduler,
    WarmupCosineScheduler,
    PolynomialLRDecay,
    ExponentialMovingAverage,
)

__all__ = [
    # Trainers
    "Trainer",
    "SegmentationTrainer",
    "ClassificationTrainer",
    # Losses
    "DiceLoss",
    "DiceBCELoss",
    "FocalLoss",
    "TverskyLoss",
    "FocalTverskyLoss",
    "BoundaryLoss",
    "LovaszHingeLoss",
    "CombinedLoss",
    "LabelSmoothingCrossEntropy",
    "create_segmentation_loss",
    "create_classification_loss",
    # Callbacks
    "Callback",
    "CallbackList",
    "EarlyStopping",
    "ModelCheckpoint",
    "LearningRateLogger",
    "MetricsLogger",
    "WandBLogger",
    "ProgressCallback",
    # Optimizers
    "get_optimizer",
    "get_scheduler",
    "WarmupCosineScheduler",
    "PolynomialLRDecay",
    "ExponentialMovingAverage",
]
