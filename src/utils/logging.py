"""
Logging Utilities
==================

Functions for setting up and managing logging.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

# Try to import rich for better console output
try:
    from rich.console import Console
    from rich.logging import RichHandler
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


def setup_logging(
    level: Union[str, int] = "INFO",
    log_file: Optional[Union[str, Path]] = None,
    use_rich: bool = True,
    name: Optional[str] = None
) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        use_rich: Whether to use rich formatting (if available)
        name: Logger name (None for root logger)
        
    Returns:
        Configured logger
    """
    # Convert string level to int
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    
    # Get or create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    simple_formatter = logging.Formatter('%(levelname)s: %(message)s')
    
    # Console handler
    if use_rich and RICH_AVAILABLE:
        console_handler = RichHandler(
            console=Console(stderr=True),
            show_time=True,
            show_level=True,
            show_path=False,
            markup=True,
            rich_tracebacks=True,
        )
        console_handler.setFormatter(logging.Formatter('%(message)s'))
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(simple_formatter)
    
    console_handler.setLevel(level)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "diafootai") -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def create_experiment_log_path(
    base_dir: Union[str, Path],
    experiment_name: Optional[str] = None
) -> Path:
    """
    Create a unique log path for an experiment.
    
    Args:
        base_dir: Base directory for logs
        experiment_name: Optional experiment name
        
    Returns:
        Path to log file
    """
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if experiment_name:
        filename = f"{experiment_name}_{timestamp}.log"
    else:
        filename = f"experiment_{timestamp}.log"
    
    return base_dir / filename


class LoggerContext:
    """
    Context manager for temporarily changing log level.
    
    Example:
        >>> with LoggerContext('DEBUG'):
        ...     # Detailed logging
        ...     pass
    """
    
    def __init__(self, level: str, logger_name: Optional[str] = None):
        self.level = getattr(logging, level.upper())
        self.logger = logging.getLogger(logger_name)
        self.previous_level = None
    
    def __enter__(self):
        self.previous_level = self.logger.level
        self.logger.setLevel(self.level)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.setLevel(self.previous_level)


class ProgressLogger:
    """
    Logger for tracking training progress.
    
    Example:
        >>> progress = ProgressLogger(total_epochs=100, logger=logger)
        >>> for epoch in range(100):
        ...     progress.log_epoch(epoch, {'loss': 0.5, 'accuracy': 0.9})
    """
    
    def __init__(
        self,
        total_epochs: int,
        logger: Optional[logging.Logger] = None
    ):
        self.total_epochs = total_epochs
        self.logger = logger or get_logger()
        self.history = []
    
    def log_epoch(self, epoch: int, metrics: dict) -> None:
        """Log metrics for an epoch."""
        self.history.append({'epoch': epoch, **metrics})
        
        # Format metrics string
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        
        self.logger.info(
            f"Epoch [{epoch + 1}/{self.total_epochs}] {metrics_str}"
        )
    
    def log_batch(
        self,
        epoch: int,
        batch: int,
        total_batches: int,
        metrics: dict
    ) -> None:
        """Log metrics for a batch."""
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        
        self.logger.debug(
            f"Epoch [{epoch + 1}] Batch [{batch + 1}/{total_batches}] {metrics_str}"
        )
    
    def get_history(self) -> list:
        """Get logging history."""
        return self.history


def log_system_info(logger: Optional[logging.Logger] = None) -> None:
    """
    Log system and environment information.
    
    Args:
        logger: Logger to use (creates new one if None)
    """
    import platform
    import torch
    
    logger = logger or get_logger()
    
    logger.info("=" * 50)
    logger.info("System Information")
    logger.info("=" * 50)
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Python: {platform.python_version()}")
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"CUDA Available: {torch.cuda.is_available()}")
    logger.info(f"MPS Available: {torch.backends.mps.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        logger.info(
            f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )
    
    logger.info("=" * 50)
