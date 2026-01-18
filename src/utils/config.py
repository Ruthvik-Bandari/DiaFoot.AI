"""
Configuration Management Utilities
===================================

Functions for loading, saving, and managing configuration files.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing configuration
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """
    Save configuration to a YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save the configuration
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def merge_configs(
    base_config: Dict[str, Any],
    override_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Recursively merge two configuration dictionaries.
    
    Values in override_config take precedence over base_config.
    
    Args:
        base_config: Base configuration
        override_config: Configuration with override values
        
    Returns:
        Merged configuration dictionary
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def get_config_value(
    config: Dict[str, Any],
    key_path: str,
    default: Any = None
) -> Any:
    """
    Get a value from nested config using dot notation.
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated path (e.g., 'training.optimizer.lr')
        default: Default value if key not found
        
    Returns:
        Configuration value or default
        
    Example:
        >>> config = {'training': {'optimizer': {'lr': 0.001}}}
        >>> get_config_value(config, 'training.optimizer.lr')
        0.001
    """
    keys = key_path.split('.')
    value = config
    
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    
    return value


def validate_config(config: Dict[str, Any], required_keys: list) -> bool:
    """
    Validate that all required keys exist in config.
    
    Args:
        config: Configuration dictionary
        required_keys: List of required key paths (dot notation)
        
    Returns:
        True if all required keys exist
        
    Raises:
        ValueError: If any required key is missing
    """
    missing = []
    
    for key_path in required_keys:
        if get_config_value(config, key_path) is None:
            missing.append(key_path)
    
    if missing:
        raise ValueError(f"Missing required configuration keys: {missing}")
    
    return True


class Config:
    """
    Configuration wrapper class for easier access.
    
    Example:
        >>> cfg = Config('configs/config.yaml')
        >>> print(cfg.training.batch_size)
        8
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        if config_path:
            self._config = load_config(config_path)
        else:
            self._config = {}
    
    def __getattr__(self, name: str) -> Any:
        """Get config value as attribute."""
        if name.startswith('_'):
            return super().__getattribute__(name)
        
        value = self._config.get(name)
        
        if isinstance(value, dict):
            return Config._dict_to_config(value)
        
        return value
    
    def __getitem__(self, key: str) -> Any:
        """Get config value by key."""
        return self._config.get(key)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get config value with default."""
        return get_config_value(self._config, key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self._config.copy()
    
    @staticmethod
    def _dict_to_config(d: Dict[str, Any]) -> 'Config':
        """Convert dictionary to Config object."""
        cfg = Config()
        cfg._config = d
        return cfg


# Default configuration path
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "configs" / "config.yaml"


def get_default_config() -> Config:
    """Get the default configuration."""
    if DEFAULT_CONFIG_PATH.exists():
        return Config(DEFAULT_CONFIG_PATH)
    return Config()
