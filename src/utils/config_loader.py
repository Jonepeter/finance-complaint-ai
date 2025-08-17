"""Configuration loader utility."""

import yaml
from pathlib import Path
from typing import Dict, Any

def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    # Find project root by looking for config directory
    current_path = Path(__file__).parent
    while current_path.parent != current_path:
        config_file = current_path / config_path
        if config_file.exists():
            break
        current_path = current_path.parent
    else:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    
    return config

def get_config_value(config: Dict[str, Any], key_path: str, default=None):
    """Get nested configuration value using dot notation."""
    keys = key_path.split('.')
    value = config
    
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    
    return value