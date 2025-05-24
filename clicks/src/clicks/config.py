import json
import os
from typing import Dict, Any

def load_config() -> Dict[str, Any]:
    """Load configuration from config.json file"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.json')
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise ValueError(f"Configuration file not found at {config_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON in configuration file at {config_path}")

def get_model_config(model_name: str) -> Dict[str, Any]:
    """Get configuration for a specific model"""
    config = load_config()
    model_config = config.get('models', {}).get(model_name)
    if not model_config:
        raise ValueError(f"Configuration for model {model_name} not found")
    return model_config

def get_evaluation_config() -> Dict[str, Any]:
    """Get evaluation configuration"""
    config = load_config()
    return config.get('evaluation', {})
