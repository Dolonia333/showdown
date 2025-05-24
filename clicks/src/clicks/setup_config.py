import json
import os
import sys
from pathlib import Path

def setup_config():
    """Interactive setup for configuration"""
    config_path = Path(__file__).parent.parent / 'config.json'
    
    # Load existing config if it exists
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = {
            "models": {
                "lm_studio": {
                    "endpoint": "http://localhost:1234/v1",
                    "api_key": "not-needed"
                },
                "ollama": {
                    "endpoint": "http://localhost:11434",
                    "model": "deepseek"
                },
                "openrouter": {
                    "endpoint": "https://openrouter.ai/api/v1",
                    "api_key": ""
                }
            },
            "evaluation": {
                "max_tokens": 4096,
                "temperature": 0.7,
                "timeout": 30
            }
        }

    print("Showdown Configuration Setup")
    print("-" * 30)

    # LM Studio Config
    print("\nLM Studio Configuration:")
    config["models"]["lm_studio"]["endpoint"] = input(
        f"LM Studio endpoint [{config['models']['lm_studio']['endpoint']}]: "
    ).strip() or config["models"]["lm_studio"]["endpoint"]

    # Ollama Config
    print("\nOllama Configuration:")
    config["models"]["ollama"]["endpoint"] = input(
        f"Ollama endpoint [{config['models']['ollama']['endpoint']}]: "
    ).strip() or config["models"]["ollama"]["endpoint"]
    
    config["models"]["ollama"]["model"] = input(
        f"Ollama model name [{config['models']['ollama']['model']}]: "
    ).strip() or config["models"]["ollama"]["model"]

    # OpenRouter Config
    print("\nOpenRouter Configuration:")
    openrouter_key = input("OpenRouter API key: ").strip()
    if openrouter_key:
        config["models"]["openrouter"]["api_key"] = openrouter_key

    # Evaluation Settings
    print("\nEvaluation Settings:")
    try:
        config["evaluation"]["max_tokens"] = int(input(
            f"Max tokens [{config['evaluation']['max_tokens']}]: "
        ).strip() or config["evaluation"]["max_tokens"])
    except ValueError:
        print("Invalid value for max_tokens, keeping default")

    try:
        config["evaluation"]["temperature"] = float(input(
            f"Temperature [{config['evaluation']['temperature']}]: "
        ).strip() or config["evaluation"]["temperature"])
    except ValueError:
        print("Invalid value for temperature, keeping default")

    try:
        config["evaluation"]["timeout"] = int(input(
            f"Request timeout in seconds [{config['evaluation']['timeout']}]: "
        ).strip() or config["evaluation"]["timeout"])
    except ValueError:
        print("Invalid value for timeout, keeping default")

    # Save config
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\nConfiguration saved successfully!")
    print(f"Config file location: {config_path}")

if __name__ == "__main__":
    setup_config()
