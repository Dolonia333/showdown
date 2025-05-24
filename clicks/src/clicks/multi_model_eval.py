import json
import requests
import time
from datetime import datetime
import os
from typing import Dict, Any, List
from .config import load_config, get_model_config

PROMPTS_FILE = "prompts.json"
RESULTS_FILE = "results.json"

class ModelError(Exception):
    """Custom exception for model errors"""
    pass

def get_models() -> List[Dict[str, Any]]:
    """Get model configurations with proper error handling"""
    config = load_config()
    models = []
    
    # LM Studio (Dolphin)
    try:
        lm_studio_config = get_model_config('lm_studio')
        models.append({
            "name": "Dolphin 3 (LM Studio)",
            "url": f"{lm_studio_config['endpoint']}/chat",
            "type": "openai",
            "headers": {},
            "body_fn": lambda prompt: {
                "model": "dolphin-3",
                "messages": [{"role": "user", "content": prompt}]
            }
        })
    except (KeyError, ValueError) as e:
        print(f"Warning: Failed to configure LM Studio: {e}")

    # Ollama (DeepSeek)
    try:
        ollama_config = get_model_config('ollama')
        models.append({
            "name": "DeepSeek-R1:14B (Ollama)",
            "url": f"{ollama_config['endpoint']}/api/generate",
            "type": "ollama",
            "headers": {},
            "body_fn": lambda prompt: {
                "model": ollama_config.get('model', 'deepseek'),
                "prompt": prompt
            }
        })
    except (KeyError, ValueError) as e:
        print(f"Warning: Failed to configure Ollama: {e}")

    # OpenRouter
    try:
        openrouter_config = get_model_config('openrouter')
        models.append({
            "name": "OpenRouter (Mistral-7B-Instruct)",
            "url": f"{openrouter_config['endpoint']}/chat",
            "type": "openai",
            "headers": {"Authorization": f"Bearer {openrouter_config['api_key']}"},
            "body_fn": lambda prompt: {
                "model": "mistralai/mistral-7b-instruct",
                "messages": [{"role": "user", "content": prompt}]
            }
        })
    except (KeyError, ValueError) as e:
        print(f"Warning: Failed to configure OpenRouter: {e}")

    if not models:
        raise ValueError("No models were successfully configured")
    
    return models

def query_model(model: Dict[str, Any], prompt: str) -> str:
    """Query a model with improved error handling and timeout"""
    config = load_config()
    timeout = config.get('evaluation', {}).get('timeout', 30)
    
    try:
        body = model["body_fn"](prompt)
        response = requests.post(
            model["url"],
            headers=model["headers"],
            json=body,
            timeout=timeout
        )
        response.raise_for_status()
        data = response.json()
        
        if model["type"] == "openai":
            return data["choices"][0]["message"]["content"]
        elif model["type"] == "ollama":
            return data["response"]
        else:
            raise ModelError(f"Unknown model type: {model['type']}")
            
    except requests.exceptions.RequestException as e:
        error_msg = f"Request failed: {str(e)}"
        if isinstance(e, requests.exceptions.ConnectionError):
            error_msg = f"Connection failed - Is the model server running? Error: {str(e)}"
        elif isinstance(e, requests.exceptions.Timeout):
            error_msg = f"Request timed out after {timeout} seconds"
        raise ModelError(error_msg) from e
    except (KeyError, IndexError) as e:
        raise ModelError(f"Unexpected response format: {str(e)}") from e

def main():
    with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    models = get_models()
    results = []
    for item in prompts:
        prompt = item["prompt"]
        print(f"\nPrompt: {prompt}")
        entry = {"prompt": prompt, "timestamp": datetime.now().isoformat()}
        for model in models:
            print(f"  Querying {model['name']}...")
            result = query_model(model, prompt)
            print(f"    {model['name']} response:\n{result}\n")
            entry[model["name"]] = result
            time.sleep(1)  # avoid rate limits
        results.append(entry)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nAll results saved to {RESULTS_FILE}")

if __name__ == "__main__":
    main()
