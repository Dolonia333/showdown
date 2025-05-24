# Local Development Setup Guide

This guide will help you set up and run the evaluation framework locally.

## Prerequisites

1. LM Studio
   - Download and install from [LMStudio website](https://lmstudio.ai/)
   - Load the Dolphin-3 model
   - Start the server on port 1234

2. Ollama
   - Install from [Ollama website](https://ollama.ai/)
   - Pull the DeepSeek model: `ollama pull deepseek`
   - Start the Ollama service

3. OpenRouter (Optional)
   - Get an API key from [OpenRouter](https://openrouter.ai/)

## Quick Start

1. Install the package:
```bash
pip install -e .
```

2. Run the configuration setup:
```bash
python src/clicks/setup_config.py
```

3. Test the installation:
```bash
python src/clicks/multi_model_eval.py
```

## Configuration

The `config.json` file contains important settings:

```json
{
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
      "api_key": "YOUR_API_KEY"
    }
  },
  "evaluation": {
    "max_tokens": 4096,
    "temperature": 0.7,
    "timeout": 30
  }
}
```

## Troubleshooting

### Common Issues

1. LM Studio Connection Errors
   - Ensure LM Studio is running
   - Check the server is started in LM Studio
   - Verify port 1234 is not blocked

2. Ollama Issues
   - Check Ollama service is running
   - Verify model is downloaded
   - Check port 11434 is accessible

3. OpenRouter Issues
   - Verify API key is correct
   - Check API endpoint URL
   - Ensure proper network connectivity

### Logs

Check the following log locations:
- LM Studio: Check the application logs
- Ollama: System logs (`journalctl -u ollama` on Linux)
- Application: Check terminal output

## Support

For additional help:
1. Check the GitHub issues
2. Review the evaluation logs
3. Verify configuration settings
