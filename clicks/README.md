# showdown-clicks

General Agents

[🤗 Dataset](https://huggingface.co/datasets/generalagents/showdown-clicks) | [GitHub](https://github.com/generalagents/showdown)

`showdown` is a suite of offline and online benchmarks for computer-use agents.

`showdown-clicks` is a collection of 5,679 left clicks of humans performing various tasks in a macOS desktop environment. It is intended to evaluate instruction-following and low-level control capabilities of computer-use agents.

As of March 2025, we are releasing a subset of the full set, `showdown-clicks-dev`, containing 557 clicks. All examples are annotated with the bounding box of viable click locations for the UI element.

The episodes range from tens of seconds to minutes, and screenshots are between WXGA (1280×800) and WSXGA+ (1680×1050). The recordings contain no PII and were collected in late 2024.

| Column | Description |
|--------|-------------|
| id | Unique identifier for each data entry (alphanumeric string) |
| image | Path to the screenshot image file showing the UI state |
| instruction | Natural language instruction describing the task to be performed |
| x1 | Top-left x-coordinate of the bounding box |
| y1 | Top-left y-coordinate of the bounding box | 
| x2 | Bottom-right x-coordinate of the bounding box |
| y2 | Bottom-right y-coordinate of the bounding box |
| width | Width of the image |
| height | Height of the image |

## `showdown-clicks-dev` Results

| Model                         | Accuracy      | 95% CI              | Latency [^1] |
|------------------------------|---------------|---------------------|--------------|
| Claude-3-Opus (OpenRouter)   | -             | -                   | -            |
| Dolphin Mixtral (Local)      | -             | -                   | -            |
| DeepSeek R1 14B (Local)      | -             | -                   | -            |

### Run evals
```bash
# OpenRouter API Models (requires OPENROUTER_API_KEY)
uv run eval.py --model openrouter --dataset dev --openrouter-model anthropic/claude-3-opus

# Local Dolphin Mixtral model (requires Ollama with dolphin-mixtral model)
uv run eval.py --model dolphin --dataset dev 

# Local DeepSeek model (requires Ollama with deepseek-r1:14b model)
uv run eval.py --model deepseek --dataset dev
```

When you are done with the evals, go to Modal's UI and terminate the individual apps.

## Directory Structure

The project is organized as follows:

- `data/`: Input data
  - `showdown-clicks-dev/data.csv`: Records
  - `showdown-clicks-dev/frames`: Image frames

- `results/`: Output data
  - CSV result files from evaluations
  - `showdown-clicks-dev/{$MODEL}/visualizations/`: Visualizations of model predictions
  - `report/`: Analysis reports and summary metrics

- `scripts/`: Utility scripts
  - `calculate_latency.py`: Script to calculate latency metrics
  - `collect_runs.py`: Script to collect results from multiple runs

- `src/clicks/`: Main source code
  - `api_client_base.py`: Base API client classes
  - `evaluate/`: Evaluation code
    - `ace.py`: Ace model implementation
    - `models.py`: Data models for evaluation
    - `utils.py`: Utilities for visualization and evaluation
  - `third_party/`: Third-party model integrations
    - `claude/`: Claude model integration
    - `gemini/`: Gemini model integration
    - `molmo/`: Molmo model integration
    - `omniparser/`: OmniParser model integration
    - `openai/`: OpenAI model integration
    - `openai_cua/`: OpenAI Computer Use Agent integration
    - `qwen/`: Qwen model integration
    - `ui_tars/`: UI-TARS model integration

## Setup

1. Install dependencies:
```bash
uv pip install -e .
```

2. Choose your evaluation method:

### OpenRouter API Models
1. Get an API key from [OpenRouter](https://openrouter.ai)
2. Set your API key:
```bash
export OPENROUTER_API_KEY=your_key_here
```

### Local Models via Ollama
1. Install [Ollama](https://ollama.ai)
2. Pull the required models:
```bash
# For Dolphin Mixtral
ollama pull dolphin-mixtral

# For DeepSeek
ollama pull deepseek-r1:14b
```

## Running Evaluations

1. Extract the dataset:
```bash
cd data/
tar -xf frames.tar
```

2. Run evaluations using one of:
```bash
# OpenRouter API Models
uv run eval.py --model openrouter --dataset dev --openrouter-model anthropic/claude-3-opus

# Local Dolphin Mixtral
uv run eval.py --model dolphin --dataset dev

# Local DeepSeek
uv run eval.py --model deepseek --dataset dev
```

Results will be saved in the `results/` directory.

## Usage

To run the evaluation, use the `eval.py` script:

```bash
# Run on the dev dataset (default)
uv run eval.py

# Run with a specific model (ace, claude, qwen, etc.)
uv run eval.py --model claude --api-key YOUR_API_KEY

# Run with a limited sample size (for testing)
uv run eval.py --sample-size 10

# Run with multiple workers for parallel processing
uv run eval.py --num-workers 4

# Run with a custom output file
uv run eval.py --output-file results/custom_results.csv
```

To run multi-model evaluation:
```bash
uv run multi_model_eval.py
```

To run voice evaluation:
```bash
uv run voice_eval.py
```

Requirements for voice_eval.py:
```bash
pip install SpeechRecognition pyaudio pyttsx3
```

Prompts should be in prompts.json in the same directory.
Results will be saved to results.json.

For OpenRouter, set OPENROUTER_API_KEY in your .env file or environment variables.

Local models must be running and accessible at the specified endpoints.

### Model-specific options

#### Computer-use agents

Claude:
```bash
uv run eval.py --model claude --api-key YOUR_ANTHROPIC_API_KEY --claude-model claude-3-7-sonnet-20250219 --thinking-budget 1024
```

Qwen:
```bash
uv run eval.py --model qwen --api-key YOUR_DASHSCOPE_API_KEY --qwen-model qwen2.5-vl-72b-instruct --max-tokens 4096
```

UI-TARS:
```bash
uv run eval.py --model ui-tars --api-url YOUR_UITARS_API_URL --api-key YOUR_API_KEY --ui-tars-model bytedance-research/UI-TARS-72B-SFT --max-tokens 128 --temperature 0.0 --frequency-penalty 1.0
```

OmniParser:
```bash
uv run eval.py --model omniparser --dataset dev --run-id showdown-clicks-dev --omniparser-model gpt-4o-2024-05-13 --api-url YOUR_OMNIPARSER_API_URL --omniparser-temperature 0.7
```

Operator:
```bash
uv run eval.py --model openai-cua --dataset dev --run-id showdown-clicks-dev --environment mac
```

Ace (default):
```bash
uv run eval.py --model ace
```

#### VLMs

OpenAI:
```bash
uv run eval.py --model openai --api-key YOUR_OPENAI_API_KEY --openai-model gpt-4o --dataset dev
```

Gemini:
```bash
uv run eval.py --model gemini --api-key YOUR_GEMINI_API_KEY --gemini-model gemini-1.5-pro-latest --dataset dev
```

Molmo:
```bash
uv run eval.py --model molmo --api-url YOUR_MOLMO_API_URL --dataset dev
```

## Environment Variables

Alternative to passing API keys as command-line arguments:

- `ANTHROPIC_API_KEY`: API key for Claude
- `DASHSCOPE_API_KEY`: API key for Qwen
- `OPENAI_API_KEY`: API key for OpenAI and OpenAI CUA
- `GEMINI_API_KEY`: API key for Gemini
- `GENERALAGENTS_API_KEY`: API key for General Agents (Ace)

## Visualization

The evaluation script generates visualizations of model predictions, showing both the ground truth click position, bounding box, and the predicted click position. These visualizations are saved in the `results/[run-id]/[model]/visualizations/` directory, organized by model and correctness.

## Results Format

The evaluation results are saved as CSV files in the `results/[run-id]/` directory. Each row in the CSV file contains:

| Column | Description |
|--------|-------------|
| id | Unique identifier for the evaluation item |
| recording_id | Identifier for the recording session |
| instruction | The instruction given to the model |
| image_path | Path to the image file |
| gt_x1 | Ground truth bounding box left X-coordinate |
| gt_y1 | Ground truth bounding box top Y-coordinate |
| gt_x2 | Ground truth bounding box right X-coordinate |
| gt_y2 | Ground truth bounding box bottom Y-coordinate |
| pred_x | Predicted X-coordinate |
| pred_y | Predicted Y-coordinate |
| is_in_bbox | Whether the prediction is within the ground truth bounding box |
| latency_seconds | Time taken for the model to make the prediction |
| visualization_path | Path to the visualization image |
| raw_response | Raw response from the model |

## Metrics

The evaluation script calculates the percentage of correct predictions (within the bounding box), with 95% confidence intervals created from bootstrapping.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

The images used in this evaluation dataset may contain content that some users might find offensive, inappropriate, or objectionable. These images are included solely for the purpose of evaluating model performance on realistic computer use scenarios.

We do not endorse, approve of, or claim responsibility for any content displayed in these images. The inclusion of any image in this dataset does not represent our views or opinions, and is not intended to promote any particular content, website, or viewpoint.

Researchers and users of this evaluation framework should be aware of this possibility when reviewing results and visualizations.

## Citation

If you use `showdown-clicks` in your research, please cite it as follows:

```bibtex
@misc{showdown2025,
  title={The Showdown Computer Control Evaluation Suite},
  author={General Agents Team},
  year={2025},
  url={https://github.com/generalagents/showdown},
}
```

[^1]: Latency values vary significantly by provider, demand, computational resources, geographical location, and other factors - most of which are opaque to us for models we don't have direct access to. Ace models are served via General Agent's API; Qwen, Claude, Gemini, and OpenAI models utilize their respective first-party APIs; while Molmo, UI-TARS, and OmniParser models are served through Modal.