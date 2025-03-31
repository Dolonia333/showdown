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

| Model                                                | Accuracy      | 95% CI              | Latency [^1] | 95% CI                  |
|------------------------------------------------------|---------------|---------------------|--------------|-------------------------|
| `ace-control-medium`                                         | **77.56%**    | +3.41%/-3.59%       | 533ms        | +8ms/-7ms               |
| `ace-control-small`                                          | 72.89%        | +3.59%/-3.77%       | **324ms**    | +7ms/-7ms               |
| Operator (OpenAI CUA, macOS)                         | 64.27%        | +3.95%/-3.95%       | 6385ms       | +182ms/-177ms           |
| Molmo-72B-0924                                       | 54.76%        | +4.13%/-4.13%       | 6599ms       | +113ms/-114ms           |
| Claude 3.7 Sonnet (Thinking, Computer Use)           | 53.68%        | +4.13%/-4.13%       | 9656ms       | +95ms/-97ms             |
| UI-TARS-72B-SFT                                      | 54.4%         | +4.13%/-4.13%       | 1977ms       | +15ms/-16ms             |
| OmniParser V2 + GPT-4o                               | 51.71%        | +4.12%/-4.13%       | 12642ms      | +361ms/-349ms           |
| Gemini 2.0 Flash                                     | 33.39%        | +3.95%/-3.95%       | 3069ms       | +16ms/-16ms             |
| Qwen2.5-VL-72B-Instruct                              | 24.78%        | +3.59%/-3.60%       | 3790ms       | +57ms/-55ms             |
| GPT-4o                                               | 5.21%         | +1.97%/-1.80%       | 2500ms       | +49ms/-48ms             |

### Run evals
```bash
uv run eval.py --model ace --dataset dev --num-workers 1 --run-id showdown-clicks-dev
uv run eval.py --model claude --dataset dev --num-workers 16 --run-id showdown-clicks-dev
uv run eval.py --model qwen --dataset dev --num-workers 3 --run-id showdown-clicks-dev
uv run eval.py --model gemini --dataset dev --num-workers 16 --run-id showdown-clicks-dev
uv run eval.py --model openai --dataset dev --num-workers 16 --run-id showdown-clicks-dev
uv run eval.py --model openai-cua --dataset dev --num-workers 16 --run-id showdown-clicks-dev
uv run eval.py --model molmo --dataset dev --num-workers 2 --run-id showdown-clicks-dev --api-url $YOUR_MOLMO_MODAL_API
uv run eval.py --model ui-tars --dataset dev --run-id showdown-clicks-dev --api-url $YOUR_UITARS_MODAL_API --api-key $YOUR_UITARS_API_KEY --num-workers 1 --ui-tars-model bytedance-research/UI-TARS-72B-SFT
uv run eval.py --model omniparser --dataset dev --run-id showdown-clicks-dev --omniparser-model gpt-4o-2024-05-13 --api-url $YOUR_OMNIPARSER_MODAL_API  --num-workers 4
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