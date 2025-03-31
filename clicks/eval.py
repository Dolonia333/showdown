import argparse
import multiprocessing
import os
from datetime import datetime
from functools import partial
from typing import Any, Dict, List, Optional

import clicks.evaluate.ace as ace
import pandas as pd
from clicks.api_client_base import AbstractAPIClient
from clicks.evaluate.models import EvaluationResult
from clicks.evaluate.utils import analyze_results
from clicks.third_party import (
  get_claude_api_client,
  get_gemini_api_client,
  get_molmo_api_client,
  get_omniparser_api_client,
  get_openai_api_client,
  get_openai_cua_api_client,
  get_qwen_api_client,
  get_ui_tars_api_client,
)
from colorama import Fore, Style, init
from tqdm import tqdm

init(autoreset=True)


def process_item(
  item: Dict[str, Any],
  frames_dir: str,
  api_client: AbstractAPIClient,
  run_id: str,
) -> EvaluationResult:
  try:
    image_path = item.get('image', '')

    if not image_path or image_path.startswith('data:'):
      print(f'Skipping item {item.get("id", "unknown")}: Invalid image path: {image_path}')
      raise ValueError(f'Invalid image path: {image_path}')

    if hasattr(image_path, 'item'):
      image_path = image_path.item()

    image_path = str(image_path)
    result = api_client.process_single_item(item, frames_dir, run_id)

    return result
  except Exception as e:
    print(f'Error processing item {item.get("id", "unknown")}: {str(e)}')
    raise e


def evaluate_csv(
  csv_file: str,
  frames_dir: str,
  api_client: Any,
  output_file: Optional[str] = None,
  sample_size: Optional[int] = None,
  num_workers: int = 1,
  run_id: str = datetime.now().strftime('%Y-%m-%d-%H-%M'),
) -> List[Dict[str, Any]]:
  results: List[Dict[str, Any]] = []

  df = pd.read_csv(csv_file)
  print(f'Loaded {len(df)} items from {csv_file}')

  if sample_size is not None and sample_size < len(df):
    df = df.sample(sample_size, random_state=42)
    print(f'Sampled {len(df)} items for evaluation')

  items: List[Dict[str, Any]] = [
    {str(k): v for k, v in item.items()} for item in df.to_dict('records')
  ]

  print(f'Using {num_workers} concurrent workers for processing')

  process_func = partial(
    process_item,
    frames_dir=frames_dir,
    api_client=api_client,
    run_id=run_id,
  )

  try:
    multiprocessing.set_start_method('spawn', force=True)
  except RuntimeError:
    pass

  results = []
  total_processed = 0
  total_in_bbox = 0

  try:
    with multiprocessing.Pool(processes=num_workers, maxtasksperchild=1) as pool:
      with tqdm(total=len(items), desc='Evaluating', unit='item') as pbar:
        for result in pool.imap_unordered(process_func, items):
          if result is not None:
            results.append(result.model_dump())
            total_processed += 1
            if result.is_in_bbox:
              total_in_bbox += 1
            running_accuracy = (total_in_bbox / total_processed) * 100 if total_processed > 0 else 0
            pbar.set_postfix({'accuracy': f'{running_accuracy:.2f}%'})

          pbar.update(1)
  except Exception as e:
    print(f'{Fore.RED}Error in multiprocessing: {e}{Style.RESET_ALL}')
    raise e

  if output_file and results:
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print(f'Results written to {output_file}')

  return results


def main():
  parser = argparse.ArgumentParser(
    description='Evaluate models on the clicks dataset with bounding box and pixel distance evaluation'
  )
  parser.add_argument(
    '--dataset',
    type=str,
    choices=['dev', 'full'],
    default='dev',
    help='Dataset to evaluate on (dev or full)',
  )
  parser.add_argument(
    '--model',
    type=str,
    choices=[
      'ace',
      'claude',
      'qwen',
      'openai',
      'openai-cua',
      'gemini',
      'molmo',
      'ui-tars',
      'omniparser',
    ],
    default='ace',
    help='Model to use for evaluation (ace, claude, qwen, openai, openai-cua, gemini, molmo, ui-tars, or omniparser)',
  )
  parser.add_argument(
    '--api-url',
    type=str,
    default='',
    help='API endpoint for the model',
  )
  parser.add_argument(
    '--api-key',
    type=str,
    default=None,
    help='API key for the model (if required)',
  )
  parser.add_argument(
    '--claude-model',
    type=str,
    default='claude-3-7-sonnet-20250219',
    help='Claude model to use (default: claude-3-7-sonnet-20250219)',
  )
  parser.add_argument(
    '--thinking-budget',
    type=int,
    default=1024,
    help='Budget for Claude thinking tokens (default: 1024, 0 to disable)',
  )
  parser.add_argument(
    '--tool-version',
    type=str,
    default='20250124',
    help='Version of Claude computer use tools (default: 20250124)',
  )
  parser.add_argument(
    '--qwen-model',
    type=str,
    default='qwen2.5-vl-72b-instruct',
    help='Qwen model to use (default: qwen2.5-vl-72b-instruct)',
  )
  parser.add_argument(
    '--openai-model',
    type=str,
    default='gpt-4o',
    help='OpenAI model to use (default: gpt-4o)',
  )
  parser.add_argument(
    '--openai-cua-model',
    type=str,
    default='computer-use-preview',
    help='OpenAI CUA model to use (default: computer-use-preview)',
  )
  parser.add_argument(
    '--environment',
    type=str,
    default='mac',
    choices=['browser', 'mac', 'windows', 'ubuntu'],
    help='Environment for OpenAI CUA (default: browser)',
  )
  parser.add_argument(
    '--reasoning-effort',
    type=str,
    default='medium',
    choices=['low', 'medium', 'high'],
    help='Reasoning effort for OpenAI (default: medium)',
  )
  parser.add_argument(
    '--max-tokens',
    type=int,
    default=4096,
    help='Maximum tokens for model response (default: 4096)',
  )
  parser.add_argument(
    '--sample-size',
    type=int,
    default=None,
    help='Number of samples to evaluate (optional, for testing)',
  )
  parser.add_argument(
    '--output-file', type=str, default=None, help='Path to output CSV file (optional)'
  )
  parser.add_argument(
    '--num-workers',
    type=int,
    default=1,
    help='Number of concurrent workers for processing (default: 1)',
  )
  parser.add_argument(
    '--run-id',
    type=str,
    default=None,
    help='Custom run ID (optional, defaults to current timestamp)',
  )
  parser.add_argument(
    '--gemini-model',
    type=str,
    default='gemini-1.5-pro-latest',
    help='Gemini model to use (default: gemini-1.5-pro-latest)',
  )
  parser.add_argument(
    '--top-p',
    type=float,
    default=0.9,
    help='Top-p sampling parameter (default: 0.9)',
  )
  parser.add_argument(
    '--top-k',
    type=int,
    default=50,
    help='Top-k sampling parameter (default: 50)',
  )
  parser.add_argument(
    '--temperature',
    type=float,
    default=0.0,
    help='Temperature for sampling (default: 0.0)',
  )
  parser.add_argument(
    '--frequency-penalty',
    type=float,
    default=1.0,
    help='Frequency penalty parameter for UI-TARS (default: 1.0)',
  )
  parser.add_argument(
    '--ui-tars-model',
    type=str,
    default='bytedance-research/UI-TARS-72B-SFT',
    help='UI-TARS model to use (default: bytedance-research/UI-TARS-72B-SFT)',
  )
  parser.add_argument(
    '--omniparser-model',
    type=str,
    default='gpt-4o-2024-05-13',
    help='OmniParser model to use (default: gpt-4o-2024-05-13)',
  )
  parser.add_argument(
    '--omniparser-temperature',
    type=float,
    default=0.7,
    help='Temperature for OmniParser generation (default: 0.7)',
  )
  parser.add_argument(
    '--ace-model',
    type=str,
    default='ace-control-medium',
    help='Ace model to use (default: ace-control-medium)',
  )
  args = parser.parse_args()

  base_dir = os.path.dirname(os.path.abspath(__file__))
  data_dir = os.path.join(base_dir, 'data')
  results_dir = os.path.join(base_dir, 'results')

  os.makedirs(results_dir, exist_ok=True)

  run_id = args.run_id or datetime.now().strftime('%Y-%m-%d-%H-%M')

  run_results_dir = os.path.join(results_dir, run_id)
  os.makedirs(run_results_dir, exist_ok=True)

  if args.dataset == 'dev':
    csv_file = os.path.join(data_dir, 'showdown-clicks-dev/data.csv')
  else:
    raise ValueError('Full dataset not currently supported')

  frames_dir = os.path.join(data_dir, 'showdown-clicks-dev')

  if args.output_file is None:
    if args.model == 'claude':
      model_name = args.claude_model.replace('-', '_')
      args.output_file = os.path.join(
        run_results_dir, f'claude_results_{model_name}_{args.dataset}.csv'
      )
    elif args.model == 'qwen':
      model_name = args.qwen_model.replace('-', '_').replace('.', '_')
      args.output_file = os.path.join(
        run_results_dir, f'qwen_results_{model_name}_{args.dataset}.csv'
      )
    elif args.model == 'openai':
      model_name = args.openai_model.replace('-', '_')
      args.output_file = os.path.join(
        run_results_dir, f'openai_results_{model_name}_{args.dataset}.csv'
      )
    elif args.model == 'openai-cua':
      model_name = args.openai_cua_model.replace('-', '_')
      args.output_file = os.path.join(
        run_results_dir, f'openai_cua_results_{model_name}_{args.dataset}.csv'
      )
    elif args.model == 'gemini':
      model_name = args.gemini_model.replace('-', '_').replace('.', '_')
      args.output_file = os.path.join(
        run_results_dir, f'gemini_results_{model_name}_{args.dataset}.csv'
      )
    elif args.model == 'molmo':
      args.output_file = os.path.join(run_results_dir, f'molmo_results_{args.dataset}.csv')
    elif args.model == 'ui-tars':
      model_name = args.ui_tars_model.replace('/', '_').replace('-', '_')
      args.output_file = os.path.join(
        run_results_dir, f'ui_tars_results_{model_name}_{args.dataset}.csv'
      )
    elif args.model == 'omniparser':
      model_name = args.omniparser_model.replace('-', '_')
      args.output_file = os.path.join(
        run_results_dir, f'omniparser_results_{model_name}_{args.dataset}.csv'
      )
    else:
      model_name = args.ace_model.replace('-', '_')
      args.output_file = os.path.join(
        run_results_dir, f'ace_results_{model_name}_{args.dataset}.csv'
      )

  print(f'{Fore.CYAN}Running evaluation with the following configuration:{Style.RESET_ALL}')
  print(f'{Fore.CYAN}  Dataset: {args.dataset}{Style.RESET_ALL}')
  print(f'{Fore.CYAN}  Model: {args.model}{Style.RESET_ALL}')
  print(f'{Fore.CYAN}  CSV file: {csv_file}{Style.RESET_ALL}')
  print(f'{Fore.CYAN}  Frames directory: {frames_dir}{Style.RESET_ALL}')
  print(f'{Fore.CYAN}  Run ID: {run_id}{Style.RESET_ALL}')
  print(f'{Fore.CYAN}  Results directory: {run_results_dir}{Style.RESET_ALL}')
  print(f'{Fore.CYAN}  Output file: {args.output_file}{Style.RESET_ALL}')
  print(f'{Fore.CYAN}  Concurrent workers: {args.num_workers}{Style.RESET_ALL}')

  if args.model == 'ace':
    print(f'  API URL: {args.api_url}')
    print(f'  Ace model: {args.ace_model}')
  elif args.model == 'claude':
    print(f'  Claude model: {args.claude_model}')
    print(f'  Thinking budget: {args.thinking_budget}')
    print(f'  Tool version: {args.tool_version}')
  elif args.model == 'qwen':
    print(f'  Qwen model: {args.qwen_model}')
    print(f'  Max tokens: {args.max_tokens}')
  elif args.model == 'openai':
    print(f'  OpenAI model: {args.openai_model}')
    print(f'  Max tokens: {args.max_tokens}')
    print(f'  Reasoning effort: {args.reasoning_effort}')
  elif args.model == 'openai-cua':
    print(f'  OpenAI CUA model: {args.openai_cua_model}')
    print(f'  Max tokens: {args.max_tokens}')
    print(f'  Environment: {args.environment}')
  elif args.model == 'gemini':
    print(f'  Gemini model: {args.gemini_model}')
    print(f'  Max tokens: {args.max_tokens}')
  elif args.model == 'molmo':
    print(f'  API URL: {args.api_url}')
    print(f'  Max tokens: {args.max_tokens}')
    print(f'  Temperature: {args.temperature}')
    print(f'  Top-p: {args.top_p}')
    print(f'  Top-k: {args.top_k}')
  elif args.model == 'ui-tars':
    print(f'  API URL: {args.api_url}')
    print(f'  UI-TARS model: {args.ui_tars_model}')
    print(f'  Max tokens: {args.max_tokens}')
    print(f'  Temperature: {args.temperature}')
    print(f'  Frequency penalty: {args.frequency_penalty}')
  elif args.model == 'omniparser':
    print(f'  API URL: {args.api_url or "https://omniparser-api-omniparser-api.modal.run"}')
    print(f'  OmniParser model: {args.omniparser_model}')
    print(f'  Temperature: {args.omniparser_temperature}')

  if args.sample_size:
    print(f'  Sample size: {args.sample_size}')

  if not os.path.exists(csv_file):
    print(f'{Fore.RED}ERROR: CSV file not found: {csv_file}{Style.RESET_ALL}')
    return

  if not os.path.exists(frames_dir):
    print(f'{Fore.RED}ERROR: Frames directory not found: {frames_dir}{Style.RESET_ALL}')
    return

  try:
    df = pd.read_csv(csv_file)
    print('\nFirst few rows of the CSV file:')
    print(df.head(2))
    print(f'\nTotal rows in CSV: {len(df)}')

    if not df.empty:
      sample_image_series = df.iloc[0]['image']
      sample_image = (
        sample_image_series.item()
        if hasattr(sample_image_series, 'item')
        else str(sample_image_series)
      )
      print(f'Sample image path: {sample_image}')

      full_path = os.path.join(frames_dir, sample_image)
      print(f'Full sample image path: {full_path}')
      print(f'Image exists: {os.path.exists(full_path)}')

      if not os.path.exists(full_path):
        print(f'{Fore.YELLOW}WARNING: Sample image does not exist at {full_path}{Style.RESET_ALL}')
        print(
          f'{Fore.YELLOW}Please ensure all images are available in the frames directory:{Style.RESET_ALL}'
        )
        print(f'  - Extract frames: tar -xf {os.path.join(data_dir, "frames.tar")} -C {data_dir}/')
  except Exception as e:
    print(f'{Fore.RED}Error reading CSV file: {e}{Style.RESET_ALL}')

  if args.model == 'claude':
    api_key = args.api_key or os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
      print(
        f'{Fore.RED}ERROR: Anthropic API key not provided. Please set the ANTHROPIC_API_KEY environment variable or use --api-key.{Style.RESET_ALL}'
      )
      return
    thinking_budget = args.thinking_budget if args.thinking_budget > 0 else None
    api_client = get_claude_api_client(
      api_key=api_key,
      model=args.claude_model,
      thinking_budget=thinking_budget,
      tool_version=args.tool_version,
    )
  elif args.model == 'qwen':
    api_key = args.api_key or os.environ.get('DASHSCOPE_API_KEY')
    if not api_key:
      print(
        f'{Fore.RED}ERROR: DashScope API key not provided. Please set the DASHSCOPE_API_KEY environment variable or use --api-key.{Style.RESET_ALL}'
      )
      return

    api_client = get_qwen_api_client(
      api_key=api_key,
      model=args.qwen_model,
      max_tokens=args.max_tokens,
    )

    if args.num_workers > 1:
      print(
        f'{Fore.YELLOW}Warning: Using multiple workers ({args.num_workers}) with Qwen. If you encounter errors, try reducing the number of workers.{Style.RESET_ALL}'
      )
  elif args.model == 'openai':
    api_key = args.api_key or os.environ.get('OPENAI_API_KEY')
    if not api_key:
      print(
        f'{Fore.RED}ERROR: OpenAI API key not provided. Please set the OPENAI_API_KEY environment variable or use --api-key.{Style.RESET_ALL}'
      )
      return

    api_client = get_openai_api_client(
      api_key=api_key,
      model=args.openai_model,
      max_tokens=args.max_tokens,
      reasoning_effort=args.reasoning_effort,
    )

    if args.num_workers > 1:
      print(
        f'{Fore.YELLOW}Warning: Using multiple workers ({args.num_workers}) with OpenAI. If you encounter errors, try reducing the number of workers.{Style.RESET_ALL}'
      )
  elif args.model == 'openai-cua':
    api_key = args.api_key or os.environ.get('OPENAI_API_KEY')
    if not api_key:
      print(
        f'{Fore.RED}ERROR: OpenAI API key not provided. Please set the OPENAI_API_KEY environment variable or use --api-key.{Style.RESET_ALL}'
      )
      return

    api_client = get_openai_cua_api_client(
      api_key=api_key,
      model=args.openai_cua_model,
      max_tokens=args.max_tokens,
      environment=args.environment,
    )

    if args.num_workers > 1:
      print(
        f'{Fore.YELLOW}Warning: Using multiple workers ({args.num_workers}) with OpenAI CUA. If you encounter errors, try reducing the number of workers.{Style.RESET_ALL}'
      )
  elif args.model == 'gemini':
    api_key = args.api_key or os.environ.get('GEMINI_API_KEY')
    if not api_key:
      print(
        f'{Fore.RED}ERROR: Google API key not provided. Please set the GEMINI_API_KEY environment variable or use --api-key.{Style.RESET_ALL}'
      )
      return

    api_client = get_gemini_api_client(
      api_key=api_key,
      model=args.gemini_model,
      max_tokens=args.max_tokens,
    )

    if args.num_workers > 1:
      print(
        f'{Fore.YELLOW}Warning: Using multiple workers ({args.num_workers}) with Gemini. If you encounter errors, try reducing the number of workers.{Style.RESET_ALL}'
      )
  elif args.model == 'molmo':
    if not args.api_url:
      print(
        f'{Fore.RED}ERROR: Molmo API URL not provided. Please provide it using --api-url.{Style.RESET_ALL}'
      )
      return

    api_client = get_molmo_api_client(
      api_url=args.api_url,
      api_key=args.api_key,
      max_tokens=args.max_tokens,
      temperature=args.temperature,
      top_p=args.top_p,
      top_k=args.top_k,
    )

    if args.num_workers > 1:
      print(
        f'{Fore.YELLOW}Warning: Using multiple workers ({args.num_workers}) with Molmo. If you encounter errors, try reducing the number of workers.{Style.RESET_ALL}'
      )
  elif args.model == 'ui-tars':
    if not args.api_url:
      print(
        f'{Fore.RED}ERROR: UI-TARS API URL not provided. Please provide it using --api-url.{Style.RESET_ALL}'
      )
      return

    api_client = get_ui_tars_api_client(
      api_url=args.api_url,
      api_key=args.api_key or 'super-secret-key',
      max_tokens=args.max_tokens,
      temperature=args.temperature,
      frequency_penalty=args.frequency_penalty,
      model_name=args.ui_tars_model,
    )

    if args.num_workers > 1:
      print(
        f'{Fore.YELLOW}Warning: Using multiple workers ({args.num_workers}) with UI-TARS. If you encounter errors, try reducing the number of workers.{Style.RESET_ALL}'
      )
  elif args.model == 'omniparser':
    api_client = get_omniparser_api_client(
      api_endpoint=args.api_url or 'https://omniparser-api-omniparser-api.modal.run',
      model=args.omniparser_model,
      temperature=args.omniparser_temperature,
    )

    if args.num_workers > 1:
      print(
        f'{Fore.YELLOW}Warning: Using multiple workers ({args.num_workers}) with OmniParser. If you encounter errors, try reducing the number of workers.{Style.RESET_ALL}'
      )
  else:
    api_key = args.api_key or os.environ.get('GENERALAGENTS_API_KEY')
    if not api_key:
      print(
        f'{Fore.RED}ERROR: General Agents API key not provided. Please set the GENERALAGENTS_API_KEY environment variable or use --api-key.{Style.RESET_ALL}'
      )
      return

    api_client = ace.get_api_client(api_key, args.ace_model)

  try:
    results = evaluate_csv(
      csv_file,
      frames_dir,
      api_client,
      args.output_file,
      args.sample_size,
      args.num_workers,
      run_id,
    )

    if results:
      print(f'\n{Fore.GREEN}Evaluation completed successfully!{Style.RESET_ALL}')
      print(f'Results: {len(results)} items processed')
      print(f'Results saved to: {args.output_file}')
      print(f'Visualizations saved to: {os.path.join(run_results_dir, "visualizations")}')

      results_analysis = analyze_results(results, run_id)

      metrics_dict = {
        'run_id': run_id,
        'model': args.model,
        'ci': results_analysis.ci,
        'accuracy': results_analysis.accuracy,
        'accuracy_ci_low': results_analysis.accuracy_ci_low,
        'accuracy_ci_high': results_analysis.accuracy_ci_high,
        'total_processed': results_analysis.total_processed,
      }

      base_dir = os.path.dirname(os.path.abspath(__file__))
      metrics_file = os.path.join(base_dir, 'results', 'all_metrics.csv')

      if os.path.exists(metrics_file):
        all_metrics_df = pd.read_csv(metrics_file)
        all_metrics_df = pd.concat(
          [all_metrics_df, pd.DataFrame([metrics_dict])], ignore_index=True
        )
      else:
        all_metrics_df = pd.DataFrame([metrics_dict])

      all_metrics_df.to_csv(metrics_file, index=False)
      print(f'\n{Fore.CYAN}Metrics saved to: {metrics_file}{Style.RESET_ALL}')

    else:
      print(f'\n{Fore.RED}Evaluation failed: No results returned.{Style.RESET_ALL}')

  except Exception as e:
    print(f'\n{Fore.RED}Evaluation failed with error: {e}{Style.RESET_ALL}')
    raise e


if __name__ == '__main__':
  main()
