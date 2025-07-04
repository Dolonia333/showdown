import argparse
import multiprocessing
import os
from datetime import datetime
from functools import partial
from typing import Any, Dict, List, Optional

import pandas as pd
from clicks.api_client_base import AbstractAPIClient
from clicks.base_models import EvaluationResult, ParsedPrediction
from clicks.evaluate.utils import analyze_results
from clicks.third_party import (
  get_openrouter_api_client,  # OpenRouter for API-based models
  get_dolphin_api_client,    # Local Dolphin Mixtral
  get_deepseek_api_client,   # Local DeepSeek
)
from colorama import Fore, Style, init
from tqdm import tqdm
from dataclasses import asdict

init(autoreset=True)


def process_item(
  item: Dict[str, Any],
  frames_dir: str,
  api_client: AbstractAPIClient,
  run_id: str,
) -> EvaluationResult:
  """Process a single evaluation item.
  
  Args:
      item: Dictionary containing item data
      frames_dir: Directory containing screenshot frames
      api_client: Client for model API
      run_id: Unique identifier for this evaluation run
      
  Returns:
      EvaluationResult object containing predictions and metrics
  """
  try:
    # Create base result object
    result = EvaluationResult(
      id=str(item['id']),
      instruction=str(item['instruction']),
      image_path=str(item['image']),
      gt_x1=int(item['x1']),
      gt_y1=int(item['y1']),
      gt_x2=int(item['x2']),
      gt_y2=int(item['y2']),
      width=int(item['width']),
      height=int(item['height']),
      latency_seconds=0.0
    )
    
    # Validate image path
    if not result.image_path or result.image_path.startswith('data:'):
      raise ValueError(f'Invalid image path: {result.image_path}')
    
    # Process with model
    try:
      model_result = api_client.process_single_item(item, frames_dir, run_id)
      if model_result:
        # Copy fields from model result that exist
        for field in ['prediction', 'error', 'latency_seconds', 'is_in_bbox']:
          if hasattr(model_result, field):
            setattr(result, field, getattr(model_result, field))
    except Exception as e:
      result.error = str(e)
      result.latency_seconds = 0.0
      result.is_in_bbox = False
    
    return result
  except Exception as e:
    print(f'Error processing item {item.get("id", "unknown")}: {str(e)}')
    raise e


def evaluate_csv(
  csv_file: str,
  frames_dir: str,
  api_client: AbstractAPIClient,
  output_file: Optional[str] = None,
  sample_size: Optional[int] = None,
  num_workers: int = 1,
  run_id: str = datetime.now().strftime('%Y-%m-%d-%H-%M'),
) -> List[EvaluationResult]:
  """Evaluate a CSV file containing click locations.
  
  Args:
      csv_file: Path to CSV file with click data
      frames_dir: Directory containing screenshot frames
      api_client: Client for model API
      output_file: Optional path to save results CSV
      sample_size: Optional number of samples to evaluate
      num_workers: Number of concurrent workers
      run_id: Unique identifier for this evaluation run
      
  Returns:
      List of EvaluationResult objects
  """
  # Read and prepare input data
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

  results: List[EvaluationResult] = []
  total_processed = 0
  total_in_bbox = 0

  # Process items
  try:
    with multiprocessing.Pool(processes=num_workers, maxtasksperchild=1) as pool:
      with tqdm(total=len(items), desc='Evaluating', unit='item') as pbar:
        for result in pool.imap_unordered(process_func, items):
          if result is not None:
            results.append(result)
            total_processed += 1
            if getattr(result, 'is_in_bbox', False):
              total_in_bbox += 1
            running_accuracy = (total_in_bbox / total_processed) * 100 if total_processed > 0 else 0
            pbar.set_postfix({'accuracy': f'{running_accuracy:.2f}%'})
          pbar.update(1)
  except Exception as e:
    print(f'{Fore.RED}Error in multiprocessing: {e}{Style.RESET_ALL}')
    raise e

  # Save results if requested
  if output_file and results:
    results_dicts = [asdict(result) for result in results]
    results_df = pd.DataFrame(results_dicts)
    results_df.to_csv(output_file, index=False)
    print(f'Results written to {output_file}')

  return results


def main():
  """Main entry point for evaluation script."""
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
      'openrouter',   # OpenRouter API models
      'dolphin',      # Local Dolphin Mixtral
      'deepseek',     # Local DeepSeek
    ],
    default='dolphin',
    help='Model to use for evaluation (openrouter for API models, dolphin/deepseek for local models)',
  )
  parser.add_argument(
    '--api-key',
    type=str,
    default=None,
    help='API key for OpenRouter (if using openrouter model)',
  )
  parser.add_argument(
    '--openrouter-model',
    type=str,
    default='anthropic/claude-3-opus',
    help='OpenRouter model to use (default: anthropic/claude-3-opus)',
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
    '--output-file', 
    type=str, 
    default=None, 
    help='Path to output CSV file (optional)'
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

  args = parser.parse_args()

  # Set up directories
  base_dir = os.path.dirname(os.path.abspath(__file__))
  data_dir = os.path.join(base_dir, 'data')
  results_dir = os.path.join(base_dir, 'results')

  os.makedirs(results_dir, exist_ok=True)

  # Configure run ID
  run_id = args.run_id or datetime.now().strftime('%Y-%m-%d-%H-%M')
  if args.model in ['dolphin', 'deepseek']:
    run_id = f'{run_id}-{args.model}'

  run_results_dir = os.path.join(results_dir, run_id)
  os.makedirs(run_results_dir, exist_ok=True)

  # Set up input paths
  if args.dataset == 'dev':
    csv_file = os.path.join(data_dir, 'showdown-clicks-dev/data.csv')
  else:
    raise ValueError('Full dataset not currently supported')

  frames_dir = os.path.join(data_dir, 'showdown-clicks-dev')

  # Configure output file
  if args.output_file is None:
    if args.model == 'openrouter':
      model_name = args.openrouter_model.replace('/', '_').replace('-', '_')
      args.output_file = os.path.join(
        run_results_dir, f'openrouter_results_{model_name}_{args.dataset}.csv'
      )
    else:
      args.output_file = os.path.join(run_results_dir, f'{args.model}_results_{args.dataset}.csv')

  # Print configuration
  print(f'{Fore.CYAN}Running evaluation with the following configuration:{Style.RESET_ALL}')
  print(f'{Fore.CYAN}  Dataset: {args.dataset}{Style.RESET_ALL}')
  print(f'{Fore.CYAN}  Model: {args.model}{Style.RESET_ALL}')
  print(f'{Fore.CYAN}  CSV file: {csv_file}{Style.RESET_ALL}')
  print(f'{Fore.CYAN}  Frames directory: {frames_dir}{Style.RESET_ALL}')
  print(f'{Fore.CYAN}  Run ID: {run_id}{Style.RESET_ALL}')
  print(f'{Fore.CYAN}  Results directory: {run_results_dir}{Style.RESET_ALL}')
  print(f'{Fore.CYAN}  Output file: {args.output_file}{Style.RESET_ALL}')
  print(f'{Fore.CYAN}  Concurrent workers: {args.num_workers}{Style.RESET_ALL}')

  if args.model == 'openrouter':
    print(f'  OpenRouter model: {args.openrouter_model}')
    print(f'  Max tokens: {args.max_tokens}')
  elif args.model == 'dolphin':
    print(f'  Using local Dolphin Mixtral model via Ollama')
  elif args.model == 'deepseek':
    print(f'  Using local DeepSeek model via Ollama')

  if args.sample_size:
    print(f'  Sample size: {args.sample_size}')

  # Validate input files exist
  if not os.path.exists(csv_file):
    print(f'{Fore.RED}ERROR: CSV file not found: {csv_file}{Style.RESET_ALL}')
    return

  if not os.path.exists(frames_dir):
    print(f'{Fore.RED}ERROR: Frames directory not found: {frames_dir}{Style.RESET_ALL}')
    return

  # Check sample data
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
  
  # Initialize API client
  try:
    if args.model == 'openrouter':
      api_key = args.api_key or os.environ.get('OPENROUTER_API_KEY')
      if not api_key:
        print(
          f'{Fore.RED}ERROR: OpenRouter API key not provided. Please set the OPENROUTER_API_KEY environment variable or use --api-key.{Style.RESET_ALL}'
        )
        return
      
      api_client = get_openrouter_api_client(
        api_key=api_key,
        model=args.openrouter_model,
        max_tokens=args.max_tokens
      )

      if args.num_workers > 1:
        print(
          f'{Fore.YELLOW}Warning: Using multiple workers ({args.num_workers}) with OpenRouter. If you encounter errors, try reducing the number of workers.{Style.RESET_ALL}'
        )
    elif args.model == 'dolphin':
      print(f'  Using local Dolphin Mixtral model via Ollama')
      api_client = get_dolphin_api_client()
      
      if args.num_workers > 1:
        print(f'{Fore.YELLOW}Warning: Using multiple workers with local Dolphin model.{Style.RESET_ALL}')
        
    elif args.model == 'deepseek':
      print(f'  Using local DeepSeek model via Ollama')
      api_client = get_deepseek_api_client()
      
      if args.num_workers > 1:
        print(f'{Fore.YELLOW}Warning: Using multiple workers with local DeepSeek model.{Style.RESET_ALL}')
  except Exception as e:
    print(f'{Fore.RED}Error initializing API client: {e}{Style.RESET_ALL}')
    return
  
  # Run evaluation
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

      # Analyze results
      results_analysis = analyze_results(results)

      # Save metrics
      metrics_dict = {
        'run_id': run_id,
        'model': args.model,
        'openrouter_model': args.openrouter_model if args.model == 'openrouter' else None,
        'ci': results_analysis.ci,
        'accuracy': results_analysis.accuracy,
        'accuracy_ci_low': results_analysis.accuracy_ci_low,
        'accuracy_ci_high': results_analysis.accuracy_ci_high,
        'total_processed': results_analysis.total_processed,
      }

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
