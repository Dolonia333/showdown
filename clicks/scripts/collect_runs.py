import glob
import os

import pandas as pd
from clicks.evaluate.utils import analyze_results


def summarize_all_results():
  base_dir = os.path.dirname(os.path.abspath(__file__))
  results_dir = os.path.join(base_dir, '..', 'results', 'showdown-clicks-dev')

  output_file = os.path.join(results_dir, '..', 'report', 'metrics.csv')

  run_dirs = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]

  all_metrics = []

  print(f'Found {len(run_dirs)} result directories to process.')

  result_files = glob.glob(os.path.join(results_dir, '*.csv'))

  for result_file in result_files:
    try:
      print(f'Processing {result_file}...')

      # Extract model name from filename
      filename = os.path.basename(result_file)

      # Skip if this is a summary file
      if filename.startswith('summary_') or 'all_metrics' in filename:
        continue

      # Parse model name from filename
      name = filename.split('.csv')[0]

      # Read the results
      results_df = pd.read_csv(result_file)

      # Convert DataFrame to list of dictionaries for analysis
      # Ensure all keys are strings to match analyze_results parameter type
      results = [{str(k): v for k, v in item.items()} for item in results_df.to_dict('records')]

      if not results:
        print(f'No results found in {result_file}, skipping.')
        continue

      # Apply the same analysis logic
      results_analysis = analyze_results(results)

      # Create metrics dictionary
      metrics_dict = {
        'model': name,
        'ci': results_analysis.ci,
        'accuracy': results_analysis.accuracy,
        'total_correct': results_analysis.total_correct,
        'total_processed': results_analysis.total_processed,
        'accuracy_ci_low': results_analysis.accuracy_ci_low,
        'accuracy_ci_high': results_analysis.accuracy_ci_high,
        'result_file': result_file,
      }

      all_metrics.append(metrics_dict)
      print(f'Added metrics for {result_file}')

    except Exception as e:
      print(f'Error processing {result_file}: {e}')

  if all_metrics:
    # Create DataFrame and save to CSV
    all_metrics_df = pd.DataFrame(all_metrics)
    all_metrics_df.to_csv(output_file, index=False)
    print(f'All metrics saved to: {output_file}')
  else:
    print('No metrics were generated.')


if __name__ == '__main__':
  summarize_all_results()
