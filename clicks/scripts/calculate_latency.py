import glob
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class LatencyMetrics:
  mean_latency: float
  ci: float
  latency_ci_low: Optional[float]
  latency_ci_high: Optional[float]


def calculate_latency_metrics(latencies: List[float], ci: float = 0.95) -> LatencyMetrics:
  if not latencies:
    return LatencyMetrics(
      mean_latency=0,
      ci=ci,
      latency_ci_low=None,
      latency_ci_high=None,
    )

  latencies = sorted(latencies)
  # Drop the last ten latency values to remove potential outliers or anomalies
  latencies = latencies[:-10]

  mean_latency = np.mean(latencies)

  def calculate_mean(data):
    return np.mean(data)

  latency_ci = None
  try:
    latencies_array = np.array(latencies)
    latency_bootstrap = stats.bootstrap(
      (latencies_array,),
      calculate_mean,
      confidence_level=ci,
      method='percentile',
    )
    latency_ci = latency_bootstrap.confidence_interval
  except Exception as e:
    print(f'Error calculating latency confidence interval: {e}')

  return LatencyMetrics(
    mean_latency=float(mean_latency),
    ci=float(ci),
    latency_ci_low=float(latency_ci.low) if latency_ci else None,
    latency_ci_high=float(latency_ci.high) if latency_ci else None,
  )


def read_and_analyze_results(directory: str = '.') -> Dict[str, LatencyMetrics]:
  csv_files = glob.glob(os.path.join(directory, '*.csv'))

  if not csv_files:
    print(f'No CSV files found in {directory}')
    return {}

  results = {}

  for csv_file in csv_files:
    model_name = os.path.basename(csv_file).replace('.csv', '')
    try:
      df = pd.read_csv(csv_file)

      if 'latency_seconds' in df.columns:
        latencies = df['latency_seconds'].dropna().tolist()
        metrics = calculate_latency_metrics(latencies)
        results[model_name] = metrics
      else:
        print(f'No latency column found in {csv_file}')
    except Exception as e:
      print(f'Error processing {csv_file}: {e}')

  return results


def print_latency_summary(results: Dict[str, LatencyMetrics]) -> None:
  if not results:
    print('No results to display')
    return

  data = []
  for model_name, metrics in results.items():
    ci_str = (
      f'[{metrics.latency_ci_low:.2f}, {metrics.latency_ci_high:.2f}]'
      if metrics.latency_ci_low is not None
      else 'N/A'
    )
    data.append(
      {
        'Model': model_name,
        'Mean (s)': f'{metrics.mean_latency:.2f}',
        '95% CI': ci_str,
      }
    )

  df = pd.DataFrame(data)
  print('\nLatency Summary:')
  print('-' * 80)
  print(df.to_string(index=False))
  # calculated from a separate script
  print('operator | 3.88178 | [3.77679, 3.98342]')


current_dir = os.path.join(
  os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'showdown-clicks-dev'
)

results = read_and_analyze_results(current_dir)

print_latency_summary(results)

# Save results to CSV
output_data = []
for model_name, metrics in results.items():
  output_data.append(
    {
      'model': model_name,
      'mean_latency': metrics.mean_latency,
      'ci_low': metrics.latency_ci_low,
      'ci_high': metrics.latency_ci_high,
    }
  )

output_dir = os.path.join(current_dir, '..', 'report')
os.makedirs(output_dir, exist_ok=True)
output_df = pd.DataFrame(output_data)
output_path = os.path.join(output_dir, 'latency_results.csv')
output_df.to_csv(output_path, index=False)
print(f'\nResults saved to: {output_path}')
