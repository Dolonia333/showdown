import argparse
import json
import os
from datetime import datetime
from typing import Dict, Any, List

import pandas as pd
from tqdm import tqdm

from clicks.third_party.dolphin import get_dolphin_api_client
from clicks.evaluate.utils import analyze_results
from clicks.base_models import EvaluationResult

def load_data(data_path: str) -> pd.DataFrame:
    """Load the evaluation dataset"""
    return pd.read_csv(data_path)

def save_results(results: List[EvaluationResult], output_file: str):
    """Save evaluation results to a JSON file"""
    output = []
    for result in results:
        output.append({
            "id": result.id,
            "instruction": result.instruction,
            "image_path": result.image_path,
            "gt_x1": result.gt_x1,
            "gt_y1": result.gt_y1,
            "gt_x2": result.gt_x2,
            "gt_y2": result.gt_y2,
            "width": result.width,
            "height": result.height,
            "prediction": {
                "x": result.prediction.x,
                "y": result.prediction.y,
                "raw_response": result.prediction.raw_response
            } if result.prediction else None,
            "error": result.error,
            "latency_seconds": result.latency_seconds,
            "is_in_bbox": result.is_in_bbox,
            "visualization_path": result.visualization_path
        })

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Evaluate Dolphin Mixtral model')
    parser.add_argument('--data-path', type=str, default='data/showdown-clicks-dev/data.csv',
                      help='Path to evaluation data CSV')
    parser.add_argument('--frames-dir', type=str, default='data/showdown-clicks-dev/frames',
                      help='Directory containing image frames')
    parser.add_argument('--output-dir', type=str, default='results',
                      help='Directory to save results')
    parser.add_argument('--sample-size', type=int, default=None,
                      help='Number of samples to evaluate (default: all)')
    
    args = parser.parse_args()

    # Create results directory
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M')
    run_id = f"{timestamp}-dolphin"
    results_dir = os.path.join(args.output_dir, run_id)
    os.makedirs(results_dir, exist_ok=True)

    # Load data
    df = load_data(args.data_path)
    if args.sample_size:
        df = df.sample(n=args.sample_size, random_state=42)

    # Initialize model
    client = get_dolphin_api_client()

    # Run evaluation
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        item = row.to_dict()        # Fix image path resolution by using just the basename of the image
        image_name = os.path.basename(item['image'].replace('frames/', ''))
        item['image'] = image_name
        try:
            result = client.process_single_item(item, args.frames_dir, run_id)
            results.append(result)
        except Exception as e:
            print(f"Error processing item {item['id']}: {str(e)}")
            continue

    # Save results
    output_file = os.path.join(results_dir, 'results.json')
    save_results(results, output_file)

    # Analyze results
    analyze_results(results, os.path.join(results_dir, 'analysis.txt'))

if __name__ == '__main__':
    main()
