#!/usr/bin/env python3
"""
Plotting utilities for ranking variance metrics.
Creates plots for total variation distance, Jensen-Shannon divergence, and rank edit distance
by comparing pairs of runs for each model.
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import subprocess
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from plotting_utils import clean_model_name

RESULTS_DIR = "logs/results"
VARIANCES_DIR = "logs/variances"
PLOTS_DIR = "logs/plots"

def find_model_run_pairs() -> Dict[str, List[str]]:
    """
    Find pairs of runs for each model.
    
    Returns:
        Dictionary mapping model names to lists of run directories
    """
    model_runs = {}
    
    # Get all result directories
    result_dirs = glob.glob(os.path.join(RESULTS_DIR, "*"))
    
    for result_dir in result_dirs:
        if not os.path.isdir(result_dir):
            continue
            
        dir_name = os.path.basename(result_dir)
        
        # Extract model name (everything before the first timestamp)
        # Pattern: model_name_timestamp_hash
        parts = dir_name.split('_')
        if len(parts) >= 3:
            # Find the first part that looks like a timestamp (8 digits)
            timestamp_idx = None
            for i, part in enumerate(parts):
                if re.match(r'\d{8}', part):
                    timestamp_idx = i
                    break
            
            if timestamp_idx is not None:
                model_name = '_'.join(parts[:timestamp_idx])
                clean_model = clean_model_name(model_name)
                
                if clean_model not in model_runs:
                    model_runs[clean_model] = []
                model_runs[clean_model].append(dir_name)
    
    # Filter to only models with at least 2 runs
    model_runs = {k: v for k, v in model_runs.items() if len(v) >= 2}
    
    return model_runs

def run_variance_analysis(run1: str, run2: str) -> Optional[pd.DataFrame]:
    """
    Run the ranking variance analysis between two runs.
    
    Args:
        run1: First run directory name
        run2: Second run directory name
        
    Returns:
        DataFrame with variance metrics or None if failed
    """
    try:
        # Run the measure_ranking_variance script
        cmd = [
            'python', 'measure_ranking_variance.py',
            '--run1', run1,
            '--run2', run2
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
        
        if result.returncode != 0:
            print(f"Error running variance analysis: {result.stderr}")
            return None
        
        # Find the output CSV file
        short_id1 = run1[-12:]
        short_id2 = run2[-12:]
        csv_filename = f"ranking_variance_{short_id1}_vs_{short_id2}.csv"
        csv_path = os.path.join(VARIANCES_DIR, csv_filename)
        
        if os.path.exists(csv_path):
            return pd.read_csv(csv_path)
        else:
            print(f"Expected output file not found: {csv_path}")
            return None
            
    except Exception as e:
        print(f"Error in variance analysis: {e}")
        return None

def collect_variance_data() -> pd.DataFrame:
    """
    Collect variance data for all model pairs.
    
    Returns:
        DataFrame with variance metrics for all models
    """
    model_runs = find_model_run_pairs()
    all_data = []
    
    print(f"Found {len(model_runs)} models with multiple runs:")
    for model, runs in model_runs.items():
        print(f"  {model}: {len(runs)} runs")
    
    for model, runs in model_runs.items():
        # For each model, compare the first two runs
        if len(runs) >= 2:
            run1, run2 = runs[0], runs[1]
            print(f"\nAnalyzing {model}: {run1} vs {run2}")
            
            variance_data = run_variance_analysis(run1, run2)
            
            if variance_data is not None:
                # Add model name to the data
                variance_data['model'] = model
                all_data.append(variance_data)
            else:
                print(f"Failed to get variance data for {model}")
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    else:
        return pd.DataFrame()

def create_variance_plots(data: pd.DataFrame, output_dir: str = PLOTS_DIR) -> None:
    """
    Create plots for the three variance metrics.
    
    Args:
        data: DataFrame with variance metrics
        output_dir: Directory to save plots
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    if data.empty:
        print("No variance data available for plotting")
        return
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create separate plots for each metric
    metrics = ['tvd', 'jsd', 'rank_edit_distance']
    metric_names = ['Total Variation Distance', 'Jensen-Shannon Divergence', 'Rank Edit Distance']
    
    for metric, metric_name in zip(metrics, metric_names):
        if metric not in data.columns:
            print(f"Warning: {metric} not found in data")
            continue
            
        # Create pivot table for this metric
        pivot_data = data.pivot(index='model', columns='method', values=metric)
        
        # Only create plot if there's data
        if not pivot_data.empty and not pivot_data.isna().all().all():
            plt.figure(figsize=(8, 6))
            
            # Create heatmap
            sns.heatmap(pivot_data, annot=True, fmt='.4f', cmap='RdYlBu_r', 
                       cbar_kws={'label': metric_name})
            plt.title(f'{metric_name} Comparison')
            plt.xlabel('Method')
            plt.ylabel('Model')
            
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            
            # Save the plot
            plot_filename = f'{metric.replace("_", "_")}_comparison.png'
            plt.savefig(os.path.join(output_dir, plot_filename), 
                       bbox_inches='tight', dpi=300)
            plt.show()
            plt.close()
            
            print(f"✓ Created {plot_filename}")
        else:
            print(f"No data available for {metric_name} comparison")
    
    print(f"Variance comparison plots saved to {output_dir}")

def create_variance_table(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create a clean table of variance metrics.
    
    Args:
        data: DataFrame with variance metrics
        
    Returns:
        DataFrame formatted as a variance table
    """
    if data.empty:
        return pd.DataFrame()
    
    # Create the table format
    table_data = []
    
    for model in data['model'].unique():
        model_data = data[data['model'] == model]
        
        for i, (_, row) in enumerate(model_data.iterrows()):
            table_row = {
                'Model': model if i == 0 else '',  # Only show model name in first row
                'Method': row['method']
            }
            
            # Add metrics
            for metric in ['tvd', 'jsd', 'rank_edit_distance']:
                if metric in row and not pd.isna(row[metric]):
                    if metric == 'tvd':
                        table_row['Total Variation Distance'] = f"{row[metric]:.4f}"
                    elif metric == 'jsd':
                        table_row['Jensen-Shannon Divergence'] = f"{row[metric]:.4f}"
                    elif metric == 'rank_edit_distance':
                        table_row['Rank Edit Distance'] = f"{row[metric]:.0f}"
                else:
                    if metric == 'tvd':
                        table_row['Total Variation Distance'] = ''
                    elif metric == 'jsd':
                        table_row['Jensen-Shannon Divergence'] = ''
                    elif metric == 'rank_edit_distance':
                        table_row['Rank Edit Distance'] = ''
            
            table_data.append(table_row)
    
    return pd.DataFrame(table_data)

def main():
    """
    Main function to collect variance data and create plots.
    """
    print("="*80)
    print("RANKING VARIANCE ANALYSIS")
    print("="*80)
    
    # Collect variance data
    print("\n1. Collecting variance data...")
    data = collect_variance_data()
    
    if data.empty:
        print("No variance data collected. Please ensure you have multiple runs per model.")
        return
    
    print(f"\n✓ Collected variance data for {len(data['model'].unique())} models")
    print(f"✓ Available methods: {list(data['method'].unique())}")
    
    # Create variance table
    print("\n2. Creating variance table...")
    table_data = create_variance_table(data)
    if not table_data.empty:
        print("\n" + "="*80)
        print("VARIANCE METRICS TABLE")
        print("="*80)
        print(table_data.to_string(index=False))
        
        # Save table
        table_file = os.path.join(VARIANCES_DIR, "variance_metrics_table.csv")
        table_data.to_csv(table_file, index=False)
        print(f"\n✓ Table saved to: {table_file}")
    
    # Create variance plots
    print("\n3. Creating variance comparison plots...")
    create_variance_plots(data)
    
    print("\n" + "="*80)
    print("VARIANCE ANALYSIS COMPLETED")
    print("="*80)
    print("✓ Variance data collected for all model pairs")
    print("✓ Three separate plots created:")
    print("   - Total Variation Distance comparison")
    print("   - Jensen-Shannon Divergence comparison")
    print("   - Rank Edit Distance comparison")
    print("✓ All plots saved to logs/plots/")

if __name__ == "__main__":
    main() 