#!/usr/bin/env python3
"""
Consolidated variance analysis tools for moral preferences evaluation.

This module combines functionality from multiple variance analysis scripts:
- create_variance_plot.py
- create_prompts_variance_plots.py
- create_runs_variance_plots.py
- create_prompts_variance_table.py
- create_runs_variance_table.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import glob
import os
from typing import Optional, Dict, Any

# Set seaborn style for better-looking plots
sns.set_style("whitegrid")
sns.set_palette("husl")

def create_variance_table(variances_dir: str, output_dir: str = None) -> pd.DataFrame:
    """
    Create a variance metrics table from CSV files.
    
    Args:
        variances_dir: Directory containing variance CSV files
        output_dir: Directory to save output (defaults to variances_dir)
        
    Returns:
        DataFrame with variance metrics
    """
    if output_dir is None:
        output_dir = variances_dir
    
    # Find all variance CSV files
    csv_files = glob.glob(os.path.join(variances_dir, 'ranking_variance_*.csv'))
    
    if not csv_files:
        print(f"No variance CSV files found in {variances_dir}")
        return pd.DataFrame()
    
    # Read and combine all variance results
    all_data = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            all_data.append(df)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
    
    if not all_data:
        print("No valid variance data found")
        return pd.DataFrame()
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Create summary table
    summary_data = []
    
    # Check if we have the expected columns or the actual columns
    if 'Model' in combined_df.columns and 'Method' in combined_df.columns:
        # New format with Model and Method columns
        for (model, method), group in combined_df.groupby(['Model', 'Method']):
            summary_data.append({
                'Model': model,
                'Method': method,
                'Total Variation Distance': group['Total Variation Distance'].mean(),
                'Jensen-Shannon Divergence': group['Jensen-Shannon Divergence'].mean(),
                'Rank Edit Distance': group['Rank Edit Distance'].mean(),
                'Count': len(group)
            })
    elif 'method' in combined_df.columns:
        # Old format with method column
        for method, group in combined_df.groupby('method'):
            summary_data.append({
                'Model': 'Unknown',  # Model info not available in this format
                'Method': method,
                'Total Variation Distance': group['tvd'].mean(),
                'Jensen-Shannon Divergence': group['jsd'].mean(),
                'Rank Edit Distance': group['rank_edit_distance'].mean(),
                'Count': len(group)
            })
    else:
        print("Warning: Unknown variance data format")
        return pd.DataFrame()
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save summary table
    output_file = os.path.join(output_dir, 'variance_metrics_table.csv')
    summary_df.to_csv(output_file, index=False)
    print(f"Variance summary table saved to {output_file}")
    
    return summary_df

def create_variance_plots(variances_dir: str, output_dir: str = None) -> None:
    """
    Create variance comparison plots.
    
    Args:
        variances_dir: Directory containing variance data
        output_dir: Directory to save plots (defaults to variances_dir)
    """
    if output_dir is None:
        output_dir = variances_dir
    
    # Read the variance metrics data
    table_file = os.path.join(variances_dir, 'variance_metrics_table.csv')
    if not os.path.exists(table_file):
        print(f"Variance metrics table not found: {table_file}")
        return
    
    df = pd.read_csv(table_file)
    
    # Create a more visually appealing color palette
    colors = sns.color_palette("Set2", 3)
    method_colors = {'Elo': colors[0], 'Glicko2': colors[1], 'WinCount': colors[2]}
    
    metrics = [
        ('Total Variation Distance', 'tvd_plot.png'),
        ('Jensen-Shannon Divergence', 'jsd_plot.png'),
        ('Rank Edit Distance', 'rank_edit_distance_plot.png')
    ]
    
    for metric, filename in metrics:
        plt.figure(figsize=(14, 8))
        
        # Handle different column name formats
        method_col = 'Method' if 'Method' in df.columns else 'method'
        model_col = 'Model' if 'Model' in df.columns else 'model'
        
        if method_col in df.columns:
            for method in df[method_col].unique():
                method_data = df[df[method_col] == method]
                
                # Use appropriate column for the metric
                if metric == 'Total Variation Distance':
                    metric_col = 'Total Variation Distance' if 'Total Variation Distance' in df.columns else 'tvd'
                elif metric == 'Jensen-Shannon Divergence':
                    metric_col = 'Jensen-Shannon Divergence' if 'Jensen-Shannon Divergence' in df.columns else 'jsd'
                elif metric == 'Rank Edit Distance':
                    metric_col = 'Rank Edit Distance' if 'Rank Edit Distance' in df.columns else 'rank_edit_distance'
                else:
                    metric_col = metric
                
                if metric_col in method_data.columns:
                    x_values = method_data[model_col] if model_col in method_data.columns else range(len(method_data))
                    y_values = method_data[metric_col]
                    
                    bars = plt.bar(x_values, y_values, 
                                  label=method, color=method_colors.get(method, 'gray'), alpha=0.8, 
                                  edgecolor='white', linewidth=1.5)
                    
                    # Add value labels on bars
                    for bar in bars:
                        height = bar.get_height()
                        plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                                f'{height:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.title(f'{metric} Across Various Runs', fontsize=16, fontweight='bold', pad=20)
        plt.ylabel(metric, fontsize=14, fontweight='bold')
        plt.xlabel('Model', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(title='Ranking Method', title_fontsize=12, fontsize=11, 
                  frameon=True, fancybox=True, shadow=True)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        output_file = os.path.join(output_dir, filename)
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Saved: {output_file}")

def run_variance_analysis(variances_dir: str, output_dir: str = None) -> Dict[str, Any]:
    """
    Run complete variance analysis.
    
    Args:
        variances_dir: Directory containing variance CSV files
        output_dir: Directory to save results (defaults to variances_dir)
        
    Returns:
        Dictionary with analysis results
    """
    if output_dir is None:
        output_dir = variances_dir
    
    print(f"Running variance analysis on {variances_dir}")
    
    # Create variance table
    table_df = create_variance_table(variances_dir, output_dir)
    
    if table_df.empty:
        print("No variance data available")
        return {}
    
    # Create variance plots
    create_variance_plots(variances_dir, output_dir)
    
    # Summary statistics
    print("\nSummary Statistics:")
    print("=" * 50)
    for metric in ['Total Variation Distance', 'Jensen-Shannon Divergence', 'Rank Edit Distance']:
        if metric in table_df.columns:
            print(f"\n{metric}:")
            print(f"  Min: {table_df[metric].min():.4f}")
            print(f"  Max: {table_df[metric].max():.4f}")
            print(f"  Mean: {table_df[metric].mean():.4f}")
            print(f"  Std: {table_df[metric].std():.4f}")
    
    return {
        'table': table_df,
        'output_dir': output_dir
    }

def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Variance analysis for moral preferences')
    parser.add_argument('--variances-dir', default='logs/variances_over_runs',
                       help='Directory containing variance CSV files')
    parser.add_argument('--output-dir', 
                       help='Directory to save results (defaults to variances-dir)')
    
    args = parser.parse_args()
    
    run_variance_analysis(args.variances_dir, args.output_dir)

if __name__ == "__main__":
    main() 