#!/usr/bin/env python3
"""
Create bar plots comparing inconsistency scores for different ranking methods across models.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Dict, List

def load_metrics_data() -> Dict[str, Dict[str, float]]:
    """Load inconsistency scores from metrics files for each model's first run."""
    
    # Define the metrics files for each model's first run
    metrics_files = {
        'claude-3-5-sonnet': 'logs/results/anthropic_claude-3-5-sonnet-latest_split80_10_20250727_203846_1753641526_df254137_20250728_133701_1753702621_229ee3db/anthropic_claude-3-5-sonnet-latest_split80_10_20250727_203846_1753641526_df254137_metrics_20250728_133701_1753702621_229ee3db.csv',
        'gpt-4o-mini': 'logs/results/openai_gpt-4o-mini_split80_10_20250727_200027_1753639227_eef9f7bd_20250728_133701_1753702621_229ee3db/openai_gpt-4o-mini_split80_10_20250727_200027_1753639227_eef9f7bd_metrics_20250728_133701_1753702621_229ee3db.csv',
        'gemma-3n-E4B-it': 'logs/results/together_google_gemma-3n-E4B-it_split80_10_20250727_224331_1753649011_d4eadfab_20250728_133702_1753702622_0d947102/together_google_gemma-3n-E4B-it_split80_10_20250727_224331_1753649011_d4eadfab_metrics_20250728_133702_1753702622_0d947102.csv',
        'mistral-7B-instruct-v0.3': 'logs/results/together_mistralai_Mistral-7B-Instruct-v0.3_split80_10_20250727_225419_1753649659_73b450ff_20250728_133700_1753702620_3af65586/together_mistralai_Mistral-7B-Instruct-v0.3_split80_10_20250727_225419_1753649659_73b450ff_metrics_20250728_133700_1753702620_3af65586.csv',
        'deepseek-r1-distill-qwen-1.5B': 'logs/results/together_deepseek-ai_DeepSeek-R1-Distill-Qwen-1.5B_split80_10_20250728_070502_1753679102_27daab70_20250728_133659_1753702619_9f73bac6/together_deepseek-ai_DeepSeek-R1-Distill-Qwen-1.5B_split80_10_20250728_070502_1753679102_27daab70_metrics_20250728_133659_1753702619_9f73bac6.csv'
    }
    
    data = {}
    
    for model, filepath in metrics_files.items():
        if os.path.exists(filepath):
            df = pd.read_csv(filepath, index_col=0)
            data[model] = {
                'Elo': df.loc['Elo', 'inconsistency_score'],
                'Glicko2': df.loc['Glicko2', 'inconsistency_score'],
                'WinCount': df.loc['WinCount', 'inconsistency_score']
            }
        else:
            print(f"Warning: Metrics file not found for {model}: {filepath}")
    
    return data

def load_simulated_annealing_data() -> Dict[str, float]:
    """Load simulated annealing inconsistency scores from optimized rankings."""
    
    # Define the metadata files for simulated annealing results
    sa_files = {
        'claude-3-5-sonnet': 'logs/optimized_rankings/optimal_ranking_metadata_claude-3-5-sonnet_20250802_173351.txt',
        'gpt-4o-mini': 'logs/optimized_rankings/optimal_ranking_metadata_gpt-4o-mini_20250802_173635.txt',
        'gemma-3n-E4B-it': 'logs/optimized_rankings/optimal_ranking_metadata_gemma-3n-E4B-it_20250802_173647.txt',
        'mistral-7B-instruct-v0.3': 'logs/optimized_rankings/optimal_ranking_metadata_mistral-7B-instruct-v0.3_20250802_173700.txt',
        'deepseek-r1-distill-qwen-1.5B': 'logs/optimized_rankings/optimal_ranking_metadata_deepseek-r1-distill-qwen-1.5B_20250802_173706.txt'
    }
    
    data = {}
    
    for model, filepath in sa_files.items():
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                for line in f:
                    if line.startswith('inconsistency_score:'):
                        score = float(line.split(':')[1].strip())
                        data[model] = score
                        break
        else:
            print(f"Warning: Simulated annealing file not found for {model}: {filepath}")
    
    return data

def create_comparison_plot(metrics_data: Dict[str, Dict[str, float]], 
                          sa_data: Dict[str, float],
                          output_dir: str = "logs/plots"):
    """Create bar plots comparing inconsistency scores across models and methods."""
    
    # Prepare data for plotting
    models = list(metrics_data.keys())
    methods = ['Elo', 'Glicko2', 'WinCount', 'Simulated Annealing']
    
    # Create data matrix
    data_matrix = []
    for model in models:
        row = [
            metrics_data[model]['Elo'],
            metrics_data[model]['Glicko2'], 
            metrics_data[model]['WinCount'],
            sa_data.get(model, np.nan)  # Use NaN if SA data not available
        ]
        data_matrix.append(row)
    
    data_matrix = np.array(data_matrix)
    
    # Set up colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # Plot 1: Bar plot by model
    plt.figure(figsize=(14, 8))
    x = np.arange(len(models))
    width = 0.2
    multiplier = 0
    
    for i, method in enumerate(methods):
        offset = width * multiplier
        values = data_matrix[:, i]
        # Filter out NaN values
        valid_mask = ~np.isnan(values)
        if np.any(valid_mask):
            rects = plt.bar(x[valid_mask] + offset, values[valid_mask], width, 
                           label=method, color=colors[i], alpha=0.8)
            # Add value labels on bars
            for rect, val in zip(rects, values[valid_mask]):
                height = rect.get_height()
                plt.text(rect.get_x() + rect.get_width()/2., height + 0.005,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        multiplier += 1
    
    plt.xlabel('Model')
    plt.ylabel('Inconsistency Score')
    plt.title('Inconsistency Scores by Model and Ranking Method')
    plt.xticks(x + width * 1.5, models, rotation=45, ha='right')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 0.6)
    plt.tight_layout()
    
    # Save first plot
    plot1_filename = "inconsistency_scores_by_model.png"
    plot1_path = os.path.join(output_dir, plot1_filename)
    plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Bar plot by method
    plt.figure(figsize=(12, 8))
    x = np.arange(len(methods))
    width = 0.15
    multiplier = 0
    
    # Create shorter model names for better display
    model_short_names = {
        'claude-3-5-sonnet': 'Claude 3.5',
        'gpt-4o-mini': 'GPT-4o-mini',
        'gemma-3n-E4B-it': 'Gemma-3n',
        'mistral-7B-instruct-v0.3': 'Mistral-7B',
        'deepseek-r1-distill-qwen-1.5B': 'DeepSeek-R1'
    }
    
    for i, model in enumerate(models):
        offset = width * multiplier
        values = data_matrix[i, :]
        # Filter out NaN values
        valid_mask = ~np.isnan(values)
        if np.any(valid_mask):
            rects = plt.bar(x[valid_mask] + offset, values[valid_mask], width,
                           label=model_short_names[model], alpha=0.8)
            # Add value labels on bars
            for rect, val in zip(rects, values[valid_mask]):
                height = rect.get_height()
                plt.text(rect.get_x() + rect.get_width()/2., height + 0.005,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        multiplier += 1
    
    plt.xlabel('Ranking Method')
    plt.ylabel('Inconsistency Score')
    plt.title('Inconsistency Scores by Ranking Method and Model')
    plt.xticks(x + width * 2, methods)
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 0.6)
    plt.tight_layout()
    
    # Save second plot
    plot2_filename = "inconsistency_scores_by_method.png"
    plot2_path = os.path.join(output_dir, plot2_filename)
    plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot 1 (by model) saved: {plot1_path}")
    print(f"Plot 2 (by method) saved: {plot2_path}")
    
    # Also create a summary table
    summary_df = pd.DataFrame(data_matrix, index=models, columns=methods)
    summary_df = summary_df.round(4)
    
    print("\nInconsistency Scores Summary:")
    print("=" * 80)
    print(summary_df.to_string())
    
    # Save summary table
    summary_filename = "inconsistency_scores_summary.csv"
    summary_path = os.path.join(output_dir, summary_filename)
    summary_df.to_csv(summary_path)
    print(f"\nSummary table saved: {summary_path}")
    
    return plot1_path, plot2_path, summary_path

def main():
    """Main function to create the comparison plots."""
    
    print("Loading metrics data...")
    metrics_data = load_metrics_data()
    
    print("Loading simulated annealing data...")
    sa_data = load_simulated_annealing_data()
    
    print("Creating comparison plots...")
    plot1_path, plot2_path, summary_path = create_comparison_plot(metrics_data, sa_data)
    
    print(f"\n🎉 Comparison analysis completed!")
    print(f"Plot 1: {plot1_path}")
    print(f"Plot 2: {plot2_path}")
    print(f"Summary: {summary_path}")

if __name__ == "__main__":
    main() 