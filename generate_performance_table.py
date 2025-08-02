#!/usr/bin/env python3
"""
Generate a clean performance table for different models and ranking methods.
This script creates tables in the format shown in the image.
"""

import pandas as pd
from plotting_utils import load_results_data, create_performance_table
import re

def clean_model_name(model_name: str) -> str:
    """
    Clean and shorten model names for better readability.
    
    Args:
        model_name: Full model name from results directory
        
    Returns:
        Cleaned model name
    """
    # Extract the main model name from the long directory name
    if 'anthropic_claude' in model_name:
        return 'Claude-3.5-Sonnet'
    elif 'openai_gpt-4o-mini' in model_name:
        return 'GPT-4o-Mini'
    elif 'together_deepseek-ai_DeepSeek-R1-Distill-Qwen-1.5B' in model_name:
        return 'DeepSeek-R1-Distill'
    elif 'together_google_gemma-3n-E4B-it' in model_name:
        return 'Gemma-3n-E4B'
    elif 'together_mistralai_Mistral-7B-Instruct-v0.3' in model_name:
        return 'Mistral-7B-Instruct'
    else:
        # Fallback: extract the first meaningful part
        parts = model_name.split('_')
        if len(parts) >= 2:
            return f"{parts[0]}-{parts[1]}"
        return model_name

def create_clean_performance_table(data: pd.DataFrame, 
                                 methods: list = ['Elo', 'Glicko2', 'WinCount'],
                                 metrics: list = ['accuracy', 'inconsistency_score', 'mse']) -> pd.DataFrame:
    """
    Create a clean performance table with shortened model names.
    
    Args:
        data: DataFrame with performance data
        methods: List of methods to include
        metrics: List of metrics to include
        
    Returns:
        DataFrame formatted as a clean performance table
    """
    # Filter for specified methods
    data = data[data['method'].isin(methods)]
    
    # Clean model names
    data['clean_model_name'] = data['model_name'].apply(clean_model_name)
    
    # Create the table format
    table_data = []
    
    for model in data['clean_model_name'].unique():
        model_data = data[data['clean_model_name'] == model]
        
        for i, (_, row) in enumerate(model_data.iterrows()):
            table_row = {
                'Model': model if i == 0 else '',  # Only show model name in first row
                'Method': row['method']
            }
            
            # Add metrics
            for metric in metrics:
                value = row[metric]
                if pd.isna(value):
                    table_row[metric.replace('_', ' ').title()] = ''
                else:
                    # Format based on metric type
                    if metric == 'accuracy':
                        table_row['Accuracy'] = f"{value:.2f}%"
                    elif metric == 'inconsistency_score':
                        table_row['Inconsistency Score'] = f"{value:.4f}"
                    elif metric == 'mse':
                        table_row['MSE'] = f"{value:.4f}"
                    elif metric == 'mae':
                        table_row['MAE'] = f"{value:.4f}"
                    elif metric == 'log_loss':
                        table_row['Log Loss'] = f"{value:.4f}"
            
            table_data.append(table_row)
    
    table_df = pd.DataFrame(table_data)
    
    # Remove columns that have no data
    for metric in metrics:
        metric_col = metric.replace('_', ' ').title()
        if metric_col in table_df.columns:
            # Check if the column has any non-empty values
            if table_df[metric_col].isna().all() or (table_df[metric_col] == '').all():
                table_df = table_df.drop(columns=[metric_col])
    
    # Also remove any duplicate or empty columns
    columns_to_remove = []
    for col in table_df.columns:
        if col == 'Mse' and 'MSE' in table_df.columns:  # Remove duplicate Mse column
            columns_to_remove.append(col)
        elif table_df[col].isna().all() or (table_df[col] == '').all():
            columns_to_remove.append(col)
    
    table_df = table_df.drop(columns=columns_to_remove)
    
    return table_df

def main():
    """
    Generate a clean performance table for all models.
    """
    print("Loading results data...")
    data = load_results_data()
    
    if data.empty:
        print("No data found. Please run some evaluations first.")
        return
    
    print(f"Loaded data for {len(data['model_name'].unique())} models")
    print(f"Available methods: {list(data['method'].unique())}")
    
    # Create clean performance table
    print("\n" + "="*80)
    print("PERFORMANCE TABLE: All Models with Elo, Glicko2, and Win Frequency")
    print("="*80)
    
    table_data = create_clean_performance_table(data)
    
    # Print the table
    print(table_data.to_string(index=False))
    
    # Save to CSV
    output_file = "logs/clean_performance_table.csv"
    table_data.to_csv(output_file, index=False)
    print(f"\nTable saved to: {output_file}")
    
    # Also create a version with only models that have complete data
    print("\n" + "="*80)
    print("PERFORMANCE TABLE: Models with Complete Data")
    print("="*80)
    
    # Filter for models that have all metrics available
    complete_data = data.copy()
    complete_data['clean_model_name'] = complete_data['model_name'].apply(clean_model_name)
    
    # Find models with complete data (all metrics available)
    models_with_complete_data = []
    for model in complete_data['clean_model_name'].unique():
        model_data = complete_data[complete_data['clean_model_name'] == model]
        if len(model_data) == 3 and not model_data[['accuracy', 'inconsistency_score', 'mse']].isna().all().all():
            models_with_complete_data.append(model)
    
    if models_with_complete_data:
        complete_table = create_clean_performance_table(
            complete_data[complete_data['clean_model_name'].isin(models_with_complete_data)]
        )
        print(complete_table.to_string(index=False))
        
        # Save complete table
        complete_output_file = "logs/complete_performance_table.csv"
        complete_table.to_csv(complete_output_file, index=False)
        print(f"\nComplete table saved to: {complete_output_file}")
    else:
        print("No models found with complete data for all metrics.")

if __name__ == "__main__":
    main() 