#!/usr/bin/env python3
"""
Example script demonstrating how to use the plotting utilities
to generate performance tables and plots for different models and ranking methods.
"""

import pandas as pd
from plotting_utils import (
    load_results_data,
    create_performance_table,
    plot_performance_table,
    create_comparison_plots,
    generate_summary_report
)

def main():
    """
    Example usage of the plotting utilities.
    """
    print("Loading results data...")
    data = load_results_data()
    
    if data.empty:
        print("No data found. Please run some evaluations first.")
        return
    
    print(f"Loaded data for {len(data['model_name'].unique())} models:")
    for model in data['model_name'].unique():
        print(f"  - {model}")
    
    print(f"\nAvailable methods: {list(data['method'].unique())}")
    
    # Example 1: Create a performance table for all models
    print("\n" + "="*50)
    print("EXAMPLE 1: Performance Table for All Models")
    print("="*50)
    
    table_data = create_performance_table(data)
    print(table_data.to_string(index=False))
    
    # Example 2: Create a performance table for specific models
    print("\n" + "="*50)
    print("EXAMPLE 2: Performance Table for Specific Models")
    print("="*50)
    
    # Get first few models as example
    sample_models = data['model_name'].unique()[:3]
    table_data_filtered = create_performance_table(data, models=sample_models)
    print(table_data_filtered.to_string(index=False))
    
    # Example 3: Create a performance table with specific methods only
    print("\n" + "="*50)
    print("EXAMPLE 3: Performance Table with Elo and Glicko2 Only")
    print("="*50)
    
    table_data_methods = create_performance_table(
        data, 
        methods=['Elo', 'Glicko2']
    )
    print(table_data_methods.to_string(index=False))
    
    # Example 4: Create a matplotlib table plot
    print("\n" + "="*50)
    print("EXAMPLE 4: Creating Matplotlib Table Plot")
    print("="*50)
    
    # Use a subset for better visualization
    sample_data = data[data['model_name'].isin(sample_models)]
    sample_table = create_performance_table(sample_data, methods=['Elo', 'Glicko2'])
    
    plot_performance_table(
        sample_table,
        title="Model Performance Comparison (Sample)",
        save_path="logs/plots/performance_table_example.png"
    )
    
    # Example 5: Create comparison heatmaps
    print("\n" + "="*50)
    print("EXAMPLE 5: Creating Comparison Heatmaps")
    print("="*50)
    
    create_comparison_plots(
        data,
        output_dir="logs/plots",
        models=sample_models
    )
    
    # Example 6: Generate summary report
    print("\n" + "="*50)
    print("EXAMPLE 6: Generating Summary Report")
    print("="*50)
    
    generate_summary_report(data, "logs/performance_summary_example.csv")
    
    print("\n" + "="*50)
    print("All examples completed!")
    print("Check the logs/plots/ directory for generated plots.")
    print("="*50)

if __name__ == "__main__":
    main() 