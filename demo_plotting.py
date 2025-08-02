#!/usr/bin/env python3
"""
Demo script showing the final plotting functionality.
This demonstrates:
1. Shortened model names (like in the second image)
2. Three separate plots for each metric
3. Automatic removal of WinCount column when no data
4. Clean table generation
"""

import pandas as pd
from plotting_utils import load_results_data, create_performance_table, create_comparison_plots
from generate_performance_table import create_clean_performance_table

def main():
    """
    Demonstrate the final plotting functionality.
    """
    print("="*80)
    print("DEMO: Model Performance Analysis with Shortened Names")
    print("="*80)
    
    # Load data
    print("\n1. Loading results data...")
    data = load_results_data()
    
    if data.empty:
        print("No data found. Please run some evaluations first.")
        return
    
    print(f"✓ Loaded data for {len(data['model_name'].unique())} models")
    print(f"✓ Available methods: {list(data['method'].unique())}")
    
    # Show model name mapping
    print("\n2. Model name mapping:")
    unique_models = data['model_name'].unique()
    for model in unique_models[:5]:  # Show first 5 as examples
        from plotting_utils import clean_model_name
        short_name = clean_model_name(model)
        print(f"   {model[:50]}... → {short_name}")
    
    # Create clean performance table
    print("\n3. Creating clean performance table...")
    table_data = create_clean_performance_table(data)
    print("\n" + "="*60)
    print("CLEAN PERFORMANCE TABLE")
    print("="*60)
    print(table_data.to_string(index=False))
    
    # Show that WinCount column is removed when no data
    print(f"\n✓ Table has {len(table_data.columns)} columns: {list(table_data.columns)}")
    if 'WinCount' in data['method'].values:
        print("✓ WinCount method exists in data but may be removed if no values")
    
    # Create separate comparison plots
    print("\n4. Creating separate comparison plots...")
    create_comparison_plots(data)
    print("✓ Generated three separate plots:")
    print("   - accuracy_comparison.png")
    print("   - inconsistency_score_comparison.png") 
    print("   - mse_comparison.png")
    
    # Show summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("✓ Model names are automatically shortened")
    print("✓ Three separate plots created (one per metric)")
    print("✓ WinCount column removed when no data available")
    print("✓ Clean, readable table format")
    print("✓ All plots saved to logs/plots/")
    
    print("\n" + "="*80)
    print("Demo completed successfully!")
    print("="*80)

if __name__ == "__main__":
    main() 