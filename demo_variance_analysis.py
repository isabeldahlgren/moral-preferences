#!/usr/bin/env python3
"""
Demo script showing both performance metrics and variance analysis.
This demonstrates the complete analysis pipeline with both types of metrics.
"""

import pandas as pd
from plotting_utils import load_results_data, create_performance_table, create_comparison_plots
from variance_plotting import collect_variance_data, create_variance_table, create_variance_plots

def main():
    """
    Demonstrate both performance and variance analysis.
    """
    print("="*80)
    print("COMPLETE MODEL ANALYSIS DEMO")
    print("="*80)
    
    # Part 1: Performance Metrics Analysis
    print("\n" + "="*60)
    print("PART 1: PERFORMANCE METRICS ANALYSIS")
    print("="*60)
    
    print("\n1. Loading performance data...")
    perf_data = load_results_data()
    
    if not perf_data.empty:
        print(f"✓ Loaded performance data for {len(perf_data['model_name'].unique())} models")
        print(f"✓ Available methods: {list(perf_data['method'].unique())}")
        
        # Create performance table
        print("\n2. Creating performance table...")
        perf_table = create_performance_table(perf_data)
        print("\nPERFORMANCE METRICS TABLE:")
        print("-" * 60)
        print(perf_table.to_string(index=False))
        
        # Create performance plots
        print("\n3. Creating performance comparison plots...")
        create_comparison_plots(perf_data)
        print("✓ Generated performance plots:")
        print("   - accuracy_comparison.png")
        print("   - inconsistency_score_comparison.png")
        print("   - mse_comparison.png")
    else:
        print("⚠ No performance data available")
    
    # Part 2: Variance Analysis
    print("\n" + "="*60)
    print("PART 2: VARIANCE ANALYSIS")
    print("="*60)
    
    print("\n1. Collecting variance data...")
    var_data = collect_variance_data()
    
    if not var_data.empty:
        print(f"✓ Collected variance data for {len(var_data['model'].unique())} models")
        print(f"✓ Available methods: {list(var_data['method'].unique())}")
        
        # Create variance table
        print("\n2. Creating variance table...")
        var_table = create_variance_table(var_data)
        print("\nVARIANCE METRICS TABLE:")
        print("-" * 60)
        print(var_table.to_string(index=False))
        
        # Create variance plots
        print("\n3. Creating variance comparison plots...")
        create_variance_plots(var_data)
        print("✓ Generated variance plots:")
        print("   - tvd_comparison.png (Total Variation Distance)")
        print("   - jsd_comparison.png (Jensen-Shannon Divergence)")
        print("   - rank_edit_distance_comparison.png (Rank Edit Distance)")
    else:
        print("⚠ No variance data available")
    
    # Summary
    print("\n" + "="*80)
    print("ANALYSIS SUMMARY")
    print("="*80)
    
    if not perf_data.empty:
        print("✓ Performance Metrics:")
        print("   - Accuracy, Inconsistency Score, MSE")
        print("   - Methods: Elo, Glicko2, WinCount")
        print("   - Model names automatically shortened")
    
    if not var_data.empty:
        print("✓ Variance Metrics:")
        print("   - Total Variation Distance")
        print("   - Jensen-Shannon Divergence") 
        print("   - Rank Edit Distance")
        print("   - Comparing pairs of runs per model")
    
    print("\n✓ All plots saved to logs/plots/")
    print("✓ All tables saved to logs/variances/")
    print("✓ Complete analysis pipeline working!")
    
    print("\n" + "="*80)
    print("Demo completed successfully!")
    print("="*80)

if __name__ == "__main__":
    main() 