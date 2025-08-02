#!/usr/bin/env python3
"""
Create a matplotlib table plot similar to the image shown.
This script generates a clean, formatted table with model performance data.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from plotting_utils import load_results_data
from generate_performance_table import clean_model_name, create_clean_performance_table

def create_table_plot(data: pd.DataFrame, 
                     save_path: str = "logs/plots/performance_table.png",
                     figsize: tuple = (14, 10)) -> None:
    """
    Create a matplotlib table plot with proper formatting.
    
    Args:
        data: DataFrame with performance data
        save_path: Path to save the plot
        figsize: Figure size (width, height)
    """
    # Create clean performance table
    table_data = create_clean_performance_table(data)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data for the table
    # Remove the 'Mse' column if it exists (duplicate of 'MSE')
    if 'Mse' in table_data.columns:
        table_data = table_data.drop('Mse', axis=1)
    
    # Create the table
    table = ax.table(cellText=table_data.values,
                    colLabels=table_data.columns,
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    # Color header row
    for i in range(len(table_data.columns)):
        table[(0, i)].set_facecolor('#2E86AB')  # Nice blue color
        table[(0, i)].set_text_props(weight='bold', color='white', size=12)
    
    # Color alternating rows and style cells
    for i in range(1, len(table_data) + 1):
        for j in range(len(table_data.columns)):
            cell = table[(i, j)]
            
            # Color alternating rows
            if i % 2 == 0:
                cell.set_facecolor('#F8F9FA')  # Light gray
            else:
                cell.set_facecolor('#FFFFFF')  # White
            
            # Style based on column type
            if j == 0:  # Model column
                cell.set_text_props(weight='bold')
            elif j == 1:  # Method column
                cell.set_text_props(style='italic')
            
            # Add borders
            cell.set_edgecolor('#DEE2E6')
    
    # Add title
    plt.title('Model Performance Comparison\nAccuracy, Inconsistency Score, and MSE by Ranking Method', 
              fontsize=16, fontweight='bold', pad=30)
    
    # Add subtitle with information
    plt.figtext(0.5, 0.02, 
                'Note: Empty cells indicate missing data for that metric/method combination',
                ha='center', fontsize=10, style='italic', color='gray')
    
    # Save the plot
    plt.savefig(save_path, bbox_inches='tight', dpi=300, facecolor='white')
    print(f"Table plot saved to: {save_path}")
    
    # Show the plot
    plt.show()

def create_simple_table_plot(data: pd.DataFrame, 
                           save_path: str = "logs/plots/simple_performance_table.png") -> None:
    """
    Create a simple table plot with just the essential information.
    
    Args:
        data: DataFrame with performance data
        save_path: Path to save the plot
    """
    # Create clean performance table
    table_data = create_clean_performance_table(data)
    
    # Filter to only show models with some data
    table_data = table_data[table_data['Accuracy'] != '']
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Create the table
    table = ax.table(cellText=table_data.values,
                    colLabels=table_data.columns,
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Color header row
    for i in range(len(table_data.columns)):
        table[(0, i)].set_facecolor('#4CAF50')  # Green color
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color alternating rows
    for i in range(1, len(table_data) + 1):
        for j in range(len(table_data.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    plt.title('Model Performance Comparison', fontsize=14, fontweight='bold', pad=20)
    
    # Save the plot
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Simple table plot saved to: {save_path}")
    
    # Show the plot
    plt.show()

def main():
    """
    Create table plots for the performance data.
    """
    print("Loading results data...")
    data = load_results_data()
    
    if data.empty:
        print("No data found. Please run some evaluations first.")
        return
    
    print(f"Loaded data for {len(data['model_name'].unique())} models")
    
    # Create detailed table plot
    print("\nCreating detailed table plot...")
    create_table_plot(data)
    
    # Create simple table plot
    print("\nCreating simple table plot...")
    create_simple_table_plot(data)
    
    print("\nDone! Check the logs/plots/ directory for the generated plots.")

if __name__ == "__main__":
    main() 