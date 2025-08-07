import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import glob
import os

def load_results_data(results_dir: str = "logs/results") -> pd.DataFrame:
    """
    Load all results data from the results directory and combine into a single DataFrame.
    
    Args:
        results_dir: Path to the results directory
        
    Returns:
        DataFrame with columns: model_name, method, accuracy, inconsistency_score, mse, mae, log_loss
    """
    all_data = []
    
    # Find all result directories
    result_dirs = glob.glob(os.path.join(results_dir, "*"))
    
    for result_dir in result_dirs:
        if not os.path.isdir(result_dir):
            continue
            
        # Extract model name from directory path
        dir_name = os.path.basename(result_dir)
        
        # Look for metrics and evaluation files
        metrics_files = glob.glob(os.path.join(result_dir, "*metrics*.csv"))
        eval_files = glob.glob(os.path.join(result_dir, "*evaluation*.csv"))
        
        for metrics_file in metrics_files:
            try:
                metrics_df = pd.read_csv(metrics_file)
                eval_df = None
                
                # Find corresponding evaluation file
                for eval_file in eval_files:
                    if os.path.basename(metrics_file).split('_')[0] == os.path.basename(eval_file).split('_')[0]:
                        eval_df = pd.read_csv(eval_file)
                        break
                
                # Process each method in the metrics file
                for _, row in metrics_df.iterrows():
                    method = row.iloc[0]  # First column is method name
                    inconsistency_score = row['inconsistency_score']
                    
                    # Get corresponding evaluation metrics
                    accuracy = mse = mae = log_loss = np.nan
                    if eval_df is not None:
                        eval_row = eval_df[eval_df.iloc[:, 0] == method.lower()]
                        if not eval_row.empty:
                            accuracy = eval_row['accuracy'].iloc[0]
                            mse = eval_row['mse'].iloc[0]
                            mae = eval_row['mae'].iloc[0]
                            log_loss = eval_row['log_loss'].iloc[0]
                    
                    all_data.append({
                        'model_name': dir_name,
                        'method': method,
                        'accuracy': accuracy,
                        'inconsistency_score': inconsistency_score,
                        'mse': mse,
                        'mae': mae,
                        'log_loss': log_loss
                    })
                    
            except Exception as e:
                print(f"Error processing {metrics_file}: {e}")
                continue
    
    return pd.DataFrame(all_data)

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

def create_performance_table(data: pd.DataFrame, 
                           models: Optional[List[str]] = None,
                           methods: Optional[List[str]] = None,
                           metrics: List[str] = ['accuracy', 'inconsistency_score', 'mse']) -> pd.DataFrame:
    """
    Create a performance table in the format shown in the image.
    
    Args:
        data: DataFrame with performance data
        models: List of model names to include (if None, include all)
        methods: List of methods to include (if None, include all)
        metrics: List of metrics to include in the table
        
    Returns:
        DataFrame formatted as a performance table
    """
    if models is not None:
        data = data[data['model_name'].isin(models)]
    
    if methods is not None:
        data = data[data['method'].isin(methods)]
    
    # Clean model names
    data = data.copy()
    data['clean_model_name'] = data['model_name'].apply(clean_model_name)
    
    # Check which methods have data for each metric
    available_methods = {}
    for metric in metrics:
        metric_data = data[data[metric].notna()]
        if not metric_data.empty:
            available_methods[metric] = metric_data['method'].unique()
        else:
            available_methods[metric] = []
    
    # Pivot the data to create the table format
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

def plot_performance_table(data: pd.DataFrame, 
                         title: str = "Model Performance Comparison",
                         figsize: Tuple[int, int] = (12, 8),
                         save_path: Optional[str] = None) -> None:
    """
    Create a matplotlib table plot of the performance data.
    
    Args:
        data: DataFrame with performance table data
        title: Title for the plot
        figsize: Figure size (width, height)
        save_path: Optional path to save the plot
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('tight')
    ax.axis('off')
    
    # Create the table
    table = ax.table(cellText=data.values,
                    colLabels=data.columns,
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Color header row
    for i in range(len(data.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color alternating rows
    for i in range(1, len(data) + 1):
        for j in range(len(data.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    plt.show()

def create_comparison_plots(data: pd.DataFrame, 
                          output_dir: str = "logs/plots",
                          models: Optional[List[str]] = None,
                          methods: Optional[List[str]] = None) -> None:
    """
    Create comparison plots for different metrics across models and methods.
    
    Args:
        data: DataFrame with performance data
        output_dir: Directory to save plots
        models: List of models to include
        methods: List of methods to include
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    if models is not None:
        data = data[data['model_name'].isin(models)]
    
    if methods is not None:
        data = data[data['method'].isin(methods)]
    
    # Clean model names
    data = data.copy()
    data['clean_model_name'] = data['model_name'].apply(clean_model_name)
    
    # Handle duplicate model names by taking the first occurrence for each model-method combination
    data = data.drop_duplicates(subset=['clean_model_name', 'method'], keep='first')
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create separate plots for each metric
    metrics = ['accuracy', 'inconsistency_score', 'mse']
    
    for metric in metrics:
        # Create pivot table for this metric using clean model names
        pivot_data = data.pivot(index='clean_model_name', columns='method', values=metric)
        
        # Remove columns that have no data (all NaN)
        pivot_data = pivot_data.dropna(axis=1, how='all')
        
        # Only create plot if there's data
        if not pivot_data.empty and not pivot_data.isna().all().all():
            plt.figure(figsize=(8, 6))
            
            # Create heatmap
            sns.heatmap(pivot_data, annot=True, fmt='.4f', cmap='RdYlBu_r', 
                       cbar_kws={'label': metric.replace('_', ' ').title()})
            plt.title(f'{metric.replace("_", " ").title()} Comparison')
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
        else:
            print(f"No data available for {metric} comparison")
    
    print(f"Individual comparison plots saved to {output_dir}")

def generate_summary_report(data: pd.DataFrame, 
                          output_file: str = "logs/performance_summary.csv") -> None:
    """
    Generate a summary report with the best performing method for each model.
    
    Args:
        data: DataFrame with performance data
        output_file: Path to save the summary report
    """
    # Clean model names
    data = data.copy()
    data['clean_model_name'] = data['model_name'].apply(clean_model_name)
    
    summary_data = []
    
    for model in data['clean_model_name'].unique():
        model_data = data[data['clean_model_name'] == model]
        
        # Find best method for each metric
        metrics = ['accuracy', 'inconsistency_score', 'mse']
        best_methods = {}
        
        for metric in metrics:
            # Filter out NaN values
            valid_data = model_data[model_data[metric].notna()]
            
            if valid_data.empty:
                best_methods[metric] = 'N/A'
            else:
                if metric == 'accuracy':
                    best_idx = valid_data[metric].idxmax()
                else:  # Lower is better for inconsistency_score and mse
                    best_idx = valid_data[metric].idxmin()
                
                best_methods[metric] = valid_data.loc[best_idx, 'method']
        
        summary_data.append({
            'Model': model,
            'Best Accuracy Method': best_methods['accuracy'],
            'Best Inconsistency Score Method': best_methods['inconsistency_score'],
            'Best MSE Method': best_methods['mse']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_file, index=False)
    print(f"Summary report saved to {output_file}")

def main():
    """
    Main function to demonstrate the plotting utilities.
    """
    # Load data
    print("Loading results data...")
    data = load_results_data()
    
    if data.empty:
        print("No data found. Please check the results directory.")
        return
    
    print(f"Loaded data for {len(data['model_name'].unique())} models")
    print(f"Available methods: {data['method'].unique()}")
    
    # Create performance table
    print("\nCreating performance table...")
    table_data = create_performance_table(data)
    print(table_data.to_string(index=False))
    
    # Create plots
    print("\nCreating comparison plots...")
    create_comparison_plots(data)
    
    # Generate summary report
    print("\nGenerating summary report...")
    generate_summary_report(data)
    
    print("\nDone!")

if __name__ == "__main__":
    main() 