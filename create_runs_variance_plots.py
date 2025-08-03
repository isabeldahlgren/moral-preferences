import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set seaborn style for better-looking plots
sns.set_style("whitegrid")
sns.set_palette("husl")

# Read the variance metrics data
df = pd.read_csv('logs/variances_over_runs/variance_metrics_table.csv')

# Use classic seaborn color palette
colors = sns.color_palette("husl", 3)
method_colors = {'Elo': colors[0], 'Glicko2': colors[1], 'WinCount': colors[2]}

# Function to create grouped bar plots
def create_grouped_bar_plot(df, metric, title, ylabel, filename):
    plt.figure(figsize=(16, 8))
    
    # Get unique models and methods
    models = df['Model'].unique()
    methods = df['Method'].unique()
    
    # Set up bar positions
    x = np.arange(len(models))
    width = 0.25  # Width of each bar
    offsets = np.linspace(-width, width, len(methods))
    
    # Create bars for each method
    for i, method in enumerate(methods):
        method_data = df[df['Method'] == method]
        values = [method_data[method_data['Model'] == model][metric].iloc[0] 
                 if len(method_data[method_data['Model'] == model]) > 0 else 0 
                 for model in models]
        
        bars = plt.bar(x + offsets[i], values, width, 
                       label=method, color=method_colors[method], alpha=0.8,
                       edgecolor='white', linewidth=1)
        
        # Add value labels on bars - clean without boxes
        for bar, value in zip(bars, values):
            height = bar.get_height()
            if height > 0:  # Only add label if bar has height
                if metric == 'Rank Edit Distance':
                    label_text = f'{height:.0f}'
                elif metric == 'Jensen-Shannon Divergence':
                    label_text = f'{height:.4f}'
                else:
                    label_text = f'{height:.3f}'
                
                plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        label_text, ha='center', va='bottom', fontsize=10, fontweight='bold',
                        color='black')
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.ylabel(ylabel, fontsize=14, fontweight='bold')
    plt.xlabel('Model', fontsize=14, fontweight='bold')
    plt.xticks(x, models, rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(title='Ranking Method', title_fontsize=12, fontsize=11, frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

# Create all three plots
create_grouped_bar_plot(df, 'Total Variation Distance', 
                       'Total Variation Distance Across Various Runs',
                       'Total Variation Distance',
                       'logs/variances_over_runs/tvd_plot.png')

create_grouped_bar_plot(df, 'Jensen-Shannon Divergence',
                       'Jensen-Shannon Divergence Across Various Runs',
                       'Jensen-Shannon Divergence',
                       'logs/variances_over_runs/jsd_plot.png')

create_grouped_bar_plot(df, 'Rank Edit Distance',
                       'Rank Edit Distance Across Various Runs',
                       'Rank Edit Distance',
                       'logs/variances_over_runs/rank_edit_distance_plot.png')

print("Individual plots saved:")
print("- logs/variances_over_runs/tvd_plot.png")
print("- logs/variances_over_runs/jsd_plot.png") 
print("- logs/variances_over_runs/rank_edit_distance_plot.png")

# Summary statistics
print("\nSummary Statistics:")
print("=" * 50)
for metric in ['Total Variation Distance', 'Jensen-Shannon Divergence', 'Rank Edit Distance']:
    print(f"\n{metric}:")
    print(f"  Min: {df[metric].min():.4f}")
    print(f"  Max: {df[metric].max():.4f}")
    print(f"  Mean: {df[metric].mean():.4f}")
    print(f"  Std: {df[metric].std():.4f}") 