import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set seaborn style for better-looking plots
sns.set_style("whitegrid")
sns.set_palette("husl")

# Read the variance metrics data
df = pd.read_csv('logs/variances_over_prompts/variance_metrics_table.csv')

# Create a more visually appealing color palette
colors = sns.color_palette("Set2", 3)
method_colors = {'Elo': colors[0], 'Glicko2': colors[1], 'WinCount': colors[2]}

# Plot 1: Total Variation Distance
plt.figure(figsize=(14, 8))
for method in df['Method'].unique():
    method_data = df[df['Method'] == method]
    bars = plt.bar(method_data['Model'], method_data['Total Variation Distance'], 
                   label=method, color=method_colors[method], alpha=0.8, 
                   edgecolor='white', linewidth=1.5)
    
    # Add value labels on bars - positioned just above the bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.title('Total Variation Distance Across Various Prompts', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('Total Variation Distance', fontsize=14, fontweight='bold')
plt.xlabel('Model', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)
plt.legend(title='Ranking Method', title_fontsize=12, fontsize=11, frameon=True, fancybox=True, shadow=True)
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('logs/variances_over_prompts/tvd_plot.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# Plot 2: Jensen-Shannon Divergence
plt.figure(figsize=(14, 8))
for method in df['Method'].unique():
    method_data = df[df['Method'] == method]
    bars = plt.bar(method_data['Model'], method_data['Jensen-Shannon Divergence'], 
                   label=method, color=method_colors[method], alpha=0.8,
                   edgecolor='white', linewidth=1.5)
    
    # Add value labels on bars - positioned just above the bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{height:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.title('Jensen-Shannon Divergence Across Various Prompts', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('Jensen-Shannon Divergence', fontsize=14, fontweight='bold')
plt.xlabel('Model', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)
plt.legend(title='Ranking Method', title_fontsize=12, fontsize=11, frameon=True, fancybox=True, shadow=True)
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('logs/variances_over_prompts/jsd_plot.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# Plot 3: Rank Edit Distance
plt.figure(figsize=(14, 8))
for method in df['Method'].unique():
    method_data = df[df['Method'] == method]
    bars = plt.bar(method_data['Model'], method_data['Rank Edit Distance'], 
                   label=method, color=method_colors[method], alpha=0.8,
                   edgecolor='white', linewidth=1.5)
    
    # Add value labels on bars - positioned just above the bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{height:.0f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.title('Rank Edit Distance Across Various Prompts', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('Rank Edit Distance', fontsize=14, fontweight='bold')
plt.xlabel('Model', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)
plt.legend(title='Ranking Method', title_fontsize=12, fontsize=11, frameon=True, fancybox=True, shadow=True)
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('logs/variances_over_prompts/rank_edit_distance_plot.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

print("Individual plots saved:")
print("- logs/variances_over_prompts/tvd_plot.png")
print("- logs/variances_over_prompts/jsd_plot.png") 
print("- logs/variances_over_prompts/rank_edit_distance_plot.png")

# Summary statistics
print("\nSummary Statistics:")
print("=" * 50)
for metric in ['Total Variation Distance', 'Jensen-Shannon Divergence', 'Rank Edit Distance']:
    print(f"\n{metric}:")
    print(f"  Min: {df[metric].min():.4f}")
    print(f"  Max: {df[metric].max():.4f}")
    print(f"  Mean: {df[metric].mean():.4f}")
    print(f"  Std: {df[metric].std():.4f}") 