import pandas as pd
import glob
import os

# Directory containing the prompts variance CSV files
variances_dir = 'logs/variances_over_prompts'
output_dir = 'logs/variances_over_prompts'

# Find all variance CSV files in the prompts directory
csv_files = glob.glob(os.path.join(variances_dir, 'ranking_variance_*.csv'))

# Read and combine all variance results
all_results = []
for file in csv_files:
    df = pd.read_csv(file)
    all_results.append(df)

# Combine all results
combined_df = pd.concat(all_results, ignore_index=True)

# Create a summary table for plotting
summary_data = []
for _, row in combined_df.iterrows():
    # Extract model name from run1 (take the part before the first underscore)
    model_name = row['run1'].split('_')[0]
    
    # Clean up model names for display
    if model_name == 'anthropic':
        model_name = 'Claude-3.5-Sonnet'
    elif model_name == 'openai':
        model_name = 'GPT-4o-Mini'
    elif model_name == 'together':
        # Extract the actual model name from the together runs
        parts = row['run1'].split('_')
        if 'deepseek' in row['run1']:
            model_name = 'DeepSeek-R1-Distill'
        elif 'google' in row['run1']:
            model_name = 'Gemma-3n-E4B'
        elif 'mistralai' in row['run1']:
            model_name = 'Mistral-7B-Instruct'
    
    summary_data.append({
        'Model': model_name,
        'Method': row['method'],
        'Total Variation Distance': row['tvd'],
        'Jensen-Shannon Divergence': row['jsd'],
        'Rank Edit Distance': row['rank_edit_distance']
    })

# Create summary DataFrame
summary_df = pd.DataFrame(summary_data)

# Save the summary table
os.makedirs(output_dir, exist_ok=True)
summary_df.to_csv(os.path.join(output_dir, 'variance_metrics_table.csv'), index=False)

print(f"Prompts variance summary table saved to {output_dir}/variance_metrics_table.csv")
print(f"Total entries: {len(summary_df)}")
print("\nSummary by model:")
print(summary_df.groupby('Model').size()) 