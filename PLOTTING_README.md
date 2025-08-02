# Plotting Utilities for Model Performance Analysis

This directory contains utilities for generating performance tables and plots for different models and ranking methods.

## Files Overview

- `plotting_utils.py` - Core plotting utilities for loading data and creating various visualizations
- `generate_performance_table.py` - Script to generate clean performance tables with shortened model names
- `create_table_plot.py` - Script to create matplotlib table plots similar to the image format
- `demo_plotting.py` - Demo script showing performance analysis features
- `demo_variance_analysis.py` - Complete demo showing both performance and variance analysis
- `variance_plotting.py` - Core variance analysis and plotting utilities

## Quick Start

### 1. Generate a Clean Performance Table

```bash
python generate_performance_table.py
```

This will create a table showing:
- Model names (cleaned and shortened)
- Methods: Elo, Glicko2, WinCount
- Metrics: Accuracy, Inconsistency Score, MSE

### 2. Create Table Plots

```bash
python create_table_plot.py
```

This generates:
- `logs/plots/performance_table.png` - Detailed table plot
- `logs/plots/simple_performance_table.png` - Simple table plot

### 3. Run Performance Analysis Demo

```bash
python demo_plotting.py
```

This demonstrates the performance analysis features:
- Model name mapping and shortening
- Performance table generation
- Comparison plots for accuracy, inconsistency score, and MSE
- Automatic WinCount column removal when no data

### 4. Run Complete Analysis Demo (Performance + Variance)

```bash
python demo_variance_analysis.py
```

This runs both performance and variance analysis:
- **Part 1**: Performance metrics (Accuracy, Inconsistency Score, MSE)
- **Part 2**: Variance metrics (Total Variation Distance, Jensen-Shannon Divergence, Rank Edit Distance)
- Compares pairs of runs for each model
- Generates all plots and tables

## Available Functions

### Data Loading
- `load_results_data()` - Loads all results from `logs/results/` directory

### Table Creation
- `create_performance_table()` - Creates performance tables in the format shown in the image
- `create_clean_performance_table()` - Creates tables with shortened model names

### Plotting
- `plot_performance_table()` - Creates matplotlib table plots
- `create_comparison_plots()` - Creates heatmap comparisons
- `generate_summary_report()` - Generates summary reports

## Table Format

The generated tables follow this format:

| Model | Method | Accuracy | Inconsistency Score | MSE |
|-------|--------|----------|-------------------|-----|
| Model name | Elo | 52.03% | 0.4877 | 0.2524 |
| | Glicko2 | 47.23% | 0.4788 | 0.2523 |
| | Win frequency | | 0.4605 | |

## Model Name Mapping

The utilities automatically clean long model names:
- `anthropic_claude-3-5-sonnet-latest_*` → `Claude-3.5-Sonnet`
- `openai_gpt-4o-mini_*` → `GPT-4o-Mini`
- `together_deepseek-ai_DeepSeek-R1-Distill-Qwen-1.5B_*` → `DeepSeek-R1-Distill`
- `together_google_gemma-3n-E4B-it_*` → `Gemma-3n-E4B`
- `together_mistralai_Mistral-7B-Instruct-v0.3_*` → `Mistral-7B-Instruct`

## Output Files

### Performance Analysis
- `logs/clean_performance_table.csv` - Clean performance table
- `logs/complete_performance_table.csv` - Table with only complete data
- `logs/performance_summary.csv` - Summary report
- `logs/plots/performance_table.png` - Detailed table plot
- `logs/plots/simple_performance_table.png` - Simple table plot
- `logs/plots/performance_comparison.png` - Heatmap comparison
- `logs/plots/accuracy_comparison.png` - Accuracy comparison plot
- `logs/plots/inconsistency_score_comparison.png` - Inconsistency score comparison plot
- `logs/plots/mse_comparison.png` - MSE comparison plot

### Variance Analysis
- `logs/variances/variance_metrics_table.csv` - Clean variance metrics table
- `logs/variances/ranking_variance_*.csv` - Individual variance analysis results
- `logs/plots/tvd_comparison.png` - Total Variation Distance comparison
- `logs/plots/jsd_comparison.png` - Jensen-Shannon Divergence comparison
- `logs/plots/rank_edit_distance_comparison.png` - Rank Edit Distance comparison

## Customization

You can customize the tables by modifying the parameters:

```python
from plotting_utils import load_results_data, create_performance_table

data = load_results_data()

# Filter for specific models
table = create_performance_table(data, models=['model1', 'model2'])

# Filter for specific methods
table = create_performance_table(data, methods=['Elo', 'Glicko2'])

# Include specific metrics
table = create_performance_table(data, metrics=['accuracy', 'mse'])
```

## Demo Scripts

### `demo_plotting.py` - Performance Analysis Demo

This script demonstrates the performance analysis features:

```bash
python demo_plotting.py
```

**What it does:**
- Loads all results data from `logs/results/`
- Shows model name mapping (long names → shortened names)
- Creates clean performance table with shortened model names
- Demonstrates automatic WinCount column removal when no data
- Generates three separate comparison plots:
  - `accuracy_comparison.png`
  - `inconsistency_score_comparison.png`
  - `mse_comparison.png`

**Example output:**
```
✓ Model names are automatically shortened
✓ Three separate plots created (one per metric)
✓ WinCount column removed when no data available
✓ Clean, readable table format
✓ All plots saved to logs/plots/
```

### `demo_variance_analysis.py` - Complete Analysis Demo

This script runs both performance and variance analysis:

```bash
python demo_variance_analysis.py
```

**What it does:**
- **Part 1**: Performance metrics analysis (same as `demo_plotting.py`)
- **Part 2**: Variance analysis between run pairs
  - Finds pairs of runs for each model
  - Runs `measure_ranking_variance.py` between pairs
  - Collects Total Variation Distance, Jensen-Shannon Divergence, and Rank Edit Distance
  - Creates three variance comparison plots:
    - `tvd_comparison.png`
    - `jsd_comparison.png`
    - `rank_edit_distance_comparison.png`

**Example output:**
```
✓ Performance Metrics:
   - Accuracy, Inconsistency Score, MSE
   - Methods: Elo, Glicko2, WinCount
   - Model names automatically shortened
✓ Variance Metrics:
   - Total Variation Distance
   - Jensen-Shannon Divergence
   - Rank Edit Distance
   - Comparing pairs of runs per model
```

## Dependencies

- pandas
- matplotlib
- seaborn
- numpy
- scipy (for variance analysis)

Install with: `uv add seaborn scipy` 