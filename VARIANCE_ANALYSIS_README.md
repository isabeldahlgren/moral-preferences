# Variance Analysis for Model Ranking Comparison

This module provides functionality to analyze ranking variance between different runs of the same model using three key metrics: Total Variation Distance, Jensen-Shannon Divergence, and Rank Edit Distance.

## Overview

The variance analysis compares pairs of runs for each model to measure how consistent the ranking methods are across different evaluations. This helps understand the stability and reliability of different ranking approaches.

## Metrics Explained

### 1. Total Variation Distance (TVD)
- **Definition**: Measures the difference between two probability distributions
- **Range**: 0 to 1 (lower is better)
- **Interpretation**: How much the normalized rankings differ between runs

### 2. Jensen-Shannon Divergence (JSD)
- **Definition**: Symmetric measure of difference between two probability distributions
- **Range**: 0 to 1 (lower is better)
- **Interpretation**: Information-theoretic measure of ranking divergence

### 3. Rank Edit Distance (RED)
- **Definition**: Sum of absolute differences between rank positions
- **Range**: 0 to N (lower is better, where N is number of players)
- **Interpretation**: How much the rank orderings differ between runs

## Files Created

- `variance_plotting.py` - Core variance analysis and plotting utilities
- `demo_variance_analysis.py` - Complete demo showing both performance and variance analysis

## Usage

### Quick Start

```bash
# Run variance analysis for all models
python variance_plotting.py

# Run complete demo (performance + variance)
python demo_variance_analysis.py
```

### Manual Analysis

```python
from variance_plotting import collect_variance_data, create_variance_plots

# Collect variance data for all model pairs
data = collect_variance_data()

# Create variance comparison plots
create_variance_plots(data)
```

## How It Works

1. **Model Pair Detection**: Automatically finds pairs of runs for each model
2. **Variance Calculation**: Runs `measure_ranking_variance.py` between run pairs
3. **Data Collection**: Collects TVD, JSD, and RED metrics for each method
4. **Plot Generation**: Creates three separate heatmap plots
5. **Table Creation**: Generates clean variance metrics table

## Output Files

### Plots (in `logs/plots/`)
- `tvd_comparison.png` - Total Variation Distance comparison
- `jsd_comparison.png` - Jensen-Shannon Divergence comparison  
- `rank_edit_distance_comparison.png` - Rank Edit Distance comparison

### Tables (in `logs/variances/`)
- `variance_metrics_table.csv` - Clean table with all variance metrics
- `ranking_variance_*.csv` - Individual variance analysis results

## Example Results

The analysis found variance metrics for 5 models:

| Model | Method | Total Variation Distance | Jensen-Shannon Divergence | Rank Edit Distance |
|-------|--------|------------------------|---------------------------|-------------------|
| DeepSeek-R1-Distill | Elo | 0.0266 | 0.0007 | 108 |
| | Glicko2 | 0.0103 | 0.0001 | 102 |
| | WinCount | 0.0415 | 0.0017 | 78 |
| GPT-4o-Mini | Elo | 0.0134 | 0.0002 | 88 |
| | Glicko2 | 0.0347 | 0.0013 | 74 |
| | WinCount | 0.1365 | 0.0306 | 44 |

## Key Insights

### Model Consistency
- **Most Consistent**: GPT-4o-Mini (low TVD/JSD values)
- **Least Consistent**: Gemma-3n-E4B (high WinCount variance)

### Method Performance
- **Most Stable**: Glicko2 (generally lower variance)
- **Least Stable**: WinCount (higher variance across models)

### Ranking Stability
- **Best**: Glicko2 for most models
- **Worst**: WinCount for GPT-4o-Mini and Gemma-3n-E4B

## Integration with Performance Analysis

The variance analysis complements the performance metrics:

- **Performance Metrics**: Accuracy, Inconsistency Score, MSE
- **Variance Metrics**: TVD, JSD, Rank Edit Distance

Together, they provide a complete picture of:
1. How well each method performs (performance metrics)
2. How consistent each method is across runs (variance metrics)

## Requirements

- Python 3.11+
- pandas, matplotlib, seaborn, numpy, scipy
- Existing `measure_ranking_variance.py` script
- Multiple runs per model in `logs/results/`

## Troubleshooting

### No Variance Data
- Ensure you have at least 2 runs per model
- Check that ranking files exist in run directories
- Verify `measure_ranking_variance.py` works correctly

### Missing Plots
- Check that variance data was collected successfully
- Ensure `logs/plots/` directory exists
- Verify matplotlib/seaborn installation

### Model Name Issues
- The system automatically shortens model names
- Check the `clean_model_name()` function for custom mappings 