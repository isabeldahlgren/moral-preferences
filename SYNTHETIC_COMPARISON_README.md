# Synthetic Ranking Comparison Script

This script generates synthetic match data and compares different ranking methods against brute force optimization to evaluate their optimality with respect to inconsistency scores.

## Overview

The `synthetic_ranking_comparison.py` script creates controlled synthetic datasets with a small number of characters and evaluates how well different ranking methods perform compared to the truly optimal ranking found by testing all possible permutations.

## Ranking Methods Evaluated

1. **Elo Rating System** - Traditional Elo rating with k-factor=32
2. **Glicko2 Rating System** - Glicko-2 rating system with uncertainty tracking
3. **WinCount** - Simple win-count based ranking
4. **Simulated Annealing** - Optimization-based ranking using simulated annealing
5. **Brute Force** - Tests all possible permutations to find the truly optimal ranking

## Key Features

- **Synthetic Data Generation**: Creates controlled match data with configurable parameters
- **Brute Force Optimization**: Tests all possible character permutations (n! combinations)
- **Inconsistency Score**: Measures how often lower-ranked characters beat higher-ranked ones
- **Visual Comparison**: Generates plots comparing method performance
- **Detailed Analysis**: Provides comprehensive results and rankings

## Usage

### Basic Usage
```bash
python synthetic_ranking_comparison.py
```

### Advanced Options
```bash
python synthetic_ranking_comparison.py \
    --num_characters 5 \
    --num_matches_per_pair 3 \
    --seed 42 \
    --output_dir logs/my_comparison
```

### Command Line Arguments

- `--num_characters`: Number of characters to use (default: 5)
- `--num_matches_per_pair`: Number of matches per character pair (default: 3)
- `--seed`: Random seed for reproducibility (default: 42)
- `--output_dir`: Output directory for results (default: logs/synthetic_comparison)
- `--skip_brute_force`: Skip brute force optimization (useful for large character sets)

## Output Files

The script generates several output files in the specified directory:

1. **synthetic_matches.csv** - The generated match data
2. **ranking_comparison_summary.csv** - Summary of all method results
3. **ranking_methods_comparison.png** - Visual comparison plot

## Results Interpretation

### Inconsistency Score
- **Lower is better**: A score of 0.0 means perfect consistency (higher-ranked always beats lower-ranked)
- **Higher scores**: Indicate more inconsistencies in the ranking

### Performance Comparison
The script ranks methods by their inconsistency scores:
1. **Best**: Method with lowest inconsistency score
2. **Worst**: Method with highest inconsistency score

### Key Insights from Example Runs

#### 5 Characters, 3 Matches per Pair
```
Ranking by Performance (Best to Worst):
  1. SimulatedAnnealing   0.300000 (diff: +0.000000)  # Found optimal
  2. BruteForce           0.300000 (diff: +0.000000)  # Truly optimal
  3. WinCount             0.333333 (diff: +0.033333)  # Close to optimal
  4. Glicko2              0.366667 (diff: +0.066667)  # Moderate performance
  5. Elo                  0.500000 (diff: +0.200000)  # Worst performance
```

## Computational Complexity

### Brute Force Method
- **Time Complexity**: O(n! × m) where n = number of characters, m = number of matches
- **Space Complexity**: O(n!)
- **Practical Limit**: ~8-10 characters (due to factorial growth)

### Other Methods
- **Elo/Glicko2**: O(m) where m = number of matches
- **WinCount**: O(m)
- **Simulated Annealing**: O(iterations × n²) where iterations is typically 1000-5000

## Example Results Analysis

### When Simulated Annealing Performs Well
- **Scenario**: Complex win patterns with multiple local optima
- **Result**: Often finds the global optimum or very close to it
- **Example**: In our 5-character test, it found the optimal ranking

### When Traditional Methods Struggle
- **Elo**: Can get stuck in local optima, especially with limited data
- **Glicko2**: Better than Elo but still may not find global optimum
- **WinCount**: Simple but surprisingly effective for small datasets

### When WinCount Performs Well
- **Scenario**: Clear win patterns with minimal noise
- **Result**: Often close to optimal, especially for small datasets
- **Example**: In our 5-character test, it was only 0.033 points from optimal

## Recommendations

### For Small Datasets (≤6 characters)
1. **Use brute force** when possible to get the truly optimal ranking
2. **Simulated annealing** is often the best approximation method
3. **WinCount** can be surprisingly effective and computationally cheap

### For Larger Datasets (>6 characters)
1. **Skip brute force** due to computational constraints
2. **Use simulated annealing** as the primary optimization method
3. **Compare with Elo/Glicko2** for traditional baseline
4. **Consider WinCount** for quick approximations

## Technical Details

### Inconsistency Score Calculation
The inconsistency score measures the probability that a lower-ranked character beats a higher-ranked character:

```
inconsistency_score = Σ(prob_lower_beats_higher) / total_comparisons
```

### Simulated Annealing Parameters
- **Initial Temperature**: 100
- **Cooling Rate**: 0.99
- **Minimum Temperature**: 0.001
- **Maximum Iterations**: 5000
- **Neighbor Generation**: Random pair swapping

### Synthetic Data Generation
- **Random win probabilities** for each character pair
- **Configurable matches per pair** (default: 3)
- **Reproducible** with seed parameter
- **Balanced match distribution** across all pairs

## Future Enhancements

1. **Additional Ranking Methods**: Bradley-Terry, TrueSkill, etc.
2. **More Complex Synthetic Data**: Hierarchical skill levels, noise injection
3. **Statistical Significance Testing**: Bootstrap confidence intervals
4. **Scalability Analysis**: Performance vs dataset size
5. **Real-world Validation**: Testing on actual match data

## Dependencies

- `pandas`: Data manipulation and CSV I/O
- `numpy`: Numerical operations
- `matplotlib`: Plotting and visualization
- `itertools`: Permutation generation for brute force
- `collections.defaultdict`: Efficient data structures

## Example Usage Scenarios

### Quick Comparison (4 characters)
```bash
python synthetic_ranking_comparison.py --num_characters 4 --num_matches_per_pair 2
```

### Detailed Analysis (5 characters)
```bash
python synthetic_ranking_comparison.py --num_characters 5 --num_matches_per_pair 3
```

### Large Dataset (skip brute force)
```bash
python synthetic_ranking_comparison.py --num_characters 8 --num_matches_per_pair 2 --skip_brute_force
```

### Reproducible Research
```bash
python synthetic_ranking_comparison.py --seed 12345 --num_characters 5
``` 