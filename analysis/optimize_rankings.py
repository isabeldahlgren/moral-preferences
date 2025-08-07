#!/usr/bin/env python3
"""
Optimize character rankings using simulated annealing to minimize inconsistency score.

This script finds an optimal ranking of characters by minimizing the inconsistency score,
which measures the probability of lower-ranked characters beating higher-ranked ones.
"""

import argparse
import os
import sys
import random
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import List, Tuple, Dict, Any

# Add the current directory to the path so we can import from other modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from produce_rankings import calculate_inconsistency_score, csv_to_matches
from file_utils import generate_timestamp, generate_run_id


def simulated_annealing_ranking(
    matches: List[Tuple[str, str, int]], 
    initial_temp: float = 100, 
    cooling_rate: float = 0.99, 
    min_temp: float = 0.001, 
    max_iter: int = 10000,
    seed: int = None
) -> Tuple[List[str], float, List[float], List[float]]:
    """
    Find optimal ranking using simulated annealing to minimize inconsistency score.
    
    Args:
        matches: List of (character1, character2, result) tuples
        initial_temp: Initial temperature for simulated annealing
        cooling_rate: Rate at which temperature decreases
        min_temp: Minimum temperature to stop optimization
        max_iter: Maximum number of iterations
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (best_ranking, best_score, scores_history, temperatures)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Get unique characters from matches
    players = sorted(list(set([p for m in matches for p in [m[0], m[1]]])))
    n_players = len(players)
    
    print(f"Optimizing ranking for {n_players} characters using simulated annealing")
    print(f"Parameters: temp={initial_temp}, cooling_rate={cooling_rate}, max_iter={max_iter}")
    
    # Start with random ranking
    current_ranking = players.copy()
    random.shuffle(current_ranking)
    current_score = calculate_inconsistency_score(matches, current_ranking)
    
    best_ranking = current_ranking.copy()
    best_score = current_score
    
    temp = initial_temp
    iteration = 0
    
    print(f"Initial random ranking score: {current_score:.6f}")
    
    # Track progress for plotting
    scores_history = [current_score]
    temperatures = [temp]
    
    while temp > min_temp and iteration < max_iter:
        # Generate neighbor by swapping two random players
        new_ranking = current_ranking.copy()
        i, j = random.sample(range(n_players), 2)
        new_ranking[i], new_ranking[j] = new_ranking[j], new_ranking[i]
        
        new_score = calculate_inconsistency_score(matches, new_ranking)
        
        # Accept or reject the move based on Metropolis criterion
        delta_e = new_score - current_score
        if delta_e < 0 or random.random() < math.exp(-delta_e / temp):
            current_ranking = new_ranking
            current_score = new_score
            
            if new_score < best_score:
                best_ranking = current_ranking.copy()
                best_score = new_score
                print(f"Iteration {iteration}: New best score: {best_score:.6f}")
        
        temp *= cooling_rate
        iteration += 1
        
        # Track progress every 100 iterations
        if iteration % 100 == 0:
            scores_history.append(current_score)
            temperatures.append(temp)
    
    print(f"\nOptimization complete!")
    print(f"Final best score: {best_score:.6f}")
    print(f"Total iterations: {iteration}")
    print(f"Final temperature: {temp:.6f}")
    
    return best_ranking, best_score, scores_history, temperatures


def plot_optimization_progress(scores_history: List[float], temperatures: List[float], 
                             output_dir: str, model_string: str):
    """Plot the optimization progress."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot score progression
    iterations = range(0, len(scores_history) * 100, 100)
    ax1.plot(iterations, scores_history, 'b-', linewidth=1)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Inconsistency Score')
    ax1.set_title('Optimization Progress - Score')
    ax1.grid(True, alpha=0.3)
    
    # Plot temperature progression
    ax2.plot(iterations, temperatures, 'r-', linewidth=1)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Temperature')
    ax2.set_title('Optimization Progress - Temperature')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_filename = f"optimization_progress_{model_string}_{generate_timestamp()}.png"
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Optimization progress plot saved: {plot_path}")
    return plot_path


def save_optimal_ranking(ranking: List[str], score: float, output_dir: str, 
                        model_string: str, csv_files: List[str]) -> str:
    """Save the optimal ranking to a CSV file."""
    # Create ranking with positions
    ranking_data = []
    for i, character in enumerate(ranking):
        ranking_data.append({
            'position': i + 1,
            'character': character,
            'score': len(ranking) - i  # Higher position = higher score
        })
    
    df = pd.DataFrame(ranking_data)
    
    # Generate filename
    timestamp = generate_timestamp()
    filename = f"optimal_ranking_{model_string}_{timestamp}.csv"
    filepath = os.path.join(output_dir, filename)
    
    # Save to CSV
    df.to_csv(filepath, index=False)
    
    # Also save metadata
    metadata = {
        'inconsistency_score': score,
        'num_characters': len(ranking),
        'source_csv_files': ', '.join(csv_files),
        'model_string': model_string,
        'timestamp': timestamp
    }
    
    metadata_filename = f"optimal_ranking_metadata_{model_string}_{timestamp}.txt"
    metadata_path = os.path.join(output_dir, metadata_filename)
    
    with open(metadata_path, 'w') as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Optimal ranking saved: {filepath}")
    print(f"Metadata saved: {metadata_path}")
    
    return filepath


def print_ranking_summary(ranking: List[str], score: float, matches: List[Tuple[str, str, int]], csv_files: List[str]):
    """Print a summary of the optimal ranking."""
    print("\n" + "="*60)
    print("OPTIMAL RANKING SUMMARY")
    print("="*60)
    print(f"Inconsistency Score: {score:.6f}")
    print(f"Number of Characters: {len(ranking)}")
    print(f"Number of Matches: {len(matches)}")
    print(f"Source CSV files: {', '.join(csv_files)}")
    print("\nRanking (best to worst):")
    print("-" * 40)
    
    for i, character in enumerate(ranking):
        print(f"{i+1:2d}. {character}")
    
    # Calculate some statistics
    unique_characters = set([p for m in matches for p in [m[0], m[1]]])
    print(f"\nStatistics:")
    print(f"  Characters in matches: {len(unique_characters)}")
    print(f"  Characters in ranking: {len(ranking)}")
    
    if len(unique_characters) != len(ranking):
        missing = unique_characters - set(ranking)
        extra = set(ranking) - unique_characters
        if missing:
            print(f"  Missing from ranking: {missing}")
        if extra:
            print(f"  Extra in ranking: {extra}")


def optimize_rankings_from_csv_files(
    csv_files: List[str],
    output_dir: str = "logs/optimized_rankings",
    model_string: str = "unknown",
    initial_temp: float = 100,
    cooling_rate: float = 0.99,
    min_temp: float = 0.001,
    max_iter: int = 10000,
    seed: int = None,
    save_plots: bool = True
) -> Dict[str, Any]:
    """
    Optimize rankings from multiple CSV files using simulated annealing.
    
    Args:
        csv_files: List of paths to CSV files with matches
        output_dir: Directory to save results
        model_string: String identifier for the model
        initial_temp: Initial temperature for simulated annealing
        cooling_rate: Rate at which temperature decreases
        min_temp: Minimum temperature to stop optimization
        max_iter: Maximum number of iterations
        seed: Random seed for reproducibility
        save_plots: Whether to save optimization progress plots
        
    Returns:
        Dictionary with optimization results and file paths
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load matches from all CSV files
    all_matches = []
    for csv_file in csv_files:
        print(f"Loading matches from: {csv_file}")
        matches = csv_to_matches(csv_file)
        print(f"Loaded {len(matches)} matches from {csv_file}")
        all_matches.extend(matches)
    
    print(f"Total matches loaded: {len(all_matches)}")
    
    # Run simulated annealing optimization on combined data
    best_ranking, best_score, scores_history, temperatures = simulated_annealing_ranking(
        matches=all_matches,
        initial_temp=initial_temp,
        cooling_rate=cooling_rate,
        min_temp=min_temp,
        max_iter=max_iter,
        seed=seed
    )
    
    # Print summary
    print_ranking_summary(best_ranking, best_score, all_matches, csv_files)
    
    # Save results
    ranking_file = save_optimal_ranking(
        ranking=best_ranking,
        score=best_score,
        output_dir=output_dir,
        model_string=model_string,
        csv_files=csv_files
    )
    
    results = {
        'ranking': best_ranking,
        'score': best_score,
        'ranking_file': ranking_file,
        'num_matches': len(all_matches),
        'num_characters': len(best_ranking),
        'csv_files': csv_files
    }
    
    # Save plots if requested
    if save_plots:
        plot_file = plot_optimization_progress(
            scores_history, temperatures, output_dir, model_string
        )
        results['plot_file'] = plot_file
    
    return results


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Optimize character rankings using simulated annealing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python optimize_rankings.py --csv logs/csv-files/matches_train.csv --model claude-3-5-sonnet
  python optimize_rankings.py --csv train.csv test.csv --model gpt-4o-mini --max-iter 5000
  python optimize_rankings.py --csv matches1.csv matches2.csv --model test --initial-temp 200 --cooling-rate 0.995
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--csv", 
        nargs='+',
        required=True,
        help="Path(s) to CSV file(s) with matches (must have character1, character2, result columns)"
    )
    
    parser.add_argument(
        "--model", 
        required=True,
        help="Model string identifier (e.g., claude-3-5-sonnet, gpt-4o-mini)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--output-dir", 
        default="logs/optimized_rankings",
        help="Directory to save optimization results (default: logs/optimized_rankings)"
    )
    
    parser.add_argument(
        "--initial-temp", 
        type=float, 
        default=100,
        help="Initial temperature for simulated annealing (default: 100)"
    )
    
    parser.add_argument(
        "--cooling-rate", 
        type=float, 
        default=0.99,
        help="Cooling rate for simulated annealing (default: 0.99)"
    )
    
    parser.add_argument(
        "--min-temp", 
        type=float, 
        default=0.001,
        help="Minimum temperature to stop optimization (default: 0.001)"
    )
    
    parser.add_argument(
        "--max-iter", 
        type=int, 
        default=10000,
        help="Maximum number of iterations (default: 10000)"
    )
    
    parser.add_argument(
        "--seed", 
        type=int,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--no-plots", 
        action="store_true",
        help="Don't save optimization progress plots"
    )
    
    args = parser.parse_args()
    
    # Validate input files
    for csv_file in args.csv:
        if not os.path.exists(csv_file):
            print(f"❌ Error: CSV file not found: {csv_file}")
            return 1
    
    try:
        results = optimize_rankings_from_csv_files(
            csv_files=args.csv,
            output_dir=args.output_dir,
            model_string=args.model,
            initial_temp=args.initial_temp,
            cooling_rate=args.cooling_rate,
            min_temp=args.min_temp,
            max_iter=args.max_iter,
            seed=args.seed,
            save_plots=not args.no_plots
        )
        
        print(f"\n🎉 Optimization completed successfully!")
        print(f"Results saved in: {args.output_dir}")
        print(f"Optimal ranking file: {results['ranking_file']}")
        
    except Exception as e:
        print(f"\n❌ Error during optimization: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 