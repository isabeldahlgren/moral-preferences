#!/usr/bin/env python3
"""
Synthetic Ranking Comparison Script

This script generates synthetic match data with a small number of characters and compares
the optimality of different ranking methods (Elo, Glicko2, WinCount, Simulated Annealing)
against brute force optimization with respect to inconsistency score.

The brute force method tests all possible permutations of character rankings and computes
inconsistency scores to find the truly optimal ranking.
"""

import argparse
import os
import sys
import random
import math
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import List, Tuple, Dict, Any
import time

# Add the current directory to the path so we can import from other modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from produce_rankings import (
    calculate_inconsistency_score, 
    csv_to_matches,
    EloRatingSystem,
    Glicko2RatingSystem,
    count_wins_ranking
)
from optimize_rankings import simulated_annealing_ranking


def generate_synthetic_matches(
    characters: List[str], 
    num_matches_per_pair: int = 3,
    win_probability_matrix: Dict[Tuple[str, str], float] = None,
    seed: int = None
) -> List[Tuple[str, str, int]]:
    """
    Generate synthetic match data between characters.
    
    Args:
        characters: List of character names
        num_matches_per_pair: Number of matches to generate for each pair
        win_probability_matrix: Dict mapping (char1, char2) -> probability char1 wins
        seed: Random seed for reproducibility
        
    Returns:
        List of (character1, character2, result) tuples
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    matches = []
    
    # Generate matches for each pair of characters
    for i, char1 in enumerate(characters):
        for char2 in characters[i+1:]:
            # Determine win probability
            if win_probability_matrix and (char1, char2) in win_probability_matrix:
                p_char1_wins = win_probability_matrix[(char1, char2)]
            elif win_probability_matrix and (char2, char1) in win_probability_matrix:
                p_char1_wins = 1 - win_probability_matrix[(char2, char1)]
            else:
                # Default: random probability
                p_char1_wins = random.random()
            
            # Generate matches for this pair
            for _ in range(num_matches_per_pair):
                if random.random() < p_char1_wins:
                    result = 1  # char1 wins
                else:
                    result = 0  # char2 wins
                
                matches.append((char1, char2, result))
    
    return matches


def brute_force_optimal_ranking(matches: List[Tuple[str, str, int]]) -> Tuple[List[str], float]:
    """
    Find the optimal ranking by testing all possible permutations.
    
    Args:
        matches: List of (character1, character2, result) tuples
        
    Returns:
        Tuple of (optimal_ranking, optimal_score)
    """
    # Get unique characters
    characters = sorted(list(set([p for m in matches for p in [m[0], m[1]]])))
    n_characters = len(characters)
    
    print(f"Testing all {math.factorial(n_characters)} possible rankings for {n_characters} characters...")
    
    best_ranking = None
    best_score = float('inf')
    
    # Test all permutations
    for i, ranking in enumerate(itertools.permutations(characters)):
        score = calculate_inconsistency_score(matches, list(ranking))
        
        if score < best_score:
            best_score = score
            best_ranking = list(ranking)
            
        # Progress update for large numbers
        if n_characters > 5 and i % 1000 == 0:
            print(f"  Tested {i} permutations, best score so far: {best_score:.6f}")
    
    return best_ranking, best_score


def evaluate_ranking_methods(matches: List[Tuple[str, str, int]]) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate different ranking methods on the given matches.
    
    Args:
        matches: List of (character1, character2, result) tuples
        
    Returns:
        Dictionary with results for each method
    """
    results = {}
    
    # Get unique characters
    characters = sorted(list(set([p for m in matches for p in [m[0], m[1]]])))
    
    print(f"\nEvaluating ranking methods for {len(characters)} characters...")
    
    # 1. Elo Rating System
    print("  Running Elo rating system...")
    elo_system = EloRatingSystem(k_factor=32, initial_rating=1500)
    
    for char1, char2, result in matches:
        elo_system.update_ratings(char1, char2, result)
    
    elo_ranking_tuple, elo_ranking = elo_system.get_rankings()
    elo_score = calculate_inconsistency_score(matches, elo_ranking)
    
    results['Elo'] = {
        'ranking': elo_ranking,
        'score': elo_score,
        'ratings': dict(elo_ranking_tuple)
    }
    
    # 2. Glicko2 Rating System
    print("  Running Glicko2 rating system...")
    glicko_system = Glicko2RatingSystem(initial_rating=1500, initial_rd=350, initial_vol=0.06)
    
    # Group matches by player for Glicko2
    player_matches = defaultdict(list)
    for char1, char2, result in matches:
        player_matches[char1].append((char2, result))
        player_matches[char2].append((char1, 1 - result))
    
    # Update each player
    for player in characters:
        if player in player_matches:
            opponents = [opp for opp, _ in player_matches[player]]
            results_list = [res for _, res in player_matches[player]]
            glicko_system.update_player(player, opponents, results_list)
    
    glicko_ranking_tuple, glicko_ranking = glicko_system.get_rankings()
    glicko_score = calculate_inconsistency_score(matches, glicko_ranking)
    
    results['Glicko2'] = {
        'ranking': glicko_ranking,
        'score': glicko_score,
        'ratings': {player: rating for player, rating, _ in glicko_ranking_tuple}
    }
    
    # 3. Win Count Ranking
    print("  Running win count ranking...")
    win_ranking, win_ranking_list = count_wins_ranking(matches)
    win_score = calculate_inconsistency_score(matches, win_ranking_list)
    
    results['WinCount'] = {
        'ranking': win_ranking_list,
        'score': win_score,
        'win_counts': dict(win_ranking)
    }
    
    # 4. Simulated Annealing
    print("  Running simulated annealing...")
    sa_ranking, sa_score, _, _ = simulated_annealing_ranking(
        matches, 
        initial_temp=100, 
        cooling_rate=0.99, 
        min_temp=0.001, 
        max_iter=5000,
        seed=42
    )
    
    results['SimulatedAnnealing'] = {
        'ranking': sa_ranking,
        'score': sa_score
    }
    
    return results


def create_comparison_plots(results: Dict[str, Dict[str, Any]], brute_force_result: Tuple[List[str], float], 
                           output_dir: str = "logs/synthetic_comparison"):
    """
    Create comparison plots for the ranking methods.
    
    Args:
        results: Dictionary with results for each method
        brute_force_result: Tuple of (optimal_ranking, optimal_score)
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract scores for plotting
    methods = list(results.keys())
    scores = [results[method]['score'] for method in methods]
    
    # Add brute force result
    methods.append('BruteForce')
    scores.append(brute_force_result[1])
    
    # Create bar plot
    plt.figure(figsize=(12, 8))
    
    # Color coding: green for better (lower) scores
    colors = ['red' if score > brute_force_result[1] else 'green' for score in scores]
    
    bars = plt.bar(methods, scores, color=colors, alpha=0.7)
    plt.axhline(y=brute_force_result[1], color='black', linestyle='--', 
                label=f'Brute Force Optimal: {brute_force_result[1]:.6f}')
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{score:.6f}', ha='center', va='bottom', fontsize=10)
    
    plt.ylabel('Inconsistency Score (Lower is Better)')
    plt.title('Comparison of Ranking Methods vs Brute Force Optimal')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'ranking_methods_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plot saved to: {plot_path}")
    
    return plot_path


def print_detailed_results(results: Dict[str, Dict[str, Any]], brute_force_result: Tuple[List[str], float]):
    """
    Print detailed comparison results.
    
    Args:
        results: Dictionary with results for each method
        brute_force_result: Tuple of (optimal_ranking, optimal_score)
    """
    optimal_ranking, optimal_score = brute_force_result
    
    print("\n" + "="*80)
    print("DETAILED RANKING COMPARISON RESULTS")
    print("="*80)
    
    print(f"\nBrute Force Optimal Ranking:")
    print(f"  Ranking: {optimal_ranking}")
    print(f"  Inconsistency Score: {optimal_score:.6f}")
    
    print(f"\nMethod Comparisons:")
    print(f"{'Method':<20} {'Score':<12} {'Diff from Optimal':<18} {'Ranking'}")
    print("-" * 80)
    
    for method, result in results.items():
        score = result['score']
        diff = score - optimal_score
        ranking_str = " -> ".join(result['ranking'])
        
        print(f"{method:<20} {score:<12.6f} {diff:<18.6f} {ranking_str}")
    
    # Find best and worst methods
    method_scores = [(method, result['score']) for method, result in results.items()]
    method_scores.append(('BruteForce', optimal_score))
    method_scores.sort(key=lambda x: x[1])
    
    print(f"\nRanking by Performance (Best to Worst):")
    for i, (method, score) in enumerate(method_scores, 1):
        diff = score - optimal_score
        print(f"  {i}. {method:<20} {score:.6f} (diff: {diff:+.6f})")


def save_results_to_csv(results: Dict[str, Dict[str, Any]], brute_force_result: Tuple[List[str], float],
                       output_dir: str = "logs/synthetic_comparison"):
    """
    Save results to CSV files.
    
    Args:
        results: Dictionary with results for each method
        brute_force_result: Tuple of (optimal_ranking, optimal_score)
        output_dir: Directory to save files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create summary CSV
    summary_data = []
    optimal_ranking, optimal_score = brute_force_result
    
    for method, result in results.items():
        summary_data.append({
            'method': method,
            'inconsistency_score': result['score'],
            'difference_from_optimal': result['score'] - optimal_score,
            'ranking': ' -> '.join(result['ranking'])
        })
    
    # Add brute force result
    summary_data.append({
        'method': 'BruteForce',
        'inconsistency_score': optimal_score,
        'difference_from_optimal': 0.0,
        'ranking': ' -> '.join(optimal_ranking)
    })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_dir, 'ranking_comparison_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    
    print(f"Summary results saved to: {summary_path}")
    
    return summary_path


def main():
    """Main function to run the synthetic ranking comparison."""
    parser = argparse.ArgumentParser(description='Compare ranking methods on synthetic data')
    parser.add_argument('--num_characters', type=int, default=5, 
                       help='Number of characters to use (default: 5)')
    parser.add_argument('--num_matches_per_pair', type=int, default=3,
                       help='Number of matches per character pair (default: 3)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--output_dir', type=str, default='logs/synthetic_comparison',
                       help='Output directory for results (default: logs/synthetic_comparison)')
    parser.add_argument('--skip_brute_force', action='store_true',
                       help='Skip brute force optimization (useful for large character sets)')
    
    args = parser.parse_args()
    
    # Generate synthetic characters
    characters = [f"Character_{i+1}" for i in range(args.num_characters)]
    
    print(f"Generating synthetic match data for {len(characters)} characters...")
    print(f"Characters: {characters}")
    
    # Generate synthetic matches
    matches = generate_synthetic_matches(
        characters, 
        num_matches_per_pair=args.num_matches_per_pair,
        seed=args.seed
    )
    
    print(f"Generated {len(matches)} matches")
    
    # Save matches to CSV for reference
    os.makedirs(args.output_dir, exist_ok=True)
    matches_df = pd.DataFrame(matches, columns=['character1', 'character2', 'result'])
    matches_path = os.path.join(args.output_dir, 'synthetic_matches.csv')
    matches_df.to_csv(matches_path, index=False)
    print(f"Matches saved to: {matches_path}")
    
    # Evaluate ranking methods
    results = evaluate_ranking_methods(matches)
    
    # Run brute force optimization (if not skipped)
    if args.skip_brute_force:
        print("\nSkipping brute force optimization as requested.")
        brute_force_result = (None, float('inf'))
    else:
        print("\nRunning brute force optimization...")
        start_time = time.time()
        brute_force_result = brute_force_optimal_ranking(matches)
        end_time = time.time()
        print(f"Brute force completed in {end_time - start_time:.2f} seconds")
    
    # Create plots and save results
    if not args.skip_brute_force:
        create_comparison_plots(results, brute_force_result, args.output_dir)
        save_results_to_csv(results, brute_force_result, args.output_dir)
        print_detailed_results(results, brute_force_result)
    else:
        # Just print the method results without brute force comparison
        print("\nRanking Method Results:")
        for method, result in results.items():
            print(f"  {method}: {result['score']:.6f} - {' -> '.join(result['ranking'])}")
    
    print(f"\nAll results saved to: {args.output_dir}")


if __name__ == "__main__":
    main() 