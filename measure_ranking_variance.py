import os
import glob
import pandas as pd
import numpy as np
from scipy.spatial.distance import jensenshannon
import argparse

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'logs', 'results')
VARIANCES_DIR = os.path.join(os.path.dirname(__file__), 'logs', 'variances')

os.makedirs(VARIANCES_DIR, exist_ok=True)

def normalize(scores):
    total = sum(scores)
    if total == 0:
        return np.zeros_like(scores)
    return np.array(scores) / total

def total_variation_distance(p, q):
    return 0.5 * np.sum(np.abs(p - q))

def jensen_shannon_divergence(p, q):
    # scipy returns sqrt(JS), so square it for the divergence
    return jensenshannon(p, q, base=2) ** 2

def rank_edit_distance(ranking1, ranking2, players1, players2):
    """
    Calculate rank edit distance between two rankings.
    This is the sum of absolute differences between rankings of characters.
    
    Args:
        ranking1: List of ratings/scores for first ranking
        ranking2: List of ratings/scores for second ranking
        players1: List of player names for first ranking
        players2: List of player names for second ranking
    
    Returns:
        float: Sum of absolute rank differences
    """
    # Create dictionaries mapping players to their ranks
    # Sort by rating/score in descending order to get ranks
    rank1_dict = {}
    rank2_dict = {}
    
    # Create (rating, player) pairs and sort by rating descending
    pairs1 = list(zip(ranking1, players1))
    pairs1.sort(key=lambda x: x[0], reverse=True)
    
    pairs2 = list(zip(ranking2, players2))
    pairs2.sort(key=lambda x: x[0], reverse=True)
    
    # Assign ranks (1-based)
    for rank, (_, player) in enumerate(pairs1, 1):
        rank1_dict[player] = rank
    
    for rank, (_, player) in enumerate(pairs2, 1):
        rank2_dict[player] = rank
    
    # Calculate sum of absolute rank differences
    total_distance = 0
    for player in players1:
        if player in rank2_dict:
            total_distance += abs(rank1_dict[player] - rank2_dict[player])
    
    return total_distance

def extract_normalized_rankings(csv_path, method):
    df = pd.read_csv(csv_path)
    df = df[df['Model'] == method]
    # Sort by Player for consistent order
    df = df.sort_values('Player')
    players = df['Player'].tolist()
    ratings = df['Rating'].to_numpy(dtype=float)
    norm_ratings = normalize(ratings)
    return players, norm_ratings

def get_ranking_file(run_dir):
    files = glob.glob(os.path.join(RESULTS_DIR, run_dir, '*_rankings_*.csv'))
    if not files:
        raise FileNotFoundError(f"No ranking CSV found in {run_dir}")
    return files[0]

def short_id(run):
    return run[-12:]

def main():
    parser = argparse.ArgumentParser(description="Measure variance in normalized ELO, Glicko2, and WinCount rankings between two runs.")
    parser.add_argument('--run1', required=True, help='First run directory name (under logs/results)')
    parser.add_argument('--run2', required=True, help='Second run directory name (under logs/results)')
    args = parser.parse_args()

    run_ids = [args.run1, args.run2]
    methods = ['Elo', 'Glicko2', 'WinCount']
    all_rankings = {m: [] for m in methods}
    all_players = {m: [] for m in methods}

    for run in run_ids:
        ranking_file = get_ranking_file(run)
        for method in methods:
            players, norm_scores = extract_normalized_rankings(ranking_file, method)
            # Skip methods that don't have data (e.g., WinCount in older results)
            if len(players) == 0:
                print(f"Warning: No data found for {method} in {run}, skipping...")
                continue
            all_rankings[method].append(norm_scores)
            all_players[method].append(players)

    results = []
    for method in methods:
        # Only process methods that have data from both runs
        if len(all_rankings[method]) == 2:
            p, q = all_rankings[method]
            players1, players2 = all_players[method]
            
            # Find intersection of characters present in both runs
            common_players = set(players1) & set(players2)
            if len(common_players) == 0:
                print(f"Warning: No common characters found for {method}, skipping...")
                continue
                
            # Create player-to-index mappings
            player_to_idx1 = {player: idx for idx, player in enumerate(players1)}
            player_to_idx2 = {player: idx for idx, player in enumerate(players2)}
            
            # Extract ratings for common players only
            p_common = np.array([p[player_to_idx1[player]] for player in common_players])
            q_common = np.array([q[player_to_idx2[player]] for player in common_players])
            
            # Re-normalize the common player ratings
            p_common = normalize(p_common)
            q_common = normalize(q_common)
            
            tvd = total_variation_distance(p_common, q_common)
            jsd = jensen_shannon_divergence(p_common, q_common)
            red = rank_edit_distance(p_common, q_common, list(common_players), list(common_players))
            
            results.append({
                'method': method,
                'run1': run_ids[0],
                'run2': run_ids[1],
                'common_players': len(common_players),
                'tvd': tvd,
                'jsd': jsd,
                'rank_edit_distance': red
            })
            print(f"{method} - Common players: {len(common_players)}")
            print(f"{method} - TVD: {tvd:.4f}")
            print(f"{method} - JSD: {jsd:.4f}")
            print(f"{method} - Rank Edit Distance: {red:.0f}\n")
        else:
            print(f"Skipping {method} - data not available from both runs")

    out_csv = os.path.join(
        VARIANCES_DIR,
        f"ranking_variance_{short_id(run_ids[0])}_vs_{short_id(run_ids[1])}.csv"
    )
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"Saved variance results to {out_csv}")

if __name__ == '__main__':
    main() 