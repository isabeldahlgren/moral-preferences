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

def main():
    parser = argparse.ArgumentParser(description="Measure variance in normalized ELO and Glicko2 rankings between two runs.")
    parser.add_argument('--run1', required=True, help='First run directory name (under logs/results)')
    parser.add_argument('--run2', required=True, help='Second run directory name (under logs/results)')
    args = parser.parse_args()

    run_ids = [args.run1, args.run2]
    methods = ['Elo', 'Glicko2']
    all_rankings = {m: [] for m in methods}
    all_players = None

    for run in run_ids:
        ranking_file = get_ranking_file(run)
        for method in methods:
            players, norm_scores = extract_normalized_rankings(ranking_file, method)
            if all_players is None:
                all_players = players
            elif players != all_players:
                raise ValueError(f"Player mismatch in {run}. All runs must have the same characters.")
            all_rankings[method].append(norm_scores)

    results = []
    for method in methods:
        p, q = all_rankings[method]
        tvd = total_variation_distance(p, q)
        jsd = jensen_shannon_divergence(p, q)
        results.append({
            'method': method,
            'run1': run_ids[0],
            'run2': run_ids[1],
            'tvd': tvd,
            'jsd': jsd
        })
        print(f"{method} - TVD: {tvd:.4f}")
        print(f"{method} - JSD: {jsd:.4f}\n")

    out_csv = os.path.join(VARIANCES_DIR, f'ranking_variance_{run_ids[0]}_vs_{run_ids[1]}.csv')
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"Saved variance results to {out_csv}")

if __name__ == '__main__':
    main() 