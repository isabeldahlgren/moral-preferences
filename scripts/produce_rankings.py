# %%

import pandas as pd
import numpy as np
import math
import argparse
import os
from collections import defaultdict
import random
import matplotlib.pyplot as plt
from scipy.stats import norm
from typing import Dict, Any
from scipy.optimize import differential_evolution, minimize

# Import file utilities for consistent naming
from file_utils import (
    generate_timestamp,
    generate_run_id
)

# All file paths are provided via CLI arguments. No hardcoded paths.

# Define utility functions for ranking methods


def normalize_ranking(ranking):
    """
    Normalize a ranking (list of (player, score)) to a probability distribution.
    Returns a dict mapping player to normalized score.
    """
    total = sum(score for _, score in ranking)
    return {player: score / total for player, score in ranking}


def plot_normalized_ranking(ranking, title="Normalized Ranking", top_n=None):
    """
    Plot a normalized bar chart of a ranking (list of (player, score) pairs).

    Parameters:
    - ranking: list of (player, score)
    - title: title of the plot
    - top_n: if specified, only show the top N players
    """
    # Normalize scores
    dist = normalize_ranking(ranking)

    # Sort and optionally trim to top_n
    sorted_items = sorted(dist.items(), key=lambda x: x[1], reverse=True)
    if top_n:
        sorted_items = sorted_items[:top_n]

    players = [player for player, _ in sorted_items]
    scores = [score for _, score in sorted_items]

    # Plot
    plt.figure(figsize=(10, 6))  # Fixed height for better proportions
    # reverse to get top on top
    bars = plt.barh(players[::-1], scores[::-1], color="skyblue", height=0.6)
    plt.xlabel("Normalized Score")
    plt.title(title)
    plt.grid(axis='x', alpha=0.3)  # Add subtle grid for better readability
    plt.tight_layout()


def calculate_inconsistency_score(matches: list, ranking_order: list) -> float:
    """
    Calculate inconsistency score as probability of lower-ranked beating higher-ranked.

    Parameters:
    matches: list of (character1, character2, result) tuples
    ranking_order: list of players in ranking order (best to worst)

    Returns:
    float: inconsistency score (lower is better)
    """
    # Create ranking position lookup
    rank_position = {player: i for i, player in enumerate(ranking_order)}

    # Count wins and total games for each player pair
    wins = defaultdict(lambda: defaultdict(int))
    total_games = defaultdict(lambda: defaultdict(int))

    for character1, character2, result in matches:
        if character1 in rank_position and character2 in rank_position:
            total_games[character1][character2] += 1
            total_games[character2][character1] += 1

            if result == 1:  # character1 wins
                wins[character1][character2] += 1
            else:  # character2 wins
                wins[character2][character1] += 1

    inconsistency_prob = 0
    comparisons = 0

    # For each pair where one player is ranked higher than the other
    for i, higher_ranked_player in enumerate(ranking_order):
        for lower_ranked_player in ranking_order[i + 1 :]:
            # Check if these players have played against each other
            if total_games[higher_ranked_player][lower_ranked_player] > 0:
                # Calculate probability that lower ranked player beats higher ranked
                lower_beats_higher = wins[lower_ranked_player][higher_ranked_player]
                total_between_pair = total_games[higher_ranked_player][
                    lower_ranked_player
                ]
                prob_lower_wins = lower_beats_higher / total_between_pair

                inconsistency_prob += prob_lower_wins
                comparisons += 1

    # Normalize over the number of comparisons
    if comparisons > 0:
        inconsistency_score = inconsistency_prob / comparisons
    else:
        inconsistency_score = 0

    return inconsistency_score


def csv_to_matches(
    csv_file,
    player_col_a="character1",
    player_col_b="character2",
    result_col="result",
):
    """
    This function reads a CSV file and extracts matches into a list of tuples.
    """

    # Read CSV
    if isinstance(csv_file, str) and "\n" in csv_file:
        # CSV content as string
        from io import StringIO

        df = pd.read_csv(StringIO(csv_file))
    else:
        # CSV file path
        df = pd.read_csv(csv_file)

    # Process matches and create duplicated matches list
    matches = []

    for _, row in df.iterrows():
        character1 = row[player_col_a]
        character2 = row[player_col_b]
        result = row[result_col]
        matches.append((character1, character2, result))

    return matches


# %%
# Define ranking methods which score players


class EloRatingSystem:
    def __init__(self, k_factor=32, initial_rating=1500):
        self.k_factor = k_factor
        self.initial_rating = initial_rating
        self.ratings = defaultdict(lambda: initial_rating)

        # Track statistics for confidence intervals
        self.games_played = defaultdict(int)
        self.rating_history = defaultdict(list)
        self.match_results = []  # Store all matches for analysis
        self.performance_variance = defaultdict(list)

    def expected_score(self, rating_a, rating_b):
        """Calculate expected score for player A against player B"""
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def update_ratings(self, character1, character2, result):
        """
        Update ratings based on match result
        result: 1 if character1 wins, 0 if character2 wins, 0.5 for draw
        """
        # Store match for confidence interval calculations
        self.match_results.append((character1, character2, result))

        rating_a = self.ratings[character1]
        rating_b = self.ratings[character2]

        expected_a = self.expected_score(rating_a, rating_b)
        expected_b = 1 - expected_a

        # Store rating history
        self.rating_history[character1].append(rating_a)
        self.rating_history[character2].append(rating_b)

        # Store performance variance data
        self.performance_variance[character1].append((result - expected_a) ** 2)
        self.performance_variance[character2].append(((1 - result) - expected_b) ** 2)

        # Update ratings
        self.ratings[character1] = rating_a + self.k_factor * (result - expected_a)
        self.ratings[character2] = rating_b + self.k_factor * (
            (1 - result) - expected_b
        )

        # Update games played
        self.games_played[character1] += 1
        self.games_played[character2] += 1

    def _calculate_rating_reliability(self, player):
        """
        Calculate rating reliability based on Elo theory
        Returns standard deviation of the rating estimate
        """
        games = self.games_played[player]

        if games == 0:
            # No games played, maximum uncertainty
            return 200  # Standard initial uncertainty in Elo systems

        # Method 1: Based on number of games (Glickman's approach)
        # Standard error decreases with square root of number of games
        base_uncertainty = 200  # Initial rating deviation
        min_uncertainty = 50  # Minimum uncertainty after many games

        # Uncertainty decreases with more games
        uncertainty_games = max(
            min_uncertainty, base_uncertainty / math.sqrt(games + 1)
        )

        # Method 2: Based on performance variance
        if len(self.performance_variance[player]) > 1:
            performance_var = np.var(self.performance_variance[player])
            # Scale performance variance to rating scale
            uncertainty_performance = (
                math.sqrt(performance_var) * 400 / math.sqrt(games)
            )
        else:
            uncertainty_performance = uncertainty_games

        # Method 3: Based on rating stability
        if len(self.rating_history[player]) > 3:
            recent_ratings = self.rating_history[player][-10:]  # Last 10 ratings
            rating_std = (
                np.std(recent_ratings) if len(recent_ratings) > 1 else uncertainty_games
            )
            uncertainty_stability = rating_std / math.sqrt(len(recent_ratings))
        else:
            uncertainty_stability = uncertainty_games

        # Combine different uncertainty estimates
        combined_uncertainty = (
            uncertainty_games + uncertainty_performance + uncertainty_stability
        ) / 3

        return max(min_uncertainty, min(combined_uncertainty, base_uncertainty))

    def _calculate_glicko_style_rd(self, player):
        """
        Calculate Rating Deviation similar to Glicko system
        """
        games = self.games_played[player]

        if games == 0:
            return 350  # Initial RD in Glicko

        # Base RD calculation
        initial_rd = 350
        min_rd = 30

        # RD decreases with more games
        rd = max(min_rd, initial_rd / math.sqrt(1 + games * 0.1))

        return rd

    def get_confidence_interval(self, player, confidence_level=0.95):
        """
        Calculate confidence interval for a player's rating
        """
        rating = self.ratings[player]

        # Calculate standard error using multiple methods
        method1_se = self._calculate_rating_reliability(player)
        method2_rd = self._calculate_glicko_style_rd(player)

        # Use average of methods for robustness
        standard_error = (method1_se + method2_rd) / 2

        # Calculate confidence interval
        alpha = 1 - confidence_level
        z_score = norm.ppf(1 - alpha / 2)

        margin_error = z_score * standard_error
        lower_bound = rating - margin_error
        upper_bound = rating + margin_error

        return (lower_bound, upper_bound, standard_error)

    def get_rankings(self):
        """Return sorted list of (player, rating) tuples"""
        ranking = sorted(self.ratings.items(), key=lambda x: x[1], reverse=True)
        ranking_list = [player for player, _ in ranking]
        return ranking, ranking_list

    def get_rankings_with_ci(self, confidence_level=0.95):
        """
        Return sorted list of (player, rating, lower_bound, upper_bound, standard_error, games) tuples
        """
        rankings = []
        for player, rating in sorted(
            self.ratings.items(), key=lambda x: x[1], reverse=True
        ):
            lower, upper, se = self.get_confidence_interval(player, confidence_level)
            games = self.games_played[player]
            rankings.append((player, rating, lower, upper, se, games))

        return rankings


class Elo2KRatingSystem:
    def __init__(
        self,
        initial_rating=1500,
        initial_rd=350,
        k_min=16,
        k_max=32,
        rd_min=30,
        rd_max=350,
        c=15.8,
        time_factor=0.0,
    ):
        """
        Initialize Elo2K rating system

        Parameters:
        - initial_rating: Starting rating for new players
        - initial_rd: Initial rating deviation (uncertainty)
        - k_min: Minimum K-factor
        - k_max: Maximum K-factor
        - rd_min: Minimum rating deviation
        - rd_max: Maximum rating deviation
        - c: Rating deviation increase per time period (Glicko-style)
        - time_factor: Time decay factor (0 = no time decay)
        """
        self.initial_rating = initial_rating
        self.initial_rd = initial_rd
        self.k_min = k_min
        self.k_max = k_max
        self.rd_min = rd_min
        self.rd_max = rd_max
        self.c = c
        self.time_factor = time_factor

        # Player data
        self.ratings = defaultdict(lambda: initial_rating)
        self.rating_deviations = defaultdict(lambda: initial_rd)
        self.games_played = defaultdict(int)
        self.last_game_time = defaultdict(int)
        self.rating_history = defaultdict(list)
        self.match_results = []

        # Time tracking
        self.current_time = 0

    def _g(self, rd):
        """Glicko-style g function for rating deviation scaling"""
        return 1 / math.sqrt(1 + 3 * (rd / math.pi) ** 2)

    def _expected_score(self, rating_a, rating_b, rd_b):
        """Calculate expected score with rating deviation consideration"""
        g_rd = self._g(rd_b)
        return 1 / (1 + 10 ** (g_rd * (rating_b - rating_a) / 400))

    def _calculate_k_factor(self, player):
        """Calculate dynamic K-factor based on rating deviation and games played"""
        rd = self.rating_deviations[player]
        games = self.games_played[player]

        # K-factor based on rating deviation (higher RD = higher K)
        k_rd = self.k_min + (self.k_max - self.k_min) * (rd - self.rd_min) / (
            self.rd_max - self.rd_min
        )
        k_rd = max(self.k_min, min(self.k_max, k_rd))

        # Additional scaling for new players
        if games < 20:
            k_rd *= 1.5
        elif games < 50:
            k_rd *= 1.2

        return k_rd

    def _update_rating_deviation(
        self, player, opponent_rd, expected_score, actual_score
    ):
        """Update rating deviation based on game outcome"""
        current_rd = self.rating_deviations[player]

        # Time-based RD increase (if time_factor > 0)
        if self.time_factor > 0:
            time_since_last = self.current_time - self.last_game_time[player]
            rd_increase = self.c * math.sqrt(time_since_last * self.time_factor)
            current_rd = min(self.rd_max, math.sqrt(current_rd**2 + rd_increase**2))

        # Game-based RD decrease (playing reduces uncertainty)
        g_opponent = self._g(opponent_rd)
        d_squared = 1 / (g_opponent**2 * expected_score * (1 - expected_score))

        # Update RD
        new_rd_squared = 1 / (1 / current_rd**2 + 1 / d_squared)
        new_rd = math.sqrt(new_rd_squared)

        # Ensure RD stays within bounds
        self.rating_deviations[player] = max(self.rd_min, min(self.rd_max, new_rd))

    def update_ratings(self, character1, character2, result, time_period=None):
        """
        Update ratings based on match result
        result: 1 if character1 wins, 0 if character2 wins, 0.5 for draw
        time_period: Optional time period for time-based RD updates
        """
        if time_period is not None:
            self.current_time = time_period
        else:
            self.current_time += 1

        # Store match for analysis
        self.match_results.append((character1, character2, result, self.current_time))

        # Get current ratings and RDs
        rating_a = self.ratings[character1]
        rating_b = self.ratings[character2]
        rd_a = self.rating_deviations[character1]
        rd_b = self.rating_deviations[character2]

        # Store rating history
        self.rating_history[character1].append(rating_a)
        self.rating_history[character2].append(rating_b)

        # Calculate expected scores
        expected_a = self._expected_score(rating_a, rating_b, rd_b)
        expected_b = self._expected_score(rating_b, rating_a, rd_a)

        # Calculate K-factors
        k_a = self._calculate_k_factor(character1)
        k_b = self._calculate_k_factor(character2)

        # Update rating deviations first
        self._update_rating_deviation(character1, rd_b, expected_a, result)
        self._update_rating_deviation(character2, rd_a, expected_b, 1 - result)

        # Update ratings
        self.ratings[character1] = rating_a + k_a * (result - expected_a)
        self.ratings[character2] = rating_b + k_b * ((1 - result) - expected_b)

        # Update tracking variables
        self.games_played[character1] += 1
        self.games_played[character2] += 1
        self.last_game_time[character1] = self.current_time
        self.last_game_time[character2] = self.current_time

    def get_confidence_interval(self, player, confidence_level=0.95):
        """
        Calculate confidence interval for a player's rating using RD
        """
        rating = self.ratings[player]
        rd = self.rating_deviations[player]

        # Calculate confidence interval using rating deviation
        alpha = 1 - confidence_level
        z_score = norm.ppf(1 - alpha / 2)

        margin_error = z_score * rd
        lower_bound = rating - margin_error
        upper_bound = rating + margin_error

        return (lower_bound, upper_bound, rd)

    def get_rankings(self):
        """Return sorted list of (player, rating) tuples"""
        ranking = sorted(self.ratings.items(), key=lambda x: x[1], reverse=True)
        ranking_list = [player for player, _ in ranking]
        return ranking, ranking_list

    def get_rankings_with_ci(self, confidence_level=0.95):
        """
        Return sorted list of (player, rating, lower_bound, upper_bound, rd, games) tuples
        """
        rankings = []
        for player, rating in sorted(
            self.ratings.items(), key=lambda x: x[1], reverse=True
        ):
            lower, upper, rd = self.get_confidence_interval(player, confidence_level)
            games = self.games_played[player]
            rankings.append((player, rating, lower, upper, rd, games))

        return rankings


class Glicko2RatingSystem:
    def __init__(self, initial_rating=1500, initial_rd=350, initial_vol=0.06):
        self.initial_rating = initial_rating
        self.initial_rd = initial_rd  # Rating deviation
        self.initial_vol = initial_vol  # Volatility
        self.tau = 0.5  # System constant
        self.ratings = {}
        self.rd = {}
        self.vol = {}

        # Track additional statistics for enhanced analysis
        self.games_played = defaultdict(int)
        self.rating_history = defaultdict(list)
        self.rd_history = defaultdict(list)
        self.last_update_period = defaultdict(int)
        self.current_period = 0

    def _g(self, rd):
        """g function from Glicko-2 paper"""
        return 1 / math.sqrt(1 + 3 * rd**2 / (math.pi**2))

    def _E(self, mu, mu_j, rd_j):
        """Expected outcome function"""
        return 1 / (1 + math.exp(-self._g(rd_j) * (mu - mu_j)))

    def _f(self, x, delta, rd, v, a, tau):
        """Function f from step 5 of Glicko-2"""
        ex = math.exp(x)
        return (ex * (delta**2 - rd**2 - v - ex) / (2 * (rd**2 + v + ex) ** 2)) - (
            x - a
        ) / tau**2

    def update_player(self, player, opponents, results):
        """Update a single player's rating based on multiple games"""
        if player not in self.ratings:
            self.ratings[player] = self.initial_rating
            self.rd[player] = self.initial_rd
            self.vol[player] = self.initial_vol

        # Store rating history before update
        self.rating_history[player].append(self.ratings[player])
        self.rd_history[player].append(self.rd[player])

        # Convert to Glicko-2 scale
        mu = (self.ratings[player] - 1500) / 173.7178
        phi = self.rd[player] / 173.7178
        sigma = self.vol[player]

        # Step 2: Calculate v and delta
        v = 0
        delta = 0

        for opp, result in zip(opponents, results):
            if opp not in self.ratings:
                self.ratings[opp] = self.initial_rating
                self.rd[opp] = self.initial_rd
                self.vol[opp] = self.initial_vol

            mu_j = (self.ratings[opp] - 1500) / 173.7178
            phi_j = self.rd[opp] / 173.7178

            g_phi_j = self._g(phi_j)
            E_mu_mu_j = self._E(mu, mu_j, phi_j)

            v += g_phi_j**2 * E_mu_mu_j * (1 - E_mu_mu_j)
            delta += g_phi_j * (result - E_mu_mu_j)

        v = 1 / v if v > 0 else float("inf")

        # Step 3: Calculate new volatility
        a = math.log(sigma**2)
        if delta**2 > phi**2 + v:
            B = math.log(delta**2 - phi**2 - v)
        else:
            k = 1
            while self._f(a - k * self.tau, delta, phi, v, a, self.tau) < 0:
                k += 1
            B = a - k * self.tau

        # Approximate solution using iteration
        fA = self._f(a, delta, phi, v, a, self.tau)
        fB = self._f(B, delta, phi, v, a, self.tau)

        while abs(B - a) > 1e-6:
            C = a + (a - B) * fA / (fB - fA)
            fC = self._f(C, delta, phi, v, a, self.tau)
            if fC * fB < 0:
                a = B
                fA = fB
            else:
                fA = fA / 2
            B = C
            fB = fC

        new_sigma = math.exp(a / 2)

        # Step 4: Update rating deviation
        phi_star = math.sqrt(phi**2 + new_sigma**2)

        # Step 5: Update rating and RD
        new_phi = 1 / math.sqrt(1 / phi_star**2 + 1 / v)
        new_mu = mu + new_phi**2 * delta

        # Convert back to original scale
        self.ratings[player] = new_mu * 173.7178 + 1500
        self.rd[player] = new_phi * 173.7178
        self.vol[player] = new_sigma

        # Update tracking statistics
        self.games_played[player] += len(results)
        self.last_update_period[player] = self.current_period

    def advance_period(self):
        """Advance to next rating period (increases RD for inactive players)"""
        self.current_period += 1

        # Increase RD for players who haven't played recently
        for player in self.ratings:
            if self.last_update_period[player] < self.current_period:
                # Apply time decay to RD
                current_rd = self.rd[player] / 173.7178
                current_vol = self.vol[player]

                # Increase RD due to inactivity
                new_rd = math.sqrt(current_rd**2 + current_vol**2)
                self.rd[player] = min(new_rd * 173.7178, self.initial_rd)

    def get_confidence_interval(self, player, confidence_level=0.95):
        """
        Calculate confidence interval for a player's rating
        In Glicko-2, RD directly represents the standard deviation of the rating
        """
        if player not in self.ratings:
            return (None, None, None)

        rating = self.ratings[player]
        rd = self.rd[player]  # RD is already the standard deviation

        # Calculate confidence interval
        alpha = 1 - confidence_level
        z_score = norm.ppf(1 - alpha / 2)

        margin_error = z_score * rd
        lower_bound = rating - margin_error
        upper_bound = rating + margin_error

        return (lower_bound, upper_bound, rd)

    def get_rating_interval_95(self, player):
        """
        Get 95% confidence interval (convenience method)
        """
        return self.get_confidence_interval(player, 0.95)

    def get_rating_reliability(self, player):
        """
        Get rating reliability score (0-1)
        Lower RD = higher reliability
        """
        if player not in self.rd:
            return 0.0

        rd = self.rd[player]
        max_rd = self.initial_rd
        min_rd = 30  # Practical minimum RD

        # Normalize RD to 0-1 scale (inverted so higher = more reliable)
        reliability = 1 - (rd - min_rd) / (max_rd - min_rd)
        return max(0, min(1, reliability))

    def get_rankings(self):
        """Return sorted list of (player, rating, rd) tuples"""
        ranking = sorted(
            [(p, self.ratings[p], self.rd[p]) for p in self.ratings],
            key=lambda x: x[1],
            reverse=True,
        )
        ranking_list = [player for player, _, _ in ranking]
        return ranking, ranking_list

    def get_rankings_with_ci(self, confidence_level=0.95):
        """
        Return sorted list of (player, rating, lower_bound, upper_bound, rd, volatility, games) tuples
        """
        rankings = []
        for player in self.ratings:
            rating = self.ratings[player]
            rd = self.rd[player]
            vol = self.vol[player]
            games = self.games_played[player]

            lower, upper, _ = self.get_confidence_interval(player, confidence_level)
            rankings.append((player, rating, lower, upper, rd, vol, games))

        # Sort by rating (descending)
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings


class BradleyTerryModel:
    def __init__(self, max_iter=1000, tol=1e-8):
        self.max_iter = max_iter
        self.tol = tol
        self.ratings = {}
        self.players = set()
        self.confidence_intervals = {}
        self.standard_errors = {}

    def fit(self, matches):
        """
        Fit Bradley-Terry model using iterative algorithm
        matches: list of (character1, character2, result) tuples
        result: 1 if character1 wins, 0 if character2 wins
        """
        # Collect all players
        for character1, character2, _ in matches:
            self.players.add(character1)
            self.players.add(character2)

        # Initialize strengths
        n_players = len(self.players)
        self.ratings = {player: 1.0 for player in self.players}

        # Count wins and total games for each pair
        wins = defaultdict(lambda: defaultdict(int))
        games = defaultdict(lambda: defaultdict(int))

        for character1, character2, result in matches:
            if result == 1:
                wins[character1][character2] += 1
            else:
                wins[character2][character1] += 1
            games[character1][character2] += 1
            games[character2][character1] += 1

        # Store for confidence interval calculation
        self.wins = wins
        self.games = games

        # Iterative algorithm
        for iteration in range(self.max_iter):
            old_strengths = self.ratings.copy()

            for player in self.players:
                numerator = sum(
                    wins[player][opp] for opp in self.players if opp != player
                )
                denominator = sum(
                    games[player][opp]
                    * self.ratings[player]
                    / (self.ratings[player] + self.ratings[opp])
                    for opp in self.players
                    if opp != player
                )

                if denominator > 0:
                    self.ratings[player] = numerator / denominator
                else:
                    self.ratings[player] = 1.0

            # Normalize to prevent numerical issues
            total_strength = sum(self.ratings.values())
            if total_strength > 0:
                for player in self.players:
                    self.ratings[player] /= total_strength / n_players

            # Check convergence
            if (
                max(abs(self.ratings[p] - old_strengths[p]) for p in self.players)
                < self.tol
            ):
                break

        # Calculate confidence intervals after fitting
        self._calculate_confidence_intervals()

    def _calculate_fisher_information_matrix(self):
        """
        Calculate the Fisher Information Matrix for the Bradley-Terry model
        """
        n_players = len(self.players)
        player_list = list(self.players)
        player_idx = {player: i for i, player in enumerate(player_list)}

        # Initialize Fisher Information Matrix
        fisher_matrix = np.zeros((n_players, n_players))

        # Fill the Fisher Information Matrix
        for i, player_i in enumerate(player_list):
            for j, player_j in enumerate(player_list):
                if i == j:
                    # Diagonal elements
                    diagonal_sum = 0
                    for opp in self.players:
                        if opp != player_i:
                            n_ij = self.games[player_i][opp]
                            if n_ij > 0:
                                pi_i = self.ratings[player_i]
                                pi_j = self.ratings[opp]
                                p_ij = pi_i / (pi_i + pi_j)
                                diagonal_sum += (
                                    n_ij
                                    * p_ij
                                    * (1 - p_ij)
                                    * (pi_j / (pi_i + pi_j)) ** 2
                                )
                    fisher_matrix[i, i] = diagonal_sum
                else:
                    # Off-diagonal elements
                    n_ij = self.games[player_i][player_j]
                    if n_ij > 0:
                        pi_i = self.ratings[player_i]
                        pi_j = self.ratings[player_j]
                        p_ij = pi_i / (pi_i + pi_j)
                        fisher_matrix[i, j] = (
                            -n_ij
                            * p_ij
                            * (1 - p_ij)
                            * (pi_i * pi_j)
                            / ((pi_i + pi_j) ** 3)
                        )

        return fisher_matrix, player_list, player_idx

    def _calculate_confidence_intervals(self, confidence_level=0.95):
        """
        Calculate confidence intervals for player strengths using Fisher Information Matrix
        """
        try:
            fisher_matrix, player_list, player_idx = (
                self._calculate_fisher_information_matrix()
            )

            # Add small regularization to diagonal for numerical stability
            regularization = 1e-8
            fisher_matrix += regularization * np.eye(len(self.players))

            # Calculate covariance matrix (inverse of Fisher Information Matrix)
            try:
                cov_matrix = np.linalg.inv(fisher_matrix)
            except np.linalg.LinAlgError:
                # If inversion fails, use pseudo-inverse
                cov_matrix = np.linalg.pinv(fisher_matrix)

            # Extract standard errors (square root of diagonal elements)
            standard_errors = np.sqrt(np.diag(cov_matrix))

            # Calculate confidence intervals
            alpha = 1 - confidence_level
            z_score = norm.ppf(1 - alpha / 2)  # Two-tailed critical value

            self.standard_errors = {}
            self.confidence_intervals = {}

            for i, player in enumerate(player_list):
                se = standard_errors[i]
                strength = self.ratings[player]

                # Calculate confidence interval on log scale for better properties
                log_strength = np.log(strength)
                log_se = se / strength  # Delta method approximation

                log_lower = log_strength - z_score * log_se
                log_upper = log_strength + z_score * log_se

                # Transform back to original scale
                lower_bound = np.exp(log_lower)
                upper_bound = np.exp(log_upper)

                self.standard_errors[player] = se
                self.confidence_intervals[player] = (lower_bound, upper_bound)

        except Exception as e:
            print(f"Warning: Could not calculate confidence intervals: {e}")
            # Fallback: set empty confidence intervals
            for player in self.players:
                self.standard_errors[player] = np.nan
                self.confidence_intervals[player] = (np.nan, np.nan)

    def get_rankings(self):
        """Return sorted list of (player, strength) tuples"""
        ranking = sorted(self.ratings.items(), key=lambda x: x[1], reverse=True)
        ranking_list = [player for player, _ in ranking]
        return ranking, ranking_list

    def get_rankings_with_ci(self, confidence_level=0.95):
        """
        Return sorted list of (player, strength, lower_bound, upper_bound, standard_error) tuples
        """
        if confidence_level != 0.95:
            # Recalculate confidence intervals with different level
            self._calculate_confidence_intervals(confidence_level)

        rankings = []
        for player, strength in sorted(
            self.ratings.items(), key=lambda x: x[1], reverse=True
        ):
            lower, upper = self.confidence_intervals.get(player, (np.nan, np.nan))
            se = self.standard_errors.get(player, np.nan)
            rankings.append((player, strength, lower, upper, se))

        return rankings

    def print_rankings_with_ci(self, confidence_level=0.95):
        """
        Print rankings with confidence intervals in a formatted table
        """
        rankings = self.get_rankings_with_ci(confidence_level)

        print(
            f"\nBradley-Terry Model Rankings with {int(confidence_level * 100)}% Confidence Intervals"
        )
        print("=" * 80)
        print(
            f"{'Rank':<4} {'Player':<20} {'Strength':<10} {'95% CI':<20} {'Std Error':<10}"
        )
        print("-" * 80)

        for rank, (player, strength, lower, upper, se) in enumerate(rankings, 1):
            ci_str = (
                f"({lower:.4f}, {upper:.4f})" if not np.isnan(lower) else "(N/A, N/A)"
            )
            se_str = f"{se:.4f}" if not np.isnan(se) else "N/A"
            print(f"{rank:<4} {player:<20} {strength:<10.4f} {ci_str:<20} {se_str:<10}")


def count_wins_ranking(matches):
    """
    Count the number of wins per player and rank accordingly.

    Parameters:
    matches: list of (character1, character2, result)

    Returns:
    List of (player, win_count), sorted descending
    """
    win_counts = defaultdict(int)
    for character1, character2, result in matches:
        if result == 1:
            win_counts[character1] += 1
        else:
            win_counts[character2] += 1

    ranking = sorted(win_counts.items(), key=lambda x: x[1], reverse=True)
    ranking_list = [player for player, _ in ranking]
    return ranking, ranking_list


def rank_players_from_csv(
    csv_file,
    models=None,
    player_col_a="character1",
    player_col_b="character2",
    result_col="result",
    duplication_factor=1,
):
    """
    Rank players from CSV file using specified rating systems

    Parameters:
    csv_file: path to CSV file or CSV content as string
    models: list of (model_name, model_args) tuples or dict
           e.g. [("Elo", {"k_factor": 32}), ("Bradley-Terry", {})]
           If None, runs all available models with default parameters
    player_col_a: column name for first player
    player_col_b: column name for second player
    result_col: column name for result (1 if character1 wins, 0 if character2 wins)
    duplication_factor: number of times to duplicate matches

    Returns:
    Dictionary with rankings from specified systems and original matches
    """

    # Default to Elo and Glicko2 models with default parameters if none specified
    if models is None:
        models = [
            ("Elo", {}),
            ("Glicko2", {}),
        ]

    # Validate model names and extract parameters
    available_models = {"Elo", "Glicko2", "Bradley-Terry", "Elo2K"}
    model_configs = {}

    for model_name, model_args in models:
        if model_name not in available_models:
            raise ValueError(
                f"Unknown model: {model_name}. Available models: {available_models}"
            )
        model_configs[model_name] = model_args if model_args else {}

    # Read CSV
    if isinstance(csv_file, str) and "\n" in csv_file:
        # CSV content as string
        from io import StringIO

        df = pd.read_csv(StringIO(csv_file))
    else:
        # CSV file path
        df = pd.read_csv(csv_file)

    # Initialize only the requested rating systems with their parameters
    rating_systems = {}

    if "Elo" in model_configs:
        rating_systems["Elo"] = EloRatingSystem(**model_configs["Elo"])

    if "Glicko2" in model_configs:
        rating_systems["Glicko2"] = Glicko2RatingSystem(**model_configs["Glicko2"])

    # Process matches and create duplicated matches list
    duplicated_matches = []
    original_matches = []

    for i in range(duplication_factor):
        for _, row in df.iterrows():
            character1 = row[player_col_a]
            character2 = row[player_col_b]
            result = row[result_col]

            # Update sequential rating systems (Elo)
            if "Elo" in model_configs:
                rating_systems["Elo"].update_ratings(character1, character2, result)

            duplicated_matches.append((character1, character2, result))
            if i == 0:
                original_matches.append((character1, character2, result))

    # Process matches for Glicko-2 (batch updates per player)
    if "Glicko2" in model_configs:
        player_games = defaultdict(lambda: {"opponents": [], "results": []})
        for character1, character2, result in duplicated_matches:
            player_games[character1]["opponents"].append(character2)
            player_games[character1]["results"].append(result)
            player_games[character2]["opponents"].append(character1)
            player_games[character2]["results"].append(1 - result)

        for player, games in player_games.items():
            rating_systems["Glicko2"].update_player(
                player, games["opponents"], games["results"]
            )

    # Return results
    result_dict = rating_systems.copy()

    return result_dict, original_matches


def evaluate_rating_models(
    csv_file: str,
    models: Dict[str, Any],
    training_csv: str = None,
    run_with_pairwise: bool = False,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate the accuracy of different rating models on test match data.

    Args:
        csv_file: Path to CSV file with columns 'character1', 'character2', 'result'
        models: Dictionary with model names as keys and model objects as values
                Each model should have a .ratings attribute (dict mapping player -> rating)
        training_csv: Optional path to training CSV. If provided, adds pairwise comparison method

    Returns:
        Dictionary with model names as keys and metrics as values:
        - mse: Mean Squared Error
        - mae: Mean Absolute Error
        - accuracy: Win prediction accuracy (%)
        - log_loss: Mean logarithmic error
    """

    # Read the CSV data
    df = pd.read_csv(csv_file)

    # Validate required columns
    required_cols = ["character1", "character2", "result"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"CSV must contain columns: {required_cols}")

    # Load training data for pairwise comparison if provided
    pairwise_probs = {}
    if training_csv:
        train_df = pd.read_csv(training_csv)
        if not all(col in train_df.columns for col in required_cols):
            raise ValueError(f"Training CSV must contain columns: {required_cols}")

        # Build pairwise win probabilities from training data
        pairwise_counts = {}

        for _, row in train_df.iterrows():
            p1, p2, result = row["character1"], row["character2"], row["result"]

            # Count wins for both directions
            pair_key = (p1, p2)
            reverse_key = (p2, p1)

            if pair_key not in pairwise_counts:
                pairwise_counts[pair_key] = {"wins": 0, "total": 0}
            if reverse_key not in pairwise_counts:
                pairwise_counts[reverse_key] = {"wins": 0, "total": 0}

            # Update counts
            pairwise_counts[pair_key]["total"] += 1
            pairwise_counts[reverse_key]["total"] += 1

            if result == 1:  # p1 wins
                pairwise_counts[pair_key]["wins"] += 1
            else:  # p2 wins (result == 0)
                pairwise_counts[reverse_key]["wins"] += 1

        # Convert counts to probabilities
        for pair_key, counts in pairwise_counts.items():
            if counts["total"] > 0:
                pairwise_probs[pair_key] = counts["wins"] / counts["total"]

    results = {}

    for model_name, model in models.items():
        predictions = []
        actual_results = []

        for _, row in df.iterrows():
            player1 = row["character1"]
            player2 = row["character2"]
            actual_result = row["result"]  # 1 if player1 wins, 0 if player2 wins

            # Skip if either player not in model ratings
            if player1 not in model.ratings or player2 not in model.ratings:
                continue

            rating1 = model.ratings[player1]
            rating2 = model.ratings[player2]

            # Compute probability that player1 beats player2
            # Using logistic function: P(1 beats 2) = 1 / (1 + exp(-(r1 - r2)/400))
            # The 400 is a common scaling factor used in Elo systems
            prob_player1_wins = 1 / (1 + math.exp(-(rating1 - rating2) / 400))

            predictions.append(prob_player1_wins)
            actual_results.append(actual_result)

        if not predictions:
            results[model_name] = {
                "mse": float("nan"),
                "mae": float("nan"),
                "accuracy": float("nan"),
                "log_loss": float("nan"),
                "num_predictions": 0,
            }
            continue

        predictions = np.array(predictions)
        actual_results = np.array(actual_results)

        # Calculate metrics

        # 1. Mean Squared Error
        mse = np.mean((predictions - actual_results) ** 2)

        # 2. Mean Absolute Error
        mae = np.mean(np.abs(predictions - actual_results))

        # 3. Win Prediction Accuracy
        predicted_winners = (predictions > 0.5).astype(int)
        accuracy = np.mean(predicted_winners == actual_results) * 100

        # 4. Mean Logarithmic Error (Log Loss)
        # Clip predictions to avoid log(0)
        eps = 1e-15
        clipped_preds = np.clip(predictions, eps, 1 - eps)
        log_loss = -np.mean(
            actual_results * np.log(clipped_preds)
            + (1 - actual_results) * np.log(1 - clipped_preds)
        )

        # TODO: Add inconsistency score calculation
        matches = csv_to_matches(csv_file)
        _, ranking_list = model.get_rankings()
        inconsistency = calculate_inconsistency_score(
            matches=matches, ranking_order=ranking_list
        )

        results[model_name] = {
            "mse": round(mse, 6),
            "mae": round(mae, 6),
            "accuracy": round(accuracy, 2),
            "log_loss": round(log_loss, 6),
            "is": round(inconsistency, 6),
            "num_predictions": len(predictions),
        }

    if run_with_pairwise:
        # Evaluate pairwise comparisons
        predictions = []
        actual_results = []

        for _, row in df.iterrows():
            player1 = row["character1"]
            player2 = row["character2"]
            actual_result = row["result"]  # 1 if player1 wins, 0 if player2 wins

            # Skip if either player not in model ratings
            if player1 not in model.ratings or player2 not in model.ratings:
                continue

            # Compute probability that player1 beats player2
            prob_player1_wins = pairwise_probs[(player1, player2)]
            predictions.append(prob_player1_wins)
            actual_results.append(actual_result)

        predictions = np.array(predictions)
        actual_results = np.array(actual_results)

        # Calculate metrics

        # 1. Mean Squared Error
        mse = np.mean((predictions - actual_results) ** 2)

        # 2. Mean Absolute Error
        mae = np.mean(np.abs(predictions - actual_results))

        # 3. Win Prediction Accuracy
        predicted_winners = (predictions > 0.5).astype(int)
        accuracy = np.mean(predicted_winners == actual_results) * 100

        # 4. Mean Logarithmic Error (Log Loss)
        # Clip predictions to avoid log(0)
        eps = 1e-15
        clipped_preds = np.clip(predictions, eps, 1 - eps)
        log_loss = -np.mean(
            actual_results * np.log(clipped_preds)
            + (1 - actual_results) * np.log(1 - clipped_preds)
        )

        results["pairwise"] = {
            "mse": round(mse, 6),
            "mae": round(mae, 6),
            "accuracy": round(accuracy, 2),
            "log_loss": round(log_loss, 6),
            "num_predictions": len(predictions),
        }

    return results


def print_evaluation_results(results: Dict[str, Dict[str, float]]):
    """
    Pretty print the evaluation results in a tabular format.
    """
    print(
        f"{'Model':<15} {'IS':<10} {'MSE':<10} {'MAE':<10} {'Accuracy':<12} {'Log Loss':<12} {'N':<8}"
    )
    print("-" * 75)

    for model_name, metrics in results.items():
        print(
            f"{model_name:<15} {metrics['is']:<10.6f} {metrics['mse']:<10.6f} {metrics['mae']:<10.6f} "
            f"{metrics['accuracy']:<12.2f}% {metrics['log_loss']:<12.6f} {metrics['num_predictions']:<8}"
        )


def save_rankings_and_plots(
    models: Dict[str, Any],
    matches: list,
    output_dir: str,
    model_abbreviation: str,
    save_plots: bool = True
) -> Dict[str, str]:
    """
    Save rankings, metrics, and plots to the output directory.
    
    Args:
        models: Dictionary of trained rating models
        matches: List of match tuples
        output_dir: Directory to save results
        model_abbreviation: Model abbreviation for naming files
        save_plots: Whether to save plots

    Returns:
        Dictionary mapping result type to file path
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp and run ID for consistent naming
    timestamp = generate_timestamp()
    run_id = generate_run_id()
    
    results_files = {}
    
    # Get rankings from all models
    rankings_data = {}
    for model_name, model in models.items():
        if hasattr(model, 'get_rankings_with_ci'):
            rankings_with_ci = model.get_rankings_with_ci()
            rankings_data[model_name] = rankings_with_ci
        else:
            rankings, _ = model.get_rankings()
            rankings_data[model_name] = rankings
    
    # Save rankings to CSV with timestamped naming
    rankings_filename = f"{model_abbreviation}_rankings_{timestamp}_{run_id}.csv"
    rankings_csv_path = os.path.join(output_dir, rankings_filename)
    with open(rankings_csv_path, 'w') as f:
        f.write("Model,Rank,Player,Rating,Lower_Bound,Upper_Bound,Std_Error,Games\n")
        for model_name, rankings in rankings_data.items():
            for rank, ranking_data in enumerate(rankings, 1):
                if len(ranking_data) >= 2:
                    player = ranking_data[0]
                    rating = ranking_data[1]
                    # Handle different ranking formats
                    lower_bound = ranking_data[2] if len(ranking_data) > 2 else ""
                    upper_bound = ranking_data[3] if len(ranking_data) > 3 else ""
                    std_error = ranking_data[4] if len(ranking_data) > 4 else ""
                    games = ranking_data[5] if len(ranking_data) > 5 else ""
                    
                    f.write(f"{model_name},{rank},{player},{rating},{lower_bound},{upper_bound},{std_error},{games}\n")
    
    results_files['rankings'] = rankings_csv_path
    
    # Calculate and save metrics
    metrics_data = {}
    for model_name, model in models.items():
        # Calculate inconsistency score
        _, ranking_list = model.get_rankings()
        inconsistency = calculate_inconsistency_score(matches, ranking_list)
        
        metrics_data[model_name] = {
            'inconsistency_score': inconsistency,
            'num_players': len(ranking_list),
            'num_matches': len(matches)
        }
    
    # Save metrics to CSV with timestamped naming
    metrics_filename = f"{model_abbreviation}_metrics_{timestamp}_{run_id}.csv"
    metrics_csv_path = os.path.join(output_dir, metrics_filename)
    metrics_df = pd.DataFrame.from_dict(metrics_data, orient='index')
    metrics_df.to_csv(metrics_csv_path)
    results_files['metrics'] = metrics_csv_path
    
    # Save plots if requested
    if save_plots:
        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Plot rankings for each model
        for model_name, rankings in rankings_data.items():
            if len(rankings) > 0 and len(rankings[0]) >= 2:
                # Generate timestamped plot filename
                plot_filename = f"{model_abbreviation}_{model_name.lower()}_{timestamp}_{run_id}.png"
                plot_path = os.path.join(plots_dir, plot_filename)
                
                # Extract just (player, rating) tuples for plotting
                plot_rankings = [(ranking[0], ranking[1]) for ranking in rankings]
                
                # Create and save plot
                try:
                    plot_normalized_ranking(
                        plot_rankings, 
                        title=f"{model_name} Rankings - {model_abbreviation}",
                        top_n=10
                    )
                    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    results_files[f'{model_name.lower()}_plot'] = plot_path
                except Exception as e:
                    print(f"Warning: Could not save plot for {model_name}: {e}")
        
        # Add normalized win count plot
        try:
            win_rankings, _ = count_wins_ranking(matches)
            win_plot_filename = f"{model_abbreviation}_win_counts_{timestamp}_{run_id}.png"
            win_plot_path = os.path.join(plots_dir, win_plot_filename)
            
            plot_normalized_ranking(
                win_rankings,
                title=f"Normalized Win Counts - {model_abbreviation}",
                top_n=10
            )
            plt.savefig(win_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            results_files['win_counts_plot'] = win_plot_path
        except Exception as e:
            print(f"Warning: Could not save win counts plot: {e}")
    
    return results_files


def produce_rankings(
    train_csv: str,
    test_csv: str = None,
    model_abbreviation: str = "unknown",
    output_dir: str = "logs/results",
    save_plots: bool = True,
    seed: int = None
) -> Dict[str, str]:
    """
    Produce rankings and evaluation metrics from match data.
    
    Args:
        train_csv: Path to training CSV file
        test_csv: Optional path to test CSV file for evaluation
        model_abbreviation: Model abbreviation for naming output files
        output_dir: Directory to save results
        save_plots: Whether to save ranking plots
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary mapping result type to file path
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Generate timestamp and run ID for consistent naming
    timestamp = generate_timestamp()
    run_id = generate_run_id()
    
    # Create model-specific output directory with timestamp
    model_output_dir = os.path.join(output_dir, f"{model_abbreviation}_{timestamp}_{run_id}")
    
    print(f"Producing rankings for model: {model_abbreviation}")
    print(f"Run ID: {run_id}")
    print(f"Timestamp: {timestamp}")
    print(f"Training data: {train_csv}")
    if test_csv:
        print(f"Test data: {test_csv}")
    print(f"Output directory: {model_output_dir}")
    
    # Only Elo and Glicko2 models
    models = [
        ("Elo", {"k_factor": 32, "initial_rating": 1500}),
        ("Glicko2", {"initial_rating": 1500, "initial_rd": 350, "initial_vol": 0.06}),
    ]
    
    # Train models on training data
    print("Training rating models...")
    trained_models, matches = rank_players_from_csv(
        train_csv, models=models, duplication_factor=1
    )
    
    # Save rankings and plots
    print("Saving rankings and plots...")
    results_files = save_rankings_and_plots(
        trained_models, matches, model_output_dir, model_abbreviation, save_plots
    )
    
    # Evaluate on test data if provided
    if test_csv:
        print("Evaluating models on test data...")
        models_to_test = {
            "elo": trained_models["Elo"],
            "glicko2": trained_models["Glicko2"],
        }
        
        test_results = evaluate_rating_models(
            test_csv, models=models_to_test, training_csv=train_csv
        )
        
        # Save evaluation results with timestamped naming
        eval_filename = f"{model_abbreviation}_evaluation_{timestamp}_{run_id}.csv"
        eval_csv_path = os.path.join(model_output_dir, eval_filename)
        eval_df = pd.DataFrame.from_dict(test_results, orient='index')
        eval_df.to_csv(eval_csv_path)
        results_files['evaluation'] = eval_csv_path
        
        # Print evaluation results
        print("\nEvaluation Results:")
        print_evaluation_results(test_results)
    
    print(f"\n✅ Results saved to: {model_output_dir}")
    return results_files


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Produce rankings and evaluation metrics from match data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python produce_rankings.py --train deepseek-qwen_train.csv --test deepseek-qwen_test.csv --model deepseek-qwen
  python produce_rankings.py --train gpt-4o-mini_train.csv --model gpt-4o-mini --output-dir results/
        """
    )
    
    parser.add_argument(
        "--train", 
        required=True,
        help="Path to training CSV file with match data"
    )
    
    parser.add_argument(
        "--test", 
        help="Path to test CSV file with match data (optional)"
    )
    
    parser.add_argument(
        "--model", 
        required=True,
        help="Model abbreviation for naming output files (e.g., deepseek-qwen, gpt-4o-mini)"
    )
    
    parser.add_argument(
        "--output-dir", 
        default="logs/results",
        help="Directory to save results"
    )
    
    parser.add_argument(
        "--no-plots", 
        action="store_true",
        help="Skip generating plots"
    )
    
    parser.add_argument(
        "--seed", 
        type=int,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    try:
        results_files = produce_rankings(
            train_csv=args.train,
            test_csv=args.test,
            model_abbreviation=args.model,
            output_dir=args.output_dir,
            save_plots=not args.no_plots,
            seed=args.seed
        )
        
        print("\n✅ Successfully generated rankings and metrics")
        print("Generated files:")
        for result_type, file_path in results_files.items():
            print(f"  {result_type}: {file_path}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())