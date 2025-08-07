#!/usr/bin/env python3
"""
Advanced Analysis Examples for Moral Preferences Evaluation

This file demonstrates advanced usage patterns and analysis techniques.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add the parent directory to the path so we can import the modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.produce_rankings import EloRatingSystem, Glicko2RatingSystem
from analysis.plotting_utils import load_results_data
from analysis.variance_analysis import run_variance_analysis

def custom_ranking_analysis():
    """Demonstrate custom ranking analysis."""
    print("🔬 CUSTOM RANKING ANALYSIS")
    print("=" * 60)
    
    # Example: Analyze specific ranking methods
    print("1. Creating custom Elo rating system...")
    elo_system = EloRatingSystem(k_factor=32, initial_rating=1500)
    
    # Simulate some matches
    matches = [
        ('doctor', 'teacher', 1),  # doctor wins
        ('doctor', 'student', 1),  # doctor wins
        ('teacher', 'student', 0), # student wins
        ('engineer', 'artist', 1), # engineer wins
    ]
    
    print("2. Processing matches...")
    for char1, char2, result in matches:
        elo_system.update_ratings(char1, char2, result)
        print(f"   {char1} vs {char2}: {'Win' if result == 1 else 'Loss'}")
    
    # Get rankings
    rankings = elo_system.get_rankings()
    print("\n3. Final Elo Rankings:")
    for i, (player, rating) in enumerate(rankings, 1):
        print(f"   {i}. {player}: {rating:.1f}")
    
    # Get confidence intervals
    print("\n4. Rankings with Confidence Intervals:")
    rankings_with_ci = elo_system.get_rankings_with_ci(confidence_level=0.95)
    for player, rating, ci_lower, ci_upper in rankings_with_ci:
        print(f"   {player}: {rating:.1f} [{ci_lower:.1f}, {ci_upper:.1f}]")
    
    print("\n✅ Custom ranking analysis complete!")

def model_comparison_analysis():
    """Demonstrate model comparison analysis."""
    print("\n📊 MODEL COMPARISON ANALYSIS")
    print("=" * 60)
    
    print("1. Loading results data...")
    data = load_results_data()
    
    if data.empty:
        print("No data found. Please run some evaluations first.")
        return
    
    print(f"✅ Loaded data for {len(data['model_name'].unique())} models")
    
    # Analyze performance by model
    print("\n2. Performance by Model:")
    model_performance = data.groupby('model_name').agg({
        'accuracy': ['mean', 'std'],
        'mse': ['mean', 'std'],
        'inconsistency_score': ['mean', 'std']
    }).round(4)
    
    print(model_performance)
    
    # Analyze performance by method
    print("\n3. Performance by Method:")
    method_performance = data.groupby('method').agg({
        'accuracy': ['mean', 'std'],
        'mse': ['mean', 'std'],
        'inconsistency_score': ['mean', 'std']
    }).round(4)
    
    print(method_performance)
    
    # Create custom visualization
    print("\n4. Creating custom visualization...")
    plt.figure(figsize=(12, 8))
    
    # Box plot of accuracy by method
    plt.subplot(2, 2, 1)
    sns.boxplot(data=data, x='method', y='accuracy')
    plt.title('Accuracy by Ranking Method')
    plt.xticks(rotation=45)
    
    # Box plot of MSE by method
    plt.subplot(2, 2, 2)
    sns.boxplot(data=data, x='method', y='mse')
    plt.title('MSE by Ranking Method')
    plt.xticks(rotation=45)
    
    # Scatter plot of accuracy vs inconsistency
    plt.subplot(2, 2, 3)
    for method in data['method'].unique():
        method_data = data[data['method'] == method]
        plt.scatter(method_data['inconsistency_score'], method_data['accuracy'], 
                   label=method, alpha=0.7)
    plt.xlabel('Inconsistency Score')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Inconsistency')
    plt.legend()
    
    # Heatmap of model-method performance
    plt.subplot(2, 2, 4)
    pivot_data = data.pivot_table(values='accuracy', index='model_name', columns='method', aggfunc='mean')
    sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='YlOrRd')
    plt.title('Accuracy Heatmap')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('logs/advanced_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Custom visualization saved to logs/advanced_analysis.png")
    print("\n✅ Model comparison analysis complete!")

def variance_deep_dive():
    """Demonstrate deep dive into variance analysis."""
    print("\n📈 VARIANCE DEEP DIVE")
    print("=" * 60)
    
    # Check for variance data
    variance_dirs = [
        'logs/variances_over_runs',
        'logs/variances_over_prompts'
    ]
    
    for var_dir in variance_dirs:
        if os.path.exists(var_dir):
            table_file = os.path.join(var_dir, 'variance_metrics_table.csv')
            if os.path.exists(table_file):
                print(f"1. Analyzing variance data from {var_dir}...")
                
                df = pd.read_csv(table_file)
                print(f"✅ Loaded variance data for {len(df)} model-method combinations")
                
                # Analyze variance patterns
                print("\n2. Variance Analysis Summary:")
                for metric in ['Total Variation Distance', 'Jensen-Shannon Divergence', 'Rank Edit Distance']:
                    if metric in df.columns:
                        print(f"\n{metric}:")
                        print(f"  Min: {df[metric].min():.4f}")
                        print(f"  Max: {df[metric].max():.4f}")
                        print(f"  Mean: {df[metric].mean():.4f}")
                        print(f"  Std: {df[metric].std():.4f}")
                        
                        # Find most and least variable models
                        max_var = df.loc[df[metric].idxmax()]
                        min_var = df.loc[df[metric].idxmin()]
                        print(f"  Most variable: {max_var['Model']} ({max_var['Method']}) = {max_var[metric]:.4f}")
                        print(f"  Least variable: {min_var['Model']} ({min_var['Method']}) = {min_var[metric]:.4f}")
                
                # Method comparison
                print("\n3. Method Stability Comparison:")
                method_stats = df.groupby('Method').agg({
                    'Total Variation Distance': ['mean', 'std'],
                    'Jensen-Shannon Divergence': ['mean', 'std'],
                    'Rank Edit Distance': ['mean', 'std']
                }).round(4)
                print(method_stats)
                
                # Model comparison
                print("\n4. Model Stability Comparison:")
                model_stats = df.groupby('Model').agg({
                    'Total Variation Distance': ['mean', 'std'],
                    'Jensen-Shannon Divergence': ['mean', 'std'],
                    'Rank Edit Distance': ['mean', 'std']
                }).round(4)
                print(model_stats)
                
                break
    else:
        print("⚠ No variance data found.")
        print("Run multiple evaluations with the same model to generate variance data.")
    
    print("\n✅ Variance deep dive complete!")

def statistical_analysis():
    """Demonstrate statistical analysis techniques."""
    print("\n📊 STATISTICAL ANALYSIS")
    print("=" * 60)
    
    print("1. Loading data for statistical analysis...")
    data = load_results_data()
    
    if data.empty:
        print("No data found for statistical analysis.")
        return
    
    from scipy import stats
    
    print(f"✅ Loaded {len(data)} data points")
    
    # Statistical tests
    print("\n2. Statistical Tests:")
    
    # Compare Elo vs Glicko2 accuracy
    elo_data = data[data['method'] == 'Elo']['accuracy']
    glicko_data = data[data['method'] == 'Glicko2']['accuracy']
    
    if len(elo_data) > 0 and len(glicko_data) > 0:
        # T-test
        t_stat, p_value = stats.ttest_ind(elo_data, glicko_data)
        print(f"\nElo vs Glicko2 Accuracy (T-test):")
        print(f"  T-statistic: {t_stat:.4f}")
        print(f"  P-value: {p_value:.4f}")
        print(f"  Significant difference: {'Yes' if p_value < 0.05 else 'No'}")
        
        # Effect size (Cohen's d)
        pooled_std = ((len(elo_data) - 1) * elo_data.var() + (len(glicko_data) - 1) * glicko_data.var()) / (len(elo_data) + len(glicko_data) - 2)
        cohens_d = (elo_data.mean() - glicko_data.mean()) / pooled_std**0.5
        print(f"  Cohen's d: {cohens_d:.4f}")
    
    # Correlation analysis
    print("\n3. Correlation Analysis:")
    numeric_cols = ['accuracy', 'mse', 'inconsistency_score']
    correlation_matrix = data[numeric_cols].corr()
    print("Correlation Matrix:")
    print(correlation_matrix.round(3))
    
    # Regression analysis
    print("\n4. Regression Analysis:")
    if 'inconsistency_score' in data.columns and 'accuracy' in data.columns:
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score
        
        X = data[['inconsistency_score']].values
        y = data['accuracy'].values
        
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        
        print(f"Accuracy vs Inconsistency Score:")
        print(f"  R²: {r2:.4f}")
        print(f"  Slope: {model.coef_[0]:.4f}")
        print(f"  Intercept: {model.intercept_:.4f}")
    
    print("\n✅ Statistical analysis complete!")

def main():
    """Main function to run all advanced analysis examples."""
    print("🔬 MORAL PREFERENCES EVALUATION - ADVANCED ANALYSIS")
    print("=" * 80)
    
    # Run all advanced examples
    custom_ranking_analysis()
    model_comparison_analysis()
    variance_deep_dive()
    statistical_analysis()
    
    print("\n" + "=" * 80)
    print("✅ ALL ADVANCED ANALYSIS COMPLETED!")
    print("=" * 80)
    print("\n💡 Advanced Tips:")
    print("- Use custom ranking parameters for specific use cases")
    print("- Analyze variance to understand model stability")
    print("- Perform statistical tests to validate results")
    print("- Create custom visualizations for publication")
    print("- Consider effect sizes, not just p-values")

if __name__ == "__main__":
    main() 