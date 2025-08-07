#!/usr/bin/env python3
"""
Regenerate results from existing match CSV files.

This script takes existing match CSV files and regenerates rankings, metrics, and plots
with the latest code (including WinCount rankings, updated plots, etc.) without
rerunning the expensive match generation step.
"""

import os
import glob
import argparse

# Import the ranking functions
from produce_rankings import produce_rankings


def find_match_files(csv_dir="logs/csv-files"):
    """
    Find all match CSV files and group them by unique run (including timestamp and run ID).
    Returns:
        dict: {run_key: {'train': path, 'test': path}}
    """
    match_files = {}
    csv_files = glob.glob(os.path.join(csv_dir, "matches_*.csv"))
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        if filename.startswith("matches_"):
            # Remove "matches_" prefix
            base = filename[len("matches_"):]
            # Use everything up to _train.csv or _test.csv as the unique run key
            if base.endswith("_train.csv"):
                run_key = base[:-len("_train.csv")]
                file_type = "train"
            elif base.endswith("_test.csv"):
                run_key = base[:-len("_test.csv")]
                file_type = "test"
            else:
                continue
            if run_key not in match_files:
                match_files[run_key] = {}
            match_files[run_key][file_type] = csv_file
    return match_files


def regenerate_results_for_model(model_name, train_csv, test_csv=None, output_dir="logs/results"):
    """
    Regenerate results for a specific model using existing match files.
    
    Args:
        model_name: Name of the model (e.g., "together_deepseek-ai_DeepSeek-R1-Distill-Qwen-1.5B")
        train_csv: Path to training CSV file
        test_csv: Path to test CSV file (optional)
        output_dir: Directory to save results
    """
    print(f"\n{'='*60}")
    print(f"Regenerating results for: {model_name}")
    print(f"{'='*60}")
    
    try:
        # Generate new results using the updated produce_rankings function
        results = produce_rankings(
            train_csv=train_csv,
            test_csv=test_csv,
            model_string=model_name,
            output_dir=output_dir,
            save_plots=True,
            seed=42  # Use consistent seed for reproducibility
        )
        
        print(f"✅ Successfully regenerated results for {model_name}")
        print(f"Results saved to: {output_dir}")
        
        # Print summary of generated files
        if isinstance(results, dict):
            print("Generated files:")
            for file_type, file_path in results.items():
                if os.path.exists(file_path):
                    print(f"  {file_type}: {file_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error regenerating results for {model_name}: {e}")
        return False


def main():
    """Main function to regenerate all results."""
    parser = argparse.ArgumentParser(
        description="Regenerate results from existing match CSV files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python regenerate_results.py --all
  python regenerate_results.py --model together_deepseek-ai_DeepSeek-R1-Distill-Qwen-1.5B
  python regenerate_results.py --csv-dir logs/csv-files --output-dir logs/results_new
        """
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Regenerate results for all models found in csv-files directory"
    )
    
    parser.add_argument(
        "--model",
        help="Regenerate results for a specific model (e.g., together_deepseek-ai_DeepSeek-R1-Distill-Qwen-1.5B)"
    )
    
    parser.add_argument(
        "--csv-dir",
        default="logs/csv-files",
        help="Directory containing match CSV files (default: logs/csv-files)"
    )
    
    parser.add_argument(
        "--output-dir",
        default="logs/results_regenerated",
        help="Directory to save regenerated results (default: logs/results_regenerated)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be regenerated without actually doing it"
    )
    
    args = parser.parse_args()
    
    # Find all match files
    print("Finding match CSV files...")
    match_files = find_match_files(args.csv_dir)
    
    if not match_files:
        print(f"❌ No match files found in {args.csv_dir}")
        return 1
    
    print(f"Found {len(match_files)} unique runs with match data:")
    for run_key, files in match_files.items():
        print(f"  Run {run_key}:")
        if 'train' in files:
            print(f"    Train: {os.path.basename(files['train'])}")
        if 'test' in files:
            print(f"    Test: {os.path.basename(files['test'])}")
    
    # Determine which models to regenerate
    models_to_regenerate = []
    
    if args.all:
        models_to_regenerate = list(match_files.keys())
    elif args.model:
        if args.model in match_files:
            models_to_regenerate = [args.model]
        else:
            print(f"❌ Model '{args.model}' not found in match files")
            print(f"Available models: {list(match_files.keys())}")
            return 1
    else:
        # If no specific model and not --all, default to --all for dry run
        if args.dry_run:
            models_to_regenerate = list(match_files.keys())
        else:
            print("❌ Please specify --all or --model")
            return 1
    
    # Handle dry run
    if args.dry_run:
        print(f"\nDRY RUN: Would regenerate results for {len(models_to_regenerate)} models:")
        for run_key in models_to_regenerate:
            files = match_files[run_key]
            print(f"  Run {run_key}:")
            print(f"    Train: {files.get('train', 'NOT FOUND')}")
            print(f"    Test: {files.get('test', 'NOT FOUND')}")
        return 0
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Regenerate results for each model
    successful = 0
    failed = 0
    
    for run_key in models_to_regenerate:
        files = match_files[run_key]
        
        if 'train' not in files:
            print(f"❌ No training file found for Run {run_key}, skipping...")
            failed += 1
            continue
        
        train_csv = files['train']
        test_csv = files.get('test')  # Optional
        
        success = regenerate_results_for_model(
            model_name=run_key, # Pass the run_key as the model_name
            train_csv=train_csv,
            test_csv=test_csv,
            output_dir=args.output_dir
        )
        
        if success:
            successful += 1
        else:
            failed += 1
    
    # Print summary
    print(f"\n{'='*60}")
    print("REGENERATION SUMMARY")
    print(f"{'='*60}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total: {successful + failed}")
    
    if successful > 0:
        print("\n✅ Results regenerated successfully!")
        print(f"New results saved to: {args.output_dir}")
        print("\nThe new results include:")
        print("  ✅ WinCount rankings in rankings CSV")
        print("  ✅ WinCount inconsistency scores in metrics")
        print("  ✅ WinCount evaluation in evaluation CSV")
        print("  ✅ Updated plots showing all characters")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit(main()) 