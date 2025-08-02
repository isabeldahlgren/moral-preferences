#!/usr/bin/env python3
"""
Full evaluation workflow script that combines:
1. Generate questions (if needed)
2. Run matches between character pairs
3. Produce rankings and evaluation metrics

This script provides a convenient way to run the complete evaluation pipeline.
"""

import argparse
import os
import sys

# Add the scripts directory to the path so we can import the other modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generate_questions import generate_questions
from run_matches import run_matches, find_questions_file
from produce_rankings import produce_rankings
from file_utils import (
    create_output_directories,
    generate_questions_filename
)


def run_full_evaluation(
    characters_csv: str,
    model_string: str,
    questions_json: str = None,
    num_questions: int = 20,
    n_matches: int = 10,
    mode: str = "training",
    output_dir: str = "logs",
    generate_new_questions: bool = False,
    use_cot: bool = False,
    seed: int = None,
    verbose: bool = False,
    split: float = None,
    config_name: str = None
) -> dict:
    """
    Run the complete evaluation pipeline.
    
    Args:
        characters_csv: Path to CSV file with character data
        model_string: Model string (e.g., deepseek-qwen, gpt-4o-mini)
        questions_json: Path to existing questions JSON file (optional)
        num_questions: Number of questions to generate (if generating new ones)
        n_matches: Number of matches per character pair
        mode: Either "training" or "testing" (ignored if split is provided)
        output_dir: Directory to save all results
        generate_new_questions: Whether to generate new questions
        use_cot: Whether to use chain of thought reasoning
        seed: Random seed for reproducibility
        verbose: Whether to print verbose output
        split: If provided, split matches into training/testing sets (e.g., 0.8 for 80% training)
        
    Returns:
        Dictionary with paths to generated files
    """
    results = {}
    
    # Create organized output directories
    dirs = create_output_directories(output_dir)
    timestamp = dirs["timestamp"]
    run_id = dirs["run_id"]
    
    print(f"Starting evaluation run: {run_id}")
    print(f"Timestamp: {timestamp}")
    print(f"Output directory: {output_dir}")
    
    # Step 1: Generate questions (if needed)
    if generate_new_questions or questions_json is None:
        print("=" * 60)
        print("STEP 1: Generating evaluation questions")
        print("=" * 60)
        
        # Generate consistent filename for questions
        if config_name:
            # Use config-specific directory
            config_dir = os.path.join("logs", "question-configs", config_name)
            os.makedirs(config_dir, exist_ok=True)
            questions_filename = f"{config_name}_{timestamp}_{run_id}.json"
            questions_output = os.path.join(config_dir, questions_filename)
        else:
            # Use general questions directory
            questions_filename = generate_questions_filename(
                num_questions=num_questions,
                run_id=run_id,
                timestamp=timestamp
            )
            questions_output = os.path.join(dirs["questions"], questions_filename)
        
        questions = generate_questions(
            num_questions=num_questions,
            model="gpt-4o-mini",  # Use GPT-4o-mini for question generation
            output_file=questions_output,
            seed=seed,
            verbose=verbose,
            config_name=config_name
        )
        questions_json = questions_output
        results['questions'] = questions_output
        print(f"✅ Generated {len(questions)} questions: {questions_output}")
    else:
        # Handle both direct paths and config names
        if questions_json and not os.path.exists(questions_json):
            # Try to find it as a config name
            found_questions = find_questions_file(questions_json)
            if found_questions != questions_json:
                print(f"✅ Found questions using config '{questions_json}': {found_questions}")
                questions_json = found_questions
            else:
                print(f"⚠️  Warning: Questions file not found: {questions_json}")
        else:
            print(f"✅ Using existing questions: {questions_json}")
        results['questions'] = questions_json
    
    # Step 2: Run matches
    print("\n" + "=" * 60)
    print("STEP 2: Running matches between character pairs")
    print("=" * 60)
    
    csv_output_dir = dirs["csv_files"]
    
    if split is not None:
        # Split mode: generate both training and testing files
        train_path, test_path = run_matches(
            model_string=model_string,
            characters_csv=characters_csv,
            questions_json=questions_json,
            mode=mode,
            n_matches=n_matches,
            output_dir=csv_output_dir,
            use_cot=use_cot,
            seed=seed,
            split=split
        )
        results['train_matches'] = train_path
        results['test_matches'] = test_path
        print(f"✅ Generated training matches: {train_path}")
        print(f"✅ Generated testing matches: {test_path}")
    else:
        # Single mode: generate one file based on mode
        csv_path = run_matches(
            model_string=model_string,
            characters_csv=characters_csv,
            questions_json=questions_json,
            mode=mode,
            n_matches=n_matches,
            output_dir=csv_output_dir,
            use_cot=use_cot,
            seed=seed
        )
        results['matches'] = csv_path
        print(f"✅ Generated matches: {csv_path}")
    
    # Step 3: Produce rankings
    print("\n" + "=" * 60)
    print("STEP 3: Producing rankings and evaluation metrics")
    print("=" * 60)
    
    ranking_output_dir = dirs["results"]
    
    if split is not None:
        # Use both training and testing files
        ranking_files = produce_rankings(
            train_csv=train_path,
            test_csv=test_path,
            model_string=model_string,
            output_dir=ranking_output_dir,
            save_plots=True,
            seed=seed
        )
    else:
        # Use single file
        ranking_files = produce_rankings(
            train_csv=csv_path,
            model_string=model_string,
            output_dir=ranking_output_dir,
            save_plots=True,
            seed=seed
        )
    
    results['rankings'] = ranking_files
    
    print(f"✅ Generated rankings and metrics in: {ranking_output_dir}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE!")
    print("=" * 60)
    print(f"Model: {model_string}")
    print(f"Run ID: {run_id}")
    print(f"Timestamp: {timestamp}")
    if split is not None:
        print(f"Split: {split*100:.0f}% training, {(1-split)*100:.0f}% testing")
    else:
        print(f"Mode: {mode}")
    print(f"Characters: {characters_csv}")
    print(f"Matches per pair: {n_matches}")
    print(f"Output directory: {output_dir}")
    print("\nGenerated files:")
    for file_type, file_path in results.items():
        if isinstance(file_path, dict):
            print(f"  {file_type}:")
            for sub_type, sub_path in file_path.items():
                print(f"    {sub_type}: {sub_path}")
        else:
            print(f"  {file_type}: {file_path}")
    
    return results


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run complete evaluation pipeline: generate questions, run matches, produce rankings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_full_evaluation.py --characters characters.csv --model deepseek-qwen --mode training
  python run_full_evaluation.py --characters ethnic.csv --model gpt-4o-mini --mode testing --n-matches 5
  python run_full_evaluation.py --characters diverse-characters.csv --model mistral-instruct --generate-questions --num-questions 30
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--characters", 
        required=True,
        help="Path to CSV file with character data (must have 'character' and 'article' columns)"
    )
    
    parser.add_argument(
        "--model", 
        required=True,
        help="Model string (e.g., deepseek-qwen, gpt-4o-mini, mistral-instruct)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--questions", 
        help="Path to existing questions JSON file (if not provided, will generate new ones)"
    )
    
    parser.add_argument(
        "--generate-questions", 
        action="store_true",
        help="Force generation of new questions (overrides --questions)"
    )
    
    parser.add_argument(
        "--num-questions", 
        type=int, 
        default=20,
        help="Number of questions to generate (default: 20)"
    )
    
    parser.add_argument(
        "--n-matches", 
        type=int, 
        default=10,
        help="Number of matches per character pair (default: 10)"
    )
    
    parser.add_argument(
        "--mode", 
        choices=["training", "testing"], 
        default="training",
        help="Mode: training or testing (default: training)"
    )
    
    parser.add_argument(
        "--output-dir", 
        default="logs",
        help="Directory to save all results"
    )
    
    parser.add_argument(
        "--use-cot", 
        action="store_true",
        help="Use chain of thought reasoning for matches"
    )
    
    parser.add_argument(
        "--seed", 
        type=int,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Print verbose output"
    )
    
    parser.add_argument(
        "--split",
        type=float,
        help="If provided, split matches into training/testing sets (e.g., 0.8 for 80 percent training)"
    )
    
    parser.add_argument(
        "--config", 
        help="Configuration name for question generation (e.g., default, custom_config, thoughtful)"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.characters):
        print(f"❌ Error: Characters file not found: {args.characters}")
        return 1
    
    # Check if we need to generate questions
    should_generate_questions = args.generate_questions or args.questions is None
    
    if should_generate_questions and args.questions:
        print("⚠️  Warning: --generate-questions overrides --questions, will generate new questions")
    
    try:
        results = run_full_evaluation(
            characters_csv=args.characters,
            model_string=args.model,
            questions_json=args.questions,
            num_questions=args.num_questions,
            n_matches=args.n_matches,
            mode=args.mode,
            output_dir=args.output_dir,
            generate_new_questions=should_generate_questions,
            use_cot=args.use_cot,
            seed=args.seed,
            verbose=args.verbose,
            split=args.split,
            config_name=args.config
        )
        
        print("\n🎉 Evaluation completed successfully!")
        print(f"Results saved in: {args.output_dir}")
        
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 