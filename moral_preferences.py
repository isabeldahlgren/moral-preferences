#!/usr/bin/env python3
"""
Moral Preferences Evaluation Package

A simplified interface for evaluating moral preferences in AI models through character-based comparisons.

Usage:
    python moral_preferences.py generate-questions [options]
    python moral_preferences.py evaluate [options]
"""

import argparse
import os
import sys

# Add the current directory to the path so we can import the modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.generate_questions import generate_questions
from core.run_matches import run_matches
from core.produce_rankings import produce_rankings
from core.file_utils import (
    create_output_directories,
    generate_questions_filename
)


def generate_questions_command(args):
    """Generate evaluation questions for moral preference testing."""
    print("=" * 60)
    print("GENERATING EVALUATION QUESTIONS")
    print("=" * 60)
    
    # Create output directories
    dirs = create_output_directories(args.output_dir)
    timestamp = dirs["timestamp"]
    run_id = dirs["run_id"]
    
    # Generate consistent filename for questions
    questions_filename = generate_questions_filename(
        num_questions=args.num_questions,
        run_id=run_id,
        timestamp=timestamp
    )
    questions_output = os.path.join(dirs["questions"], questions_filename)
    
    questions = generate_questions(
        num_questions=args.num_questions,
        model=args.model,
        output_file=questions_output,
        seed=args.seed,
        verbose=args.verbose
    )
    
    print(f"✅ Generated {len(questions)} questions: {questions_output}")
    return questions_output


def evaluate_command(args):
    """Run matches and produce rankings in one command."""
    print("=" * 60)
    print("EVALUATING MORAL PREFERENCES")
    print("=" * 60)
    
    # Create output directories
    dirs = create_output_directories(args.output_dir)
    timestamp = dirs["timestamp"]
    run_id = dirs["run_id"]
    
    # Determine questions file to use
    if args.questions:
        questions_json = args.questions
        print(f"✅ Using specified questions file: {questions_json}")
    else:
        # Use default questions file in data directory
        default_questions = os.path.join(os.path.dirname(__file__), "data", "questions.json")
        if os.path.exists(default_questions):
            questions_json = default_questions
            print(f"✅ Using default questions file: {questions_json}")
        else:
            print("❌ Error: No questions file specified and default data/questions.json not found")
            print("Please either:")
            print("  1. Specify a questions file with --questions")
            print("  2. Generate questions first with: python moral_preferences.py generate-questions")
            return 1
    
    # Step 1: Run matches
    print("\n" + "=" * 60)
    print("STEP 1: Running matches between character pairs")
    print("=" * 60)
    
    csv_output_dir = dirs["csv_files"]
    
    if args.test:
        # Test mode: split matches into training/testing sets
        print(f"🔬 Test mode: Splitting matches {args.split*100:.0f}% training, {(1-args.split)*100:.0f}% testing")
        train_path, test_path = run_matches(
            model_string=args.model,
            characters_csv=args.characters,
            questions_json=questions_json,
            mode="training",  # Ignored in split mode
            n_matches=args.n_matches,
            output_dir=csv_output_dir,
            use_cot=args.use_cot,
            seed=args.seed,
            split=args.split
        )
        print(f"✅ Generated training matches: {train_path}")
        print(f"✅ Generated testing matches: {test_path}")
    else:
        # Training mode: generate single file
        print("📊 Training mode: Generating rankings and plots only")
        csv_path = run_matches(
            model_string=args.model,
            characters_csv=args.characters,
            questions_json=questions_json,
            mode="training",
            n_matches=args.n_matches,
            output_dir=csv_output_dir,
            use_cot=args.use_cot,
            seed=args.seed
        )
        print(f"✅ Generated matches: {csv_path}")
    
    # Step 2: Produce rankings
    print("\n" + "=" * 60)
    print("STEP 2: Producing rankings and evaluation metrics")
    print("=" * 60)
    
    ranking_output_dir = dirs["results"]
    
    if args.test:
        # Test mode: use both training and testing files for evaluation
        print("🔬 Evaluating predictive accuracy with ELO and Glicko2 rankings...")
        ranking_files = produce_rankings(
            train_csv=train_path,
            test_csv=test_path,
            model_string=args.model,
            output_dir=ranking_output_dir,
            save_plots=True,
            seed=args.seed
        )
    else:
        # Training mode: use single file for rankings only
        ranking_files = produce_rankings(
            train_csv=csv_path,
            model_string=args.model,
            output_dir=ranking_output_dir,
            save_plots=True,
            seed=args.seed
        )
    
    print(f"✅ Generated rankings and metrics in: {ranking_output_dir}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE!")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Run ID: {run_id}")
    print(f"Timestamp: {timestamp}")
    print(f"Characters: {args.characters}")
    print(f"Questions: {questions_json}")
    print(f"Matches per pair: {args.n_matches}")
    print(f"Mode: {'Test' if args.test else 'Training'}")
    if args.test:
        print(f"Split: {args.split*100:.0f}% training, {(1-args.split)*100:.0f}% testing")
    print(f"Output directory: {args.output_dir}")
    
    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Moral Preferences Evaluation Package",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  generate-questions  Generate evaluation questions for moral preference testing
  evaluate           Run matches and produce rankings in one command

Examples:
  # Generate 20 questions
  python moral_preferences.py generate-questions --num-questions 20

  # Run evaluation with default questions (training mode)
  python moral_preferences.py evaluate --characters characters.csv --model openai/gpt-4o-mini

  # Run evaluation with custom questions (training mode)
  python moral_preferences.py evaluate --characters characters.csv --model openai/gpt-4o-mini --questions my_questions.json

  # Run evaluation with test mode (training/test split + predictive accuracy)
  python moral_preferences.py evaluate --characters characters.csv --model openai/gpt-4o-mini --test --split 0.8
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Generate questions command
    gen_parser = subparsers.add_parser(
        'generate-questions',
        help='Generate evaluation questions for moral preference testing'
    )
    gen_parser.add_argument(
        '--num-questions',
        type=int,
        default=20,
        help='Number of questions to generate (default: 20)'
    )
    gen_parser.add_argument(
        '--model',
        default='gpt-4o-mini',
        help='Model to use for generation (default: gpt-4o-mini)'
    )
    gen_parser.add_argument(
        '--output-dir',
        default='logs',
        help='Directory to save results (default: logs)'
    )
    gen_parser.add_argument(
        '--seed',
        type=int,
        help='Random seed for reproducibility'
    )
    gen_parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print verbose output'
    )
    
    # Evaluate command
    eval_parser = subparsers.add_parser(
        'evaluate',
        help='Run matches and produce rankings in one command'
    )
    eval_parser.add_argument(
        '--characters',
        required=True,
        help='Path to CSV file with character data (must have "character" and "article" columns)'
    )
    eval_parser.add_argument(
        '--model',
        required=True,
        help='Full Inspect model string (e.g., openai/gpt-4o-mini, together/meta-llama/Llama-3-70B-Instruct)'
    )
    eval_parser.add_argument(
        '--questions',
        help='Path to questions JSON file (default: uses questions.json in scripts directory)'
    )
    eval_parser.add_argument(
        '--n-matches',
        type=int,
        default=10,
        help='Number of matches per character pair (default: 10)'
    )
    eval_parser.add_argument(
        '--output-dir',
        default='logs',
        help='Directory to save results (default: logs)'
    )
    eval_parser.add_argument(
        '--use-cot',
        action='store_true',
        help='Use chain of thought reasoning for matches'
    )
    eval_parser.add_argument(
        '--seed',
        type=int,
        help='Random seed for reproducibility'
    )
    eval_parser.add_argument(
        '--test',
        action='store_true',
        help='Enable test mode: split matches into training/testing sets and evaluate predictive accuracy'
    )
    eval_parser.add_argument(
        '--split',
        type=float,
        default=0.8,
        help='Training/test split ratio (default: 0.8, only used with --test)'
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Validate input file for evaluate command
    if args.command == 'evaluate':
        if not os.path.exists(args.characters):
            print(f"❌ Error: Characters file not found: {args.characters}")
            return 1
        
        if args.questions and not os.path.exists(args.questions):
            print(f"❌ Error: Questions file not found: {args.questions}")
            return 1
    
    try:
        if args.command == 'generate-questions':
            generate_questions_command(args)
        elif args.command == 'evaluate':
            return evaluate_command(args)
        
        print("\n🎉 Command completed successfully!")
        print(f"Results saved in: {args.output_dir}")
        
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 