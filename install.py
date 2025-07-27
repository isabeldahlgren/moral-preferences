#!/usr/bin/env python3
"""
Installation script for the Moral Preferences Evaluation Package.

This script sets up the package for easy use by:
1. Making the main script executable
2. Creating a symlink for easy access
3. Setting up the environment
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Install the moral preferences package."""
    print("=" * 60)
    print("INSTALLING MORAL PREFERENCES EVALUATION PACKAGE")
    print("=" * 60)
    
    # Get the directory where this script is located
    script_dir = Path(__file__).parent.absolute()
    
    # Make the main script executable
    main_script = script_dir / "moral_preferences.py"
    if main_script.exists():
        os.chmod(main_script, 0o755)
        print(f"✅ Made {main_script} executable")
    else:
        print(f"❌ Error: {main_script} not found")
        return 1
    
    # Check if virtual environment is activated
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Virtual environment is activated")
    else:
        print("⚠️  Warning: Virtual environment not detected")
        print("   Consider activating your virtual environment:")
        print("   source ../.venv/bin/activate")
    
    # Check for required environment variables
    required_vars = ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"⚠️  Warning: Missing environment variables: {', '.join(missing_vars)}")
        print("   These are required for API access")
    else:
        print("✅ Environment variables are set")
    
    # Check for required files
    required_files = [
        "questions.json",
        "characters_abridged.csv",
        "generate_questions.py",
        "run_matches.py",
        "produce_rankings.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not (script_dir / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"⚠️  Warning: Missing files: {', '.join(missing_files)}")
    else:
        print("✅ All required files are present")
    
    print("\n" + "=" * 60)
    print("INSTALLATION COMPLETE!")
    print("=" * 60)
    print("You can now use the package with:")
    print()
    print("  # Generate questions")
    print("  python moral_preferences.py generate-questions --num-questions 20")
    print()
    print("  # Run evaluation (training mode)")
    print("  python moral_preferences.py evaluate --characters characters.csv --model mistral-instruct")
    print()
    print("  # Run evaluation (test mode)")
    print("  python moral_preferences.py evaluate --characters characters.csv --model mistral-instruct --test")
    print()
    print("For more options, run:")
    print("  python moral_preferences.py --help")
    print()
    print("For detailed documentation, see README.md")
    
    return 0

if __name__ == "__main__":
    exit(main()) 