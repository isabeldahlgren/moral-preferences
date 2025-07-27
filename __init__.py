"""
Moral Preferences Evaluation Package

A package for evaluating moral preferences in AI models through character-based comparisons.

Main interface:
    python moral_preferences.py generate-questions [options]
    python moral_preferences.py evaluate [options]

For detailed usage, see README.md
"""

__version__ = "1.0.0"
__author__ = "Moral Preferences Team"

# Import main functions for programmatic access
from .generate_questions import generate_questions
from .run_matches import run_matches
from .produce_rankings import produce_rankings

__all__ = [
    "generate_questions",
    "run_matches", 
    "produce_rankings"
] 