# Moral Preferences Evaluation Package - Overview

## What's New

The package now provides a **simplified interface** that makes it much easier to use while maintaining all the original functionality.

## Quick Start

### 1. Install the Package
```bash
cd scripts
source ../.venv/bin/activate
python install.py
```

### 2. Generate Questions (if needed)
```bash
python moral_preferences.py generate-questions --num-questions 20
```

### 3. Run Evaluation
```bash
# Training mode (just rankings and plots)
python moral_preferences.py evaluate --characters characters.csv --model mistral-instruct

# Test mode (with predictive accuracy evaluation)
python moral_preferences.py evaluate --characters characters.csv --model mistral-instruct --test
```

## Key Improvements

### ✅ Simplified Commands
- **One command** for generating questions
- **One command** for running matches AND producing rankings
- **Default questions file** - no need to specify questions.json every time
- **Smart defaults** - works out of the box

### ✅ Two Modes
- **Training Mode** (default): Generate rankings and plots
- **Test Mode** (with `--test` flag): Split data and evaluate predictive accuracy

### ✅ Better UX
- Clear progress indicators
- Helpful error messages
- Automatic file organization
- Consistent naming conventions

## Command Reference

### Generate Questions
```bash
python moral_preferences.py generate-questions [options]
```

**Options:**
- `--num-questions`: Number of questions (default: 20)
- `--model`: Model for generation (default: gpt-4o-mini)
- `--output-dir`: Output directory (default: logs)
- `--seed`: Random seed
- `--verbose`: Verbose output

### Evaluate
```bash
python moral_preferences.py evaluate [options]
```

**Required:**
- `--characters`: CSV file with character data
- `--model`: Model abbreviation

**Optional:**
- `--questions`: Questions JSON file (default: questions.json in scripts dir)
- `--n-matches`: Matches per pair (default: 10)
- `--output-dir`: Output directory (default: logs)
- `--use-cot`: Use chain of thought
- `--seed`: Random seed
- `--test`: Enable test mode
- `--split`: Training/test split (default: 0.8, only with --test)

## Examples

### Basic Usage
```bash
# Generate 20 questions
python moral_preferences.py generate-questions --num-questions 20

# Run evaluation with default questions
python moral_preferences.py evaluate --characters characters.csv --model mistral-instruct
```

### Advanced Usage
```bash
# Use custom questions
python moral_preferences.py evaluate --characters characters.csv --model mistral-instruct --questions my_questions.json

# Test mode with predictive accuracy
python moral_preferences.py evaluate --characters characters.csv --model mistral-instruct --test --split 0.8

# More matches per pair
python moral_preferences.py evaluate --characters characters.csv --model mistral-instruct --n-matches 20

# Use chain of thought reasoning
python moral_preferences.py evaluate --characters characters.csv --model mistral-instruct --use-cot
```

## What Each Mode Does

### Training Mode (default)
- Runs matches between character pairs
- Generates rankings using ELO, Glicko2, and Bradley-Terry
- Creates visualization plots
- **No predictive accuracy evaluation**

### Test Mode (with --test flag)
- Splits matches into training/testing sets
- Trains ranking models on training data
- Evaluates predictive accuracy on test data
- Reports metrics like MSE, MAE, Accuracy, Log Loss
- **Includes predictive accuracy evaluation**

## File Structure

```
scripts/
├── moral_preferences.py    # Main interface (NEW)
├── install.py             # Installation script (NEW)
├── __init__.py           # Package init (NEW)
├── questions.json         # Default questions
├── characters_abridged.csv # Sample characters
├── generate_questions.py  # Question generation
├── run_matches.py        # Match execution
├── produce_rankings.py   # Ranking generation
├── file_utils.py         # Utilities
└── README.md            # Detailed documentation
```

## Output Organization

All results are automatically organized in timestamped directories:

```
logs/
├── questions/           # Generated question files
├── csv-files/          # Match CSV files
├── results/            # Rankings and metrics
└── evals/             # Model evaluation logs
```

## Backward Compatibility

All original scripts still work:
- `generate_questions.py`
- `run_matches.py`
- `produce_rankings.py`
- `run_full_evaluation.py`

The new `moral_preferences.py` is just a simplified wrapper around these.

## Environment Setup

1. **Activate virtual environment:**
   ```bash
   source ../.venv/bin/activate
   ```

2. **Set environment variables:**
   ```bash
   export OPENAI_API_KEY="your-key"
   export ANTHROPIC_API_KEY="your-key"
   ```

3. **Install dependencies:**
   ```bash
   pip install pandas numpy matplotlib scipy tabulate anthropic openai instructor pydantic
   ```

## Troubleshooting

### Common Issues

1. **"No questions file found"**
   - Generate questions first: `python moral_preferences.py generate-questions`
   - Or specify a questions file: `--questions my_questions.json`

2. **"Characters file not found"**
   - Check the path to your CSV file
   - Make sure it has "character" and "article" columns

3. **API errors**
   - Check your environment variables are set
   - Verify your API keys are valid

4. **Import errors**
   - Make sure you're in the scripts directory
   - Activate the virtual environment

### Getting Help

```bash
# General help
python moral_preferences.py --help

# Command-specific help
python moral_preferences.py generate-questions --help
python moral_preferences.py evaluate --help
```

## Migration from Old Interface

If you were using the old interface:

**Old:**
```bash
python run_full_evaluation.py --characters chars.csv --model mistral-instruct --mode training
```

**New:**
```bash
python moral_preferences.py evaluate --characters chars.csv --model mistral-instruct
```

**Old:**
```bash
python run_full_evaluation.py --characters chars.csv --model mistral-instruct --split 0.8
```

**New:**
```bash
python moral_preferences.py evaluate --characters chars.csv --model mistral-instruct --test --split 0.8
```

The new interface is simpler and more intuitive! 