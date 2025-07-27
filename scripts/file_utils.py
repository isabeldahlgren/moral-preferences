#!/usr/bin/env python3
"""
Utility functions for consistent file naming and organization.
"""

import os
import time
import hashlib
from datetime import datetime
from typing import Optional


def generate_timestamp() -> str:
    """Generate a timestamp string for file naming."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def generate_run_id() -> str:
    """Generate a unique run ID for this execution."""
    timestamp = int(time.time())
    random_suffix = hashlib.md5(str(timestamp).encode()).hexdigest()[:8]
    return f"{timestamp}_{random_suffix}"


def sanitize_filename(filename: str) -> str:
    """Sanitize a filename by removing/replacing invalid characters."""
    # Replace invalid characters with underscores
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Remove leading/trailing spaces and dots
    filename = filename.strip('. ')
    
    # Limit length
    if len(filename) > 100:
        filename = filename[:100]
    
    return filename


def generate_questions_filename(
    num_questions: int,
    run_id: Optional[str] = None,
    timestamp: Optional[str] = None
) -> str:
    """
    Generate a consistent filename for question files.
    
    Args:
        num_questions: Number of questions generated
        run_id: Optional run ID (auto-generated if not provided)
        timestamp: Optional timestamp (auto-generated if not provided)
        
    Returns:
        Filename like: questions_10_20241201_143022_abc12345.json
    """
    if timestamp is None:
        timestamp = generate_timestamp()
    if run_id is None:
        run_id = generate_run_id()
    
    filename = f"questions_{num_questions}_{timestamp}_{run_id}.json"
    return filename


def generate_matches_filename(
    model: str,
    mode: str,
    n_matches: int,
    run_id: Optional[str] = None,
    timestamp: Optional[str] = None,
    split: Optional[float] = None
) -> str:
    """
    Generate a consistent filename for match files.
    
    Args:
        model: Model abbreviation
        mode: training/testing/split
        n_matches: Number of matches per character pair
        run_id: Optional run ID (auto-generated if not provided)
        timestamp: Optional timestamp (auto-generated if not provided)
        split: Split ratio if using split mode
        
    Returns:
        Filename like: matches_mistral-instruct_train_10_20241201_143022_abc12345.csv
        or: matches_mistral-instruct_split08_10_20241201_143022_abc12345_train.csv
    """
    if timestamp is None:
        timestamp = generate_timestamp()
    if run_id is None:
        run_id = generate_run_id()
    
    model_safe = sanitize_filename(model)
    
    if split is not None:
        # Split mode: create separate train/test files
        split_str = f"split{int(split*100):02d}"
        base_filename = f"matches_{model_safe}_{split_str}_{n_matches}_{timestamp}_{run_id}"
        return f"{base_filename}_{mode}.csv"
    else:
        # Single mode
        return f"matches_{model_safe}_{mode}_{n_matches}_{timestamp}_{run_id}.csv"


def generate_rankings_filename(
    model: str,
    run_id: Optional[str] = None,
    timestamp: Optional[str] = None
) -> str:
    """
    Generate a consistent filename for ranking files.
    
    Args:
        model: Model abbreviation
        run_id: Optional run ID (auto-generated if not provided)
        timestamp: Optional timestamp (auto-generated if not provided)
        
    Returns:
        Filename like: rankings_mistral-instruct_20241201_143022_abc12345.csv
    """
    if timestamp is None:
        timestamp = generate_timestamp()
    if run_id is None:
        run_id = generate_run_id()
    
    model_safe = sanitize_filename(model)
    return f"rankings_{model_safe}_{timestamp}_{run_id}.csv"


def create_output_directories(base_dir: str = "logs") -> dict:
    """
    Create organized output directories under logs/.
    
    Args:
        base_dir: Base directory for all results (default: logs)
        
    Returns:
        Dictionary with paths to different output directories
    """
    timestamp = generate_timestamp()
    run_id = generate_run_id()
    
    dirs = {
        "base": base_dir,
        "questions": os.path.join(base_dir, "questions"),
        "evals": os.path.join(base_dir, "evals"),
        "csv_files": os.path.join(base_dir, "csv-files"),
        "results": os.path.join(base_dir, "results"),
        "timestamp": timestamp,
        "run_id": run_id
    }
    
    # Create directories
    for key, path in dirs.items():
        if key not in ["timestamp", "run_id"]:
            os.makedirs(path, exist_ok=True)
    
    return dirs


def get_file_metadata(filename: str) -> dict:
    """
    Extract metadata from a filename.
    
    Args:
        filename: Filename to parse
        
    Returns:
        Dictionary with extracted metadata
    """
    metadata = {}
    
    # Parse timestamp and run_id from filename
    parts = filename.split('_')
    if len(parts) >= 3:
        # Look for timestamp pattern (YYYYMMDD_HHMMSS)
        for i, part in enumerate(parts):
            if len(part) == 8 and part.isdigit():  # Date part
                if i + 1 < len(parts) and len(parts[i + 1]) == 6 and parts[i + 1].isdigit():  # Time part
                    metadata["timestamp"] = f"{part}_{parts[i + 1]}"
                    if i + 2 < len(parts):
                        metadata["run_id"] = parts[i + 2].split('.')[0]  # Remove extension
                    break
    
    # Extract other metadata based on filename pattern
    if filename.startswith("questions_"):
        metadata["type"] = "questions"
        parts = filename.split('_')
        if len(parts) >= 3:
            metadata["model"] = parts[1]
            metadata["num_questions"] = int(parts[2]) if parts[2].isdigit() else None
    elif filename.startswith("matches_"):
        metadata["type"] = "matches"
        parts = filename.split('_')
        if len(parts) >= 4:
            metadata["model"] = parts[1]
            metadata["mode"] = parts[2]
            metadata["n_matches"] = int(parts[3]) if parts[3].isdigit() else None
    elif filename.startswith("rankings_"):
        metadata["type"] = "rankings"
        parts = filename.split('_')
        if len(parts) >= 2:
            metadata["model"] = parts[1]
    
    return metadata 