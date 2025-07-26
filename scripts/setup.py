#!/usr/bin/env python3
"""
Setup script for the Moral Preference Evaluation Package.

This package provides tools for evaluating moral preference in AI models through
character-based comparisons.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return "Moral Preference Evaluation Package"

# Read requirements
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    if os.path.exists(requirements_path):
        with open(requirements_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return [
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "matplotlib>=3.3.0",
        "scipy>=1.7.0",
        "tabulate>=0.8.0",
        "anthropic>=0.7.0",
        "openai>=1.0.0",
        "instructor>=0.4.0",
        "pydantic>=2.0.0",
        "python-dotenv>=0.19.0",
    ]

setup(
    name="moral-preference-eval",
    version="0.1.0",
    description="Tools for evaluating moral preference in AI models",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/moral-preference-eval",
    packages=find_packages(),
    py_modules=[
        "generate_questions",
        "run_matches", 
        "produce_rankings",
        "run_full_evaluation"
    ],
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=21.0.0",
            "flake8>=3.8.0",
            "mypy>=0.800",
        ],
    },
    entry_points={
        "console_scripts": [
            "moralrank=run_full_evaluation:main",
            "moralrank-generate=generate_questions:main",
            "moralrank-matches=run_matches:main", 
            "moralrank-rank=produce_rankings:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    include_package_data=True,
    zip_safe=False,
) 