#!/usr/bin/env python3
"""
3. Liga Match Predictor - Main Entry Point

Modern entry point using Typer CLI framework.

Usage:
    python main.py                    # Predict next matchday (default)
    python main.py predict --help     # Show prediction options
    python main.py <command>          # Run other commands
    python main.py --help             # List all available commands

Or with Poetry:
    poetry run liga-predictor         # After poetry install
"""

from liga_predictor.cli import app

if __name__ == "__main__":
    app()
