"""
Package entry point for python -m support.

This allows the project to be run as:
    python -m .                    # Predict next matchday (default)
    python -m . predict             # Explicit prediction
    python -m . <subcommand>        # Run other commands

Instead of:
    python main.py
    python main.py predict
    python main.py <subcommand>
"""

from main import main

if __name__ == '__main__':
    main()

