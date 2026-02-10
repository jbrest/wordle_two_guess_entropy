"""
words.py

Handles loading and organizing the Wordle word lists.
No numpy here, just clean text handling.
"""

from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_word_list(path):
    """Load a newline-separated word list into a Python list."""
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def load_words():
    """
    Returns:
        answers: list of possible solution words
        allowed: list of valid guess words (includes answers)
    """
    # Use paths relative to this source tree so execution is robust even when
    # Python is launched from a different current working directory.
    answers = load_word_list(DATA_DIR / "answers.txt")
    allowed = load_word_list(DATA_DIR / "allowed.txt")
    return answers, allowed
