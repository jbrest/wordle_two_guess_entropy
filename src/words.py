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


def load_words(answers_path=None, allowed_path=None):
    """
    Returns:
        answers: list of possible solution words
        allowed: list of valid guess words
    """
    # Use paths relative to this source tree by default so execution is robust
    # even when Python is launched from a different current working directory.
    answers_file = Path(answers_path) if answers_path is not None else (DATA_DIR / "answers.txt")
    allowed_file = Path(allowed_path) if allowed_path is not None else (DATA_DIR / "allowed.txt")
    answers = load_word_list(answers_file)
    allowed = load_word_list(allowed_file)
    return answers, allowed
