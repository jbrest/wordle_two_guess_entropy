"""
words.py

Handles loading and organizing the Wordle word lists.
No numpy here, just clean text handling.
"""

from pathlib import Path


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
    answers = load_word_list(Path("data/answers.txt"))
    allowed = load_word_list(Path("data/allowed.txt"))
    return answers, allowed