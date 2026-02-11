"""
entropy.py

Contains entropy calculations for single guesses and pairs of guesses.
"""

import numpy as np


def entropy_from_counts(counts):
    """Compute Shannon entropy from bucket counts."""
    total = counts.sum()
    probs = counts[counts > 0] / total
    return -np.sum(probs * np.log2(probs))


def single_guess_entropy(matrix_row):
    """Entropy of one guess across all answers."""
    counts = np.bincount(matrix_row)
    return entropy_from_counts(counts)


def two_guess_entropy(row1_scaled, row2):
    """
    Joint entropy of two guesses.
    Treat (pattern1, pattern2) pairs as combined outcome.
    """
    joint = row1_scaled + row2
    counts = np.bincount(joint)
    return entropy_from_counts(counts)


def three_guess_entropy(row12_scaled, row3):
    """
    Joint entropy of three guesses.
    Treat (pattern1, pattern2, pattern3) tuples as a combined outcome.

    Uses unique-value counting instead of bincount to avoid allocating very
    large intermediate arrays for 3-guess code ranges.
    """
    joint = row12_scaled + row3
    _, counts = np.unique(joint, return_counts=True)
    return entropy_from_counts(counts)
