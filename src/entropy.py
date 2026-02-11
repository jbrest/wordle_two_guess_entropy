"""
entropy.py

Contains entropy calculations for single guesses and pairs of guesses.
"""

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover - optional acceleration
    njit = None


TRIPLE_CODE_SPACE = 243**3
_TRIPLE_COUNTS = None
_TRIPLE_TOUCHED = None


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


if njit is not None:

    @njit(cache=True)
    def _three_guess_entropy_hist_numba(joint, counts, touched):
        touched_n = 0

        for idx in range(joint.size):
            code = int(joint[idx])
            if counts[code] == 0:
                touched[touched_n] = code
                touched_n += 1
            counts[code] += 1

        inv_total = 1.0 / joint.size
        entropy = 0.0

        for idx in range(touched_n):
            code = touched[idx]
            count = counts[code]
            prob = count * inv_total
            entropy -= prob * np.log2(prob)
            counts[code] = 0

        return entropy


def _ensure_triple_buffers(n_answers: int):
    """Ensure reusable buffers are sized for current answer count."""
    global _TRIPLE_COUNTS, _TRIPLE_TOUCHED

    if _TRIPLE_COUNTS is None:
        _TRIPLE_COUNTS = np.zeros(TRIPLE_CODE_SPACE, dtype=np.uint32)

    if _TRIPLE_TOUCHED is None or _TRIPLE_TOUCHED.size < n_answers:
        _TRIPLE_TOUCHED = np.empty(n_answers, dtype=np.int64)

    return _TRIPLE_COUNTS, _TRIPLE_TOUCHED


def three_guess_entropy(row12_scaled, row3):
    """
    Joint entropy of three guesses.
    Treat (pattern1, pattern2, pattern3) tuples as a combined outcome.

    Uses histogram counting over encoded 3-guess outcome codes.
    This avoids per-call sorting and enables optional Numba acceleration.
    """
    joint = row12_scaled + row3

    if njit is None:
        _, counts = np.unique(joint, return_counts=True)
        return entropy_from_counts(counts)

    counts, touched = _ensure_triple_buffers(joint.size)
    return _three_guess_entropy_hist_numba(joint, counts, touched)
