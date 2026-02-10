"""
patterns.py

Builds and stores the Wordle feedback pattern matrix.

Matrix shape:
    (n_allowed_guesses, n_answers)

Each cell contains an integer 0..242 encoding the 5-tile Wordle feedback
pattern in base-3:

    0 = gray
    1 = yellow
    2 = green

The matrix allows entropy calculations to be performed extremely quickly,
since all guess/answer feedback is precomputed once.
"""

from collections import Counter
from pathlib import Path
import numpy as np
from tqdm import tqdm


# Location where the pattern matrix is stored
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MATRIX_PATH = DATA_DIR / "pattern_matrix.npy"
ANSWERS_PATH = DATA_DIR / "answers.txt"
ALLOWED_PATH = DATA_DIR / "allowed.txt"


def encode_pattern(guess: str, answer: str) -> int:
    """
    Encode Wordle feedback for a (guess, answer) pair as a base-3 integer.

    This implementation matches standard Wordle duplicate-letter rules:

    1. First mark greens (correct letter in correct position).
       Each green consumes one instance of that letter from the answer.

    2. Then mark yellows (correct letter, wrong position) only if
       remaining unused instances of that letter exist in the answer.

    This Counter-based approach ensures behavior is identical to the
    original working version of the project.
    """
    result = [0] * 5
    counts = Counter(answer)

    # First pass: mark greens and consume letters
    for i in range(5):
        if guess[i] == answer[i]:
            result[i] = 2
            counts[guess[i]] -= 1

    # Second pass: mark yellows where letters remain unused
    for i in range(5):
        if result[i] == 0 and counts[guess[i]] > 0:
            result[i] = 1
            counts[guess[i]] -= 1

    # Convert base-3 digit list to a single integer code
    code = 0
    for r in result:
        code = code * 3 + r

    return code


def build_matrix(allowed: list[str], answers: list[str]) -> np.ndarray:
    """
    Compute the full pattern matrix from scratch.

    This is the most expensive step in the project, but it only needs to be
    done once per word list. Progress is shown so long builds don't look stuck.
    """
    n_allowed = len(allowed)
    n_answers = len(answers)

    matrix = np.zeros((n_allowed, n_answers), dtype=np.uint8)

    print("Building pattern matrix...")
    for i, guess in enumerate(tqdm(allowed)):
        for j, answer in enumerate(answers):
            matrix[i, j] = encode_pattern(guess, answer)

    np.save(MATRIX_PATH, matrix)
    print("Matrix saved to disk.")

    return matrix


def load_or_build_matrix(allowed: list[str], answers: list[str]) -> np.ndarray:
    """
    Load a previously built pattern matrix if it matches current dimensions.

    If the matrix file is missing or its shape does not match the current
    word lists, it is automatically rebuilt to ensure correctness.
    """
    n_allowed = len(allowed)
    n_answers = len(answers)

    if MATRIX_PATH.exists():
        matrix = np.load(MATRIX_PATH)

        # Guard 1: matrix dimensions must match active word lists.
        shape_ok = matrix.shape == (n_allowed, n_answers)

        # Guard 2: if either source word file is newer than the matrix cache,
        # assume the matrix is stale and force a rebuild.
        matrix_mtime = MATRIX_PATH.stat().st_mtime
        answers_mtime = ANSWERS_PATH.stat().st_mtime
        allowed_mtime = ALLOWED_PATH.stat().st_mtime
        cache_is_new_enough = (
            matrix_mtime >= answers_mtime and matrix_mtime >= allowed_mtime
        )

        if shape_ok and cache_is_new_enough:
            print("Loaded compatible pattern matrix from disk.")
            return matrix

        if not shape_ok:
            print("Matrix shape mismatch. Rebuilding.")
        else:
            print("Matrix is older than word lists. Rebuilding.")

    return build_matrix(allowed, answers)
