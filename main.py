"""
main.py

Computes entropy for every single allowed guess.
"""

from src.words import load_words
from src.patterns import load_or_build_matrix
from src.entropy import single_guess_entropy
import numpy as np


answers, allowed = load_words()
matrix = load_or_build_matrix(allowed, answers)

print("Computing single-guess entropies...")
entropies = [single_guess_entropy(matrix[i]) for i in range(len(allowed))]

top_indices = np.argsort(entropies)[-20:][::-1]

print("\nTop single guesses:")
for idx in top_indices:
    print(f"{allowed[idx]}: {entropies[idx]:.4f} bits")