"""
two_guess_main.py

Performs a full optimized two guess entropy search.

This finds the best non adaptive two word opening pairs for Wordle by maximizing
joint entropy over all possible answers.

Optimizations included:

1. Symmetry reduction
   We only evaluate pairs (i, j) where j > i.
   Entropy of (A, B) equals entropy of (B, A), so half the work is removed.

2. Entropy upper bound pruning
   H(A, B) ≤ H(A) + H(B)
   If this upper bound cannot beat the worst pair in our Top K list,
   we skip the expensive joint entropy calculation.

3. Fixed size min heap for Top K tracking
   Maintains only the best K pairs at all times and provides a fast cutoff
   for pruning.

4. Progress reporting with elapsed time and ETA
"""

import time
import heapq
import numpy as np
from tqdm import tqdm

from src.words import load_words
from src.patterns import load_or_build_matrix
from src.entropy import single_guess_entropy, two_guess_entropy


TOP_K = 50


def main():
    # Load word lists
    answers, allowed = load_words()
    n_allowed = len(allowed)

    # Load or build pattern matrix
    matrix = load_or_build_matrix(allowed, answers)

    # Precompute single guess entropies
    print("Computing single guess entropies...")
    single_entropies = np.array([
        single_guess_entropy(matrix[i]) for i in range(n_allowed)
    ])
    matrix_u16_243 = matrix.astype(np.uint16) * 243
    sorted_indices = np.argsort(single_entropies)[::-1]
    sorted_entropies = single_entropies[sorted_indices]

    # Min heap storing (entropy, i, j)
    best_pairs = []

    start_time = time.time()

    print("Starting optimized full two guess search...\n")

    early_exit_reason = None

    for pos_i in tqdm(range(n_allowed), desc="First guess"):
        i = sorted_indices[pos_i]
        row1_scaled = matrix_u16_243[i]
        h1 = sorted_entropies[pos_i]

        if len(best_pairs) == TOP_K and pos_i + 1 < n_allowed:
            next_h2 = sorted_entropies[pos_i + 1]
            cutoff = best_pairs[0][0]
            upper_bound = h1 + next_h2
            if upper_bound <= cutoff:
                early_exit_reason = (
                    f"Early exit at outer index {pos_i}: "
                    f"best possible remaining pair upper bound "
                    f"{upper_bound:.6f} <= current floor {cutoff:.6f}."
                )
                break

        for pos_j in range(pos_i + 1, n_allowed):  # Symmetry reduction
            j = sorted_indices[pos_j]
            h2 = sorted_entropies[pos_j]

            # Upper bound pruning
            if len(best_pairs) == TOP_K and (h1 + h2) <= best_pairs[0][0]:
                break

            # Compute true joint entropy
            h12 = two_guess_entropy(row1_scaled, matrix[j])

            if len(best_pairs) < TOP_K:
                heapq.heappush(best_pairs, (h12, i, j))
            elif h12 > best_pairs[0][0]:
                heapq.heapreplace(best_pairs, (h12, i, j))

        # Progress reporting every outer iteration
        if pos_i % 50 == 0 and pos_i > 0:
            elapsed = time.time() - start_time
            rate = pos_i / elapsed
            remaining = (n_allowed - pos_i) / rate if rate > 0 else 0
            tqdm.write(
                f"Processed {pos_i}/{n_allowed} first guesses. "
                f"Elapsed: {elapsed/60:.1f} min, ETA: {remaining/60:.1f} min"
            )

    # Sort final results descending
    best_pairs.sort(reverse=True)

    if early_exit_reason is not None:
        print(early_exit_reason)

    print("\nTop two guess pairs (non adaptive, exact):\n")
    for entropy, i, j in best_pairs:
        print(f"{allowed[i]} + {allowed[j]}: {entropy:.4f} bits")


if __name__ == "__main__":
    main()
