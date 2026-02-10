"""
wordle_entropy.py

Unified CLI for Wordle entropy analysis.

Modes:
-words 1 (default): top single-guess entropy words
-words 2: top two-guess non-adaptive entropy pairs

Optional:
-verbose (used only with -words 2): show individual entropies and first-word
cost vs the globally best standalone opener.
"""

import argparse
import heapq
import time

import numpy as np
from tqdm import tqdm

from src.entropy import single_guess_entropy, two_guess_entropy
from src.patterns import load_or_build_matrix
from src.words import load_words


TOP_SINGLE = 20
TOP_PAIRS = 50


def run_single_guess(answers, allowed, matrix):
    answer_set = set(answers)

    print("Computing single-guess entropies...")
    entropies = np.array([single_guess_entropy(matrix[i]) for i in range(len(allowed))])
    top_indices = np.argsort(entropies)[-TOP_SINGLE:][::-1]

    print("\nTop single guesses:")
    print("Legend: word [flag]: entropy bits")
    print("flag: [+] in answers.txt, [-] guess-only in allowed.txt")
    for idx in top_indices:
        word = allowed[idx]
        answer_flag = "+" if word in answer_set else "-"
        print(f"{word} [{answer_flag}]: {entropies[idx]:.4f} bits")


def run_two_guess(answers, allowed, matrix, verbose):
    answer_set = set(answers)
    n_allowed = len(allowed)

    print("Computing single guess entropies...")
    single_entropies = np.array(
        [single_guess_entropy(matrix[i]) for i in range(n_allowed)]
    )
    best_single_entropy = float(np.max(single_entropies))
    matrix_u16_243 = matrix.astype(np.uint16) * 243
    sorted_indices = np.argsort(single_entropies)[::-1]
    sorted_entropies = single_entropies[sorted_indices]

    best_pairs = []
    start_time = time.time()
    early_exit_reason = None

    print("Starting optimized full two guess search...\n")

    for pos_i in tqdm(range(n_allowed), desc="First guess"):
        i = sorted_indices[pos_i]
        row1_scaled = matrix_u16_243[i]
        h1 = sorted_entropies[pos_i]

        if len(best_pairs) == TOP_PAIRS and pos_i + 1 < n_allowed:
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

        for pos_j in range(pos_i + 1, n_allowed):
            j = sorted_indices[pos_j]
            h2 = sorted_entropies[pos_j]

            if len(best_pairs) == TOP_PAIRS and (h1 + h2) <= best_pairs[0][0]:
                break

            h12 = two_guess_entropy(row1_scaled, matrix[j])

            if len(best_pairs) < TOP_PAIRS:
                heapq.heappush(best_pairs, (h12, i, j))
            elif h12 > best_pairs[0][0]:
                heapq.heapreplace(best_pairs, (h12, i, j))

        if pos_i % 50 == 0 and pos_i > 0:
            elapsed = time.time() - start_time
            rate = pos_i / elapsed
            remaining = (n_allowed - pos_i) / rate if rate > 0 else 0
            tqdm.write(
                f"Processed {pos_i}/{n_allowed} first guesses. "
                f"Elapsed: {elapsed/60:.1f} min, ETA: {remaining/60:.1f} min"
            )

    best_pairs.sort(reverse=True)

    if early_exit_reason is not None:
        print(early_exit_reason)

    print("\nTop two guess pairs (non-adaptive, exact):")
    if verbose:
        print(
            "Legend: word1 + word2 [flags]: H12 bits (H1, H2) | "
            "Cost: (H_best_single - H1) bits"
        )
    else:
        print("Legend: word1 + word2 [flags]: H12 bits")
    print("flags: [++] both answers, [+-] first only, [-+] second only, [--] neither")

    for h12, i, j in best_pairs:
        word_i = allowed[i]
        word_j = allowed[j]
        flag_i = "+" if word_i in answer_set else "-"
        flag_j = "+" if word_j in answer_set else "-"

        if verbose:
            h1 = single_entropies[i]
            h2 = single_entropies[j]
            cost = best_single_entropy - h1
            print(
                f"{word_i} + {word_j} [{flag_i}{flag_j}]: {h12:.4f} bits "
                f"({h1:.4f}, {h2:.4f}) | Cost: {cost:.4f} bits"
            )
        else:
            print(f"{word_i} + {word_j} [{flag_i}{flag_j}]: {h12:.4f} bits")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Wordle entropy analyzer for one-guess and two-guess modes."
    )
    parser.add_argument(
        "-words",
        type=int,
        choices=(1, 2),
        default=1,
        help="Number of opening words to optimize (default: 1).",
    )
    parser.add_argument(
        "-verbose",
        action="store_true",
        help="For -words 2, show individual entropies and first-word cost.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    answers, allowed = load_words()
    matrix = load_or_build_matrix(allowed, answers)

    if args.words == 1:
        run_single_guess(answers, allowed, matrix)
        return

    run_two_guess(answers, allowed, matrix, args.verbose)


if __name__ == "__main__":
    main()
