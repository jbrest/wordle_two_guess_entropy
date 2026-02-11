"""
wordle_entropy.py

Unified CLI for Wordle entropy analysis.

Modes:
-words 1 (default): top single-guess entropy words
-words 2: top two-guess non-adaptive entropy pairs
-words 3: top three-guess non-adaptive entropy triples (very expensive)

Optional:
-verbose: show individual entropies and first-word cost for pair/triple outputs.
-pair WORD1 WORD2: evaluate one specific two-guess pair; overrides -words.
-triple WORD1 WORD2 WORD3: evaluate one specific three-guess combo;
  overrides -words.
-force: skip confirmation prompt for -words 3.
"""

import argparse
import heapq
import threading
import time

import numpy as np
from tqdm import tqdm

from src.entropy import single_guess_entropy, three_guess_entropy, two_guess_entropy
from src.patterns import load_or_build_matrix
from src.words import load_words


TOP_SINGLE = 20
TOP_PAIRS = 50
TOP_TRIPLES = 50
TRIPLE_STATUS_SECONDS = 1.0


def _pct(cur, total):
    if total <= 0:
        return 0.0
    return 100.0 * cur / total


def _fmt_frac(cur, total, width):
    return f"{cur:>{width},}/{total:>{width},}"


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


def run_specific_pair(answers, allowed, matrix, word1, word2, verbose):
    answer_set = set(answers)

    try:
        i = allowed.index(word1)
    except ValueError as exc:
        raise ValueError(f"word not found in allowed list: {word1}") from exc

    try:
        j = allowed.index(word2)
    except ValueError as exc:
        raise ValueError(f"word not found in allowed list: {word2}") from exc

    single_entropies = np.array([single_guess_entropy(matrix[k]) for k in range(len(allowed))])
    best_single_entropy = float(np.max(single_entropies))
    h1 = float(single_entropies[i])
    h2 = float(single_entropies[j])
    row1_scaled = matrix[i].astype(np.uint16) * 243
    h12 = two_guess_entropy(row1_scaled, matrix[j])
    cost = best_single_entropy - h1

    flag_i = "+" if word1 in answer_set else "-"
    flag_j = "+" if word2 in answer_set else "-"

    print("\nSpecific two guess pair (non-adaptive, exact):")
    if verbose:
        print(
            "Legend: word1 + word2 [flags]: H12 bits (H1, H2) | "
            "Cost: (H_best_single - H1) bits"
        )
    else:
        print("Legend: word1 + word2 [flags]: H12 bits")
    print("flags: [++] both answers, [+-] first only, [-+] second only, [--] neither")
    if verbose:
        print(
            f"{word1} + {word2} [{flag_i}{flag_j}]: {h12:.4f} bits "
            f"({h1:.4f}, {h2:.4f}) | Cost: {cost:.4f} bits"
        )
    else:
        print(f"{word1} + {word2} [{flag_i}{flag_j}]: {h12:.4f} bits")


def run_three_guess(answers, allowed, matrix, verbose):
    answer_set = set(answers)
    n_allowed = len(allowed)

    print("Computing single guess entropies...")
    single_entropies = np.array(
        [single_guess_entropy(matrix[i]) for i in range(n_allowed)]
    )
    best_single_entropy = float(np.max(single_entropies))
    matrix_u32 = matrix.astype(np.uint32)
    matrix_u32_243 = matrix_u32 * 243
    matrix_u32_59049 = matrix_u32 * 59049
    sorted_indices = np.argsort(single_entropies)[::-1]
    sorted_entropies = single_entropies[sorted_indices]
    max_entropy = float(np.log2(len(answers)))

    best_triples = []
    start_time = time.time()
    early_exit_reason = None
    outer_total = n_allowed - 2

    progress = {
        "a_cur": 0,
        "a_tot": outer_total,
        "b_cur": 0,
        "b_tot": 0,
        "c_cur": 0,
        "c_tot": 0,
        "floor": None,
        "possible_j_completed": 0,
        "actual_j_completed": 0,
        "possible_k_completed": 0,
        "actual_k_completed": 0,
    }
    stop_event = threading.Event()
    print("Starting optimized full three guess search...\n")
    w1 = len(f"{max(1, outer_total):,}")
    w2 = len(f"{max(1, n_allowed - 2):,}")
    w3 = len(f"{max(1, n_allowed - 3):,}")
    progress_bar = tqdm(
        total=0,
        bar_format="{desc}",
        desc=(
            f"1st: {_fmt_frac(0, max(1, outer_total), w1)} ({0.0:5.1f}%) | "
            f"2nd: {_fmt_frac(0, max(1, n_allowed - 2), w2)} ({0.0:5.1f}%) | "
            f"3rd: {_fmt_frac(0, 1, w3)} ({0.0:5.1f}%)"
        ),
        position=0,
    )
    stats_bar = tqdm(
        total=0,
        bar_format="{desc}",
        desc=(
            "Prune efficiency: 2nd N/A | 3rd N/A | "
            "Current max: warming | Elapsed: 0.0m | "
            "Speed: 0 triple-evals/s | Checked: 0"
        ),
        position=1,
        leave=False,
    )

    def monitor():
        while not stop_event.wait(TRIPLE_STATUS_SECONDS):
            p = progress
            elapsed = time.time() - start_time
            a_tot = max(1, p["a_tot"])
            b_tot = max(1, p["b_tot"])
            c_tot = max(1, p["c_tot"])
            line1 = (
                f"1st: {_fmt_frac(p['a_cur'], a_tot, w1)} ({_pct(p['a_cur'], a_tot):5.1f}%) | "
                f"2nd: {_fmt_frac(p['b_cur'], b_tot, w2)} ({_pct(p['b_cur'], b_tot):5.1f}%) | "
                f"3rd: {_fmt_frac(p['c_cur'], c_tot, w3)} ({_pct(p['c_cur'], c_tot):5.1f}%)"
            )
            progress_bar.set_description_str(line1)

            y_text = "N/A"
            if p["possible_j_completed"] > 0:
                y = 100.0 * (
                    1.0 - (p["actual_j_completed"] / p["possible_j_completed"])
                )
                y_text = f"{y:5.1f}% skipped"

            z_text = "N/A"
            if p["possible_k_completed"] > 0:
                z = 100.0 * (
                    1.0 - (p["actual_k_completed"] / p["possible_k_completed"])
                )
                z_text = f"{z:5.1f}% skipped"

            floor = p["floor"]
            floor_text = f"{floor:.4f} bits" if floor is not None else "warming"
            speed = p["actual_k_completed"] / elapsed if elapsed > 0 else 0.0
            line2 = (
                f"Prune efficiency: 2nd {y_text} | 3rd {z_text} | "
                f"Current max: {floor_text} | Elapsed: {elapsed/60:.1f}m | "
                f"Speed: {speed:,.0f} triple-evals/s | Checked: {p['actual_k_completed']:,}"
            )
            stats_bar.set_description_str(line2)

            progress_bar.refresh()
            stats_bar.refresh()

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()

    try:
        for pos_i in range(outer_total):
            progress["a_cur"] = pos_i + 1
            progress["b_cur"] = 0
            progress["c_cur"] = 0
            progress["b_tot"] = n_allowed - pos_i - 2
            progress["c_tot"] = 0
            outer_possible_j = max(0, n_allowed - pos_i - 2)
            outer_actual_j = 0
            i = sorted_indices[pos_i]
            h1 = sorted_entropies[pos_i]
            row1_scaled = matrix_u32_59049[i]

            if len(best_triples) == TOP_TRIPLES and pos_i + 2 < n_allowed:
                cutoff = best_triples[0][0]
                upper_bound = min(
                    max_entropy,
                    h1 + sorted_entropies[pos_i + 1] + sorted_entropies[pos_i + 2],
                )
                if upper_bound <= cutoff:
                    early_exit_reason = (
                        f"Early exit at outer index {pos_i}: "
                        f"best possible remaining triple upper bound "
                        f"{upper_bound:.6f} <= current floor {cutoff:.6f}."
                    )
                    break

            for pos_j in range(pos_i + 1, n_allowed - 1):
                outer_actual_j += 1
                progress["b_cur"] = pos_j - pos_i
                progress["c_cur"] = 0
                progress["c_tot"] = n_allowed - pos_j - 1
                middle_possible_k = max(0, n_allowed - pos_j - 1)
                middle_actual_k = 0
                j = sorted_indices[pos_j]
                h2 = sorted_entropies[pos_j]

                if len(best_triples) == TOP_TRIPLES and pos_j + 1 < n_allowed:
                    cutoff = best_triples[0][0]
                    upper_bound = min(max_entropy, h1 + h2 + sorted_entropies[pos_j + 1])
                    if upper_bound <= cutoff:
                        break

                # Compute pair entropy once for this (i, j) branch and use it for
                # tighter safe bounds on all k in the branch:
                # H123 <= H12 + H3 and H123 <= log2(|answers|).
                h12 = two_guess_entropy(matrix_u32_243[i], matrix_u32[j])
                if len(best_triples) == TOP_TRIPLES and pos_j + 1 < n_allowed:
                    cutoff = best_triples[0][0]
                    upper_bound = min(max_entropy, h12 + sorted_entropies[pos_j + 1])
                    if upper_bound <= cutoff:
                        break

                row12_scaled = row1_scaled + matrix_u32_243[j]

                for pos_k in range(pos_j + 1, n_allowed):
                    if pos_k % 2048 == 0:
                        progress["c_cur"] = pos_k - pos_j
                    h3 = sorted_entropies[pos_k]
                    if len(best_triples) == TOP_TRIPLES and min(max_entropy, h12 + h3) <= best_triples[
                        0
                    ][0]:
                        break

                    k = sorted_indices[pos_k]
                    h123 = three_guess_entropy(row12_scaled, matrix_u32[k])
                    middle_actual_k += 1

                    if len(best_triples) < TOP_TRIPLES:
                        heapq.heappush(best_triples, (h123, i, j, k))
                    elif h123 > best_triples[0][0]:
                        heapq.heapreplace(best_triples, (h123, i, j, k))

                progress["possible_k_completed"] += middle_possible_k
                progress["actual_k_completed"] += middle_actual_k
                if len(best_triples) == TOP_TRIPLES:
                    progress["floor"] = best_triples[0][0]
                progress["c_cur"] = progress["c_tot"]

            progress["possible_j_completed"] += outer_possible_j
            progress["actual_j_completed"] += outer_actual_j
    finally:
        stop_event.set()
        monitor_thread.join(timeout=1.0)
        stats_bar.close()
        progress_bar.close()

    best_triples.sort(reverse=True)

    if early_exit_reason is not None:
        print(early_exit_reason)

    print("\nTop three guess triples (non-adaptive, exact):")
    if verbose:
        print(
            "Legend: word1 + word2 + word3 [flags]: H123 bits (H1, H2, H3) | "
            "Cost: (H_best_single - H1) bits"
        )
    else:
        print("Legend: word1 + word2 + word3 [flags]: H123 bits")
    print("flags: [+++] all answers, mixed +/- indicate answer membership by position")

    for h123, i, j, k in best_triples:
        word_i = allowed[i]
        word_j = allowed[j]
        word_k = allowed[k]
        flag_i = "+" if word_i in answer_set else "-"
        flag_j = "+" if word_j in answer_set else "-"
        flag_k = "+" if word_k in answer_set else "-"

        if verbose:
            h1 = single_entropies[i]
            h2 = single_entropies[j]
            h3 = single_entropies[k]
            cost = best_single_entropy - h1
            print(
                f"{word_i} + {word_j} + {word_k} [{flag_i}{flag_j}{flag_k}]: "
                f"{h123:.4f} bits ({h1:.4f}, {h2:.4f}, {h3:.4f}) | Cost: {cost:.4f} bits"
            )
        else:
            print(f"{word_i} + {word_j} + {word_k} [{flag_i}{flag_j}{flag_k}]: {h123:.4f} bits")


def run_specific_triple(answers, allowed, matrix, word1, word2, word3, verbose):
    answer_set = set(answers)

    try:
        i = allowed.index(word1)
    except ValueError as exc:
        raise ValueError(f"word not found in allowed list: {word1}") from exc

    try:
        j = allowed.index(word2)
    except ValueError as exc:
        raise ValueError(f"word not found in allowed list: {word2}") from exc

    try:
        k = allowed.index(word3)
    except ValueError as exc:
        raise ValueError(f"word not found in allowed list: {word3}") from exc

    single_entropies = np.array(
        [single_guess_entropy(matrix[idx]) for idx in range(len(allowed))]
    )
    best_single_entropy = float(np.max(single_entropies))
    h1 = float(single_entropies[i])
    h2 = float(single_entropies[j])
    h3 = float(single_entropies[k])

    matrix_u32 = matrix.astype(np.uint32)
    row12_scaled = matrix_u32[i] * 59049 + matrix_u32[j] * 243
    h123 = three_guess_entropy(row12_scaled, matrix_u32[k])
    cost = best_single_entropy - h1

    flag_i = "+" if word1 in answer_set else "-"
    flag_j = "+" if word2 in answer_set else "-"
    flag_k = "+" if word3 in answer_set else "-"

    print("\nSpecific three guess triple (non-adaptive, exact):")
    if verbose:
        print(
            "Legend: word1 + word2 + word3 [flags]: H123 bits (H1, H2, H3) | "
            "Cost: (H_best_single - H1) bits"
        )
    else:
        print("Legend: word1 + word2 + word3 [flags]: H123 bits")
    print("flags: [+++] all answers, mixed +/- indicate answer membership by position")
    if verbose:
        print(
            f"{word1} + {word2} + {word3} [{flag_i}{flag_j}{flag_k}]: "
            f"{h123:.4f} bits ({h1:.4f}, {h2:.4f}, {h3:.4f}) | Cost: {cost:.4f} bits"
        )
    else:
        print(f"{word1} + {word2} + {word3} [{flag_i}{flag_j}{flag_k}]: {h123:.4f} bits")


def confirm_three_guess(force):
    if force:
        return True

    print("WARNING: -words 3 is extremely expensive and may run for a very long time.")
    print("Use -force to skip this confirmation in non-interactive runs.")
    try:
        response = input("Type YES to continue: ").strip()
    except EOFError:
        print("Aborted: no interactive input available. Re-run with -force.")
        return False

    if response != "YES":
        print("Aborted.")
        return False

    return True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Wordle entropy analyzer for one-, two-, and three-guess modes."
    )
    parser.add_argument(
        "-words",
        type=int,
        choices=(1, 2, 3),
        default=1,
        help="Number of opening words to optimize (default: 1).",
    )
    parser.add_argument(
        "-verbose",
        action="store_true",
        help="For pair/triple modes, show individual entropies and first-word cost.",
    )
    specific_group = parser.add_mutually_exclusive_group()
    specific_group.add_argument(
        "-pair",
        nargs=2,
        metavar=("WORD1", "WORD2"),
        help="Evaluate one specific pair; overrides -words.",
    )
    specific_group.add_argument(
        "-triple",
        nargs=3,
        metavar=("WORD1", "WORD2", "WORD3"),
        help="Evaluate one specific triple; overrides -words.",
    )
    parser.add_argument(
        "-force",
        action="store_true",
        help="Skip confirmation prompt for expensive -words 3 searches.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    answers, allowed = load_words()
    matrix = load_or_build_matrix(allowed, answers)

    if args.pair is not None:
        word1, word2 = (w.lower() for w in args.pair)
        try:
            run_specific_pair(answers, allowed, matrix, word1, word2, args.verbose)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        return

    if args.triple is not None:
        word1, word2, word3 = (w.lower() for w in args.triple)
        try:
            run_specific_triple(answers, allowed, matrix, word1, word2, word3, args.verbose)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        return

    if args.words == 1:
        run_single_guess(answers, allowed, matrix)
        return

    if args.words == 3:
        if not confirm_three_guess(args.force):
            return
        run_three_guess(answers, allowed, matrix, args.verbose)
        return

    run_two_guess(answers, allowed, matrix, args.verbose)


if __name__ == "__main__":
    main()
