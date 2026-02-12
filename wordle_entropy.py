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
import multiprocessing as mp
import os
import shutil
import sys
import time

import numpy as np
from tqdm import tqdm

from src.entropy import single_guess_entropy, three_guess_entropy, two_guess_entropy
from src.patterns import load_or_build_matrix
from src.words import load_words


TOP_SINGLE = 20
TOP_PAIRS = 50
TOP_TRIPLES = 50
DEFAULT_HISTORY_SECONDS = 30.0


_THREE_WORKER_STATE = {}


def _pct(cur, total):
    if total <= 0:
        return 0.0
    return 100.0 * cur / total


def _fmt_frac(cur, total, width):
    return f"{cur:>{width},}/{total:>{width},}"


def _compact_int(value):
    v = float(value)
    for suffix, scale in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("k", 1e3)):
        if v >= scale:
            return f"{v/scale:.2f}{suffix}"
    return f"{int(v)}"


def _fmt_ddhhmm(seconds):
    total = max(0, int(seconds))
    days = total // 86400
    hours = (total % 86400) // 3600
    minutes = (total % 3600) // 60
    return f"{days:03d}:{hours:02d}:{minutes:02d}"


def _fmt_signed_ddhhmm(seconds):
    sign = "-" if seconds < 0 else "+"
    return f"{sign}{_fmt_ddhhmm(abs(seconds))}"


def _render_dashboard(
    *,
    worker_count,
    chunk_size,
    elapsed,
    history_seconds,
    last_age,
    l1_pct,
    l2_pct,
    completed_i,
    outer_total,
    possible_j_completed,
    possible_j_total,
    prune_j_cur,
    prune_j_all,
    prune_k_cur,
    prune_k_all,
    speed_cur,
    speed_all,
    eta_s,
    ett_s,
    initial_ett_s,
    checked,
    floor,
    max_entropy,
    last_floor_delta,
    last_floor_lift_elapsed,
    recent_samples,
):
    worker_elapsed = elapsed * worker_count
    worker_eta = eta_s * worker_count
    worker_ett = ett_s * worker_count
    lines = [
        (
            "Wordle 3-guess search | "
            f"cores:{worker_count} chunk:{chunk_size} | "
            f"update every {int(history_seconds)}s | last: {int(last_age)}s ago"
        ),
        "",
        "Progress",
        f"L1 outer (i): {l1_pct:5.1f}%   {completed_i:,} / {outer_total:,}",
        f"L2 middle(j): {l2_pct:5.1f}%   {possible_j_completed:,} / {possible_j_total:,}",
        "",
        "Pruning",
        f"P2 skip cur/all: {prune_j_cur:5.1f}% / {prune_j_all:5.1f}%",
        f"P3 skip cur/all: {prune_k_cur:5.1f}% / {prune_k_all:5.1f}%",
        "",
        "Runtime",
        "               Elapsed      ETA          ETT",
        (
            f"Clock        {_fmt_ddhhmm(elapsed)}   {_fmt_ddhhmm(eta_s)}   {_fmt_ddhhmm(ett_s)}"
        ),
        (
            f"Core         {_fmt_ddhhmm(worker_elapsed)}   {_fmt_ddhhmm(worker_eta)}   {_fmt_ddhhmm(worker_ett)}"
        ),
        f"Speed cur/all: {speed_cur/1000:,.1f}k / {speed_all/1000:,.1f}k triple-evals/s",
        "",
        f"ETT delta vs initial: {_fmt_signed_ddhhmm(ett_s - initial_ett_s)}   "
        f"(initial {_fmt_ddhhmm(initial_ett_s)})"
        if initial_ett_s is not None
        else "ETT delta vs initial: N/A",
        f"checked: {_compact_int(checked)} triple-evals",
        "",
        "Floor",
        f"Current floor: {floor:.4f} bits",
        f"Theoretical max: {max_entropy:.4f} bits",
    ]

    if last_floor_lift_elapsed is not None:
        lines.append(
            f"Last floor lift: +{last_floor_delta:.4f} at {_fmt_ddhhmm(last_floor_lift_elapsed)}"
        )
    else:
        lines.append("Last floor lift: N/A (warming)")

    if recent_samples:
        lines.extend(
            [
                "",
                "Recent samples",
                "t(min)   P2cur   P3cur   spd(k/s)   ETA(h)    floor",
            ]
        )
        for s in recent_samples:
            lines.append(
                f"{int(s['elapsed_s']/60):6d}   {s['prune_j_cur_pct']:5.1f}   "
                f"{s['prune_k_cur_pct']:5.1f}   {s['speed_cur']/1000:8.1f}   "
                f"{s['eta_s']/3600:6.1f}   {s['floor']:7.4f}"
            )

    return "\n".join(lines)


def _init_three_worker(
    sorted_indices,
    sorted_entropies,
    matrix_u32,
    max_entropy,
    top_triples,
    shared_floor,
):
    _THREE_WORKER_STATE["sorted_indices"] = sorted_indices
    _THREE_WORKER_STATE["sorted_entropies"] = sorted_entropies
    _THREE_WORKER_STATE["matrix_u32"] = matrix_u32
    _THREE_WORKER_STATE["max_entropy"] = max_entropy
    _THREE_WORKER_STATE["top_triples"] = top_triples
    _THREE_WORKER_STATE["shared_floor"] = shared_floor


def _update_shared_floor(candidate_floor):
    shared_floor = _THREE_WORKER_STATE["shared_floor"]
    with shared_floor.get_lock():
        if candidate_floor > shared_floor.value:
            shared_floor.value = candidate_floor


def _worker_three_chunk(task):
    start_i, end_i = task
    sorted_indices = _THREE_WORKER_STATE["sorted_indices"]
    sorted_entropies = _THREE_WORKER_STATE["sorted_entropies"]
    matrix_u32 = _THREE_WORKER_STATE["matrix_u32"]
    max_entropy = _THREE_WORKER_STATE["max_entropy"]
    top_triples = _THREE_WORKER_STATE["top_triples"]
    shared_floor = _THREE_WORKER_STATE["shared_floor"]
    n_allowed = len(sorted_indices)

    local_best = []
    possible_j_completed = 0
    actual_j_completed = 0
    possible_k_completed = 0
    actual_k_completed = 0
    completed_i = 0
    early_exit_i = 0

    for pos_i in range(start_i, end_i):
        completed_i += 1
        outer_possible_j = max(0, n_allowed - pos_i - 2)
        outer_actual_j = 0
        possible_j_completed += outer_possible_j
        i = sorted_indices[pos_i]
        h1 = sorted_entropies[pos_i]
        row_i = matrix_u32[i]
        row1_scaled = row_i * 59049
        row_i_243 = row_i * 243

        local_floor = local_best[0][0] if len(local_best) == top_triples else -1.0
        cutoff = max(local_floor, shared_floor.value)
        if cutoff >= 0.0 and pos_i + 2 < n_allowed:
            upper_bound = min(
                max_entropy,
                h1 + sorted_entropies[pos_i + 1] + sorted_entropies[pos_i + 2],
            )
            if upper_bound <= cutoff:
                early_exit_i += 1
                continue

        for pos_j in range(pos_i + 1, n_allowed - 1):
            outer_actual_j += 1
            middle_possible_k = max(0, n_allowed - pos_j - 1)
            middle_actual_k = 0
            possible_k_completed += middle_possible_k
            j = sorted_indices[pos_j]
            h2 = sorted_entropies[pos_j]

            local_floor = local_best[0][0] if len(local_best) == top_triples else -1.0
            cutoff = max(local_floor, shared_floor.value)
            if cutoff >= 0.0 and pos_j + 1 < n_allowed:
                upper_bound = min(max_entropy, h1 + h2 + sorted_entropies[pos_j + 1])
                if upper_bound <= cutoff:
                    break

            h12 = two_guess_entropy(row_i_243, matrix_u32[j])
            local_floor = local_best[0][0] if len(local_best) == top_triples else -1.0
            cutoff = max(local_floor, shared_floor.value)
            if cutoff >= 0.0 and pos_j + 1 < n_allowed:
                upper_bound = min(max_entropy, h12 + sorted_entropies[pos_j + 1])
                if upper_bound <= cutoff:
                    break

            row12_scaled = row1_scaled + matrix_u32[j] * 243
            for pos_k in range(pos_j + 1, n_allowed):
                h3 = sorted_entropies[pos_k]
                local_floor = local_best[0][0] if len(local_best) == top_triples else -1.0
                cutoff = max(local_floor, shared_floor.value)
                if cutoff >= 0.0 and min(max_entropy, h12 + h3) <= cutoff:
                    break

                k = sorted_indices[pos_k]
                h123 = three_guess_entropy(row12_scaled, matrix_u32[k])
                middle_actual_k += 1

                if len(local_best) < top_triples:
                    heapq.heappush(local_best, (h123, i, j, k))
                    if len(local_best) == top_triples:
                        _update_shared_floor(local_best[0][0])
                elif h123 > local_best[0][0]:
                    heapq.heapreplace(local_best, (h123, i, j, k))
                    _update_shared_floor(local_best[0][0])

            actual_k_completed += middle_actual_k

        actual_j_completed += outer_actual_j

    return {
        "completed_i": completed_i,
        "possible_j_completed": possible_j_completed,
        "actual_j_completed": actual_j_completed,
        "possible_k_completed": possible_k_completed,
        "actual_k_completed": actual_k_completed,
        "local_best": local_best,
        "early_exit_i": early_exit_i,
    }


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


def run_three_guess(
    answers,
    allowed,
    matrix,
    verbose,
    workers=None,
    chunk_size=1,
    progress_mode="dashboard",
    stats_file=None,
    history_seconds=DEFAULT_HISTORY_SECONDS,
):
    answer_set = set(answers)
    n_allowed = len(allowed)

    print("Computing single guess entropies...")
    single_entropies = np.array(
        [single_guess_entropy(matrix[i]) for i in range(n_allowed)]
    )
    best_single_entropy = float(np.max(single_entropies))
    matrix_u32 = matrix.astype(np.uint32)
    sorted_indices = np.argsort(single_entropies)[::-1]
    sorted_entropies = single_entropies[sorted_indices]
    max_entropy = float(np.log2(len(answers)))

    best_triples = []
    start_time = time.time()
    early_exit_count = 0
    outer_total = n_allowed - 2
    worker_count = workers if workers is not None else (os.cpu_count() or 1)
    worker_count = max(1, int(worker_count))
    chunk_size = max(1, int(chunk_size))

    progress = {
        "completed_i": 0,
        "possible_j_completed": 0,
        "actual_j_completed": 0,
        "possible_k_completed": 0,
        "actual_k_completed": 0,
        "floor": None,
    }
    snapshots = []
    next_snapshot_time = history_seconds
    possible_j_total = (n_allowed - 1) * (n_allowed - 2) // 2
    possible_k_total = n_allowed * (n_allowed - 1) * (n_allowed - 2) // 6
    last_sample_time = start_time
    last_sample_possible_j = 0
    last_sample_actual_j = 0
    last_sample_possible_k = 0
    last_sample_actual_k = 0
    initial_ett_s = None

    print(
        f"Starting optimized full three guess search using {worker_count} worker(s), "
        f"chunk size {chunk_size}...\n"
    )

    start_methods = mp.get_all_start_methods()
    start_method = "fork" if "fork" in start_methods else "spawn"
    ctx = mp.get_context(start_method)
    shared_floor = ctx.Value("d", -1.0)
    tasks = [
        (start_i, min(start_i + chunk_size, outer_total))
        for start_i in range(0, outer_total, chunk_size)
    ]

    with ctx.Pool(
        processes=worker_count,
        initializer=_init_three_worker,
        initargs=(
            sorted_indices,
            sorted_entropies,
            matrix_u32,
            max_entropy,
            TOP_TRIPLES,
            shared_floor,
        ),
    ) as pool:
        for result in pool.imap_unordered(_worker_three_chunk, tasks, chunksize=1):
            progress["completed_i"] += result["completed_i"]
            progress["possible_j_completed"] += result["possible_j_completed"]
            progress["actual_j_completed"] += result["actual_j_completed"]
            progress["possible_k_completed"] += result["possible_k_completed"]
            progress["actual_k_completed"] += result["actual_k_completed"]
            early_exit_count += result["early_exit_i"]

            for triple in result["local_best"]:
                if len(best_triples) < TOP_TRIPLES:
                    heapq.heappush(best_triples, triple)
                elif triple[0] > best_triples[0][0]:
                    heapq.heapreplace(best_triples, triple)

            if len(best_triples) == TOP_TRIPLES:
                progress["floor"] = best_triples[0][0]
                with shared_floor.get_lock():
                    if best_triples[0][0] > shared_floor.value:
                        shared_floor.value = best_triples[0][0]

            now = time.time()
            elapsed = now - start_time
            if elapsed >= next_snapshot_time:
                sample_elapsed = now - last_sample_time
                delta_possible_j = progress["possible_j_completed"] - last_sample_possible_j
                delta_actual_j = progress["actual_j_completed"] - last_sample_actual_j
                delta_possible_k = progress["possible_k_completed"] - last_sample_possible_k
                delta_actual_k = progress["actual_k_completed"] - last_sample_actual_k

                prune_j_all = 0.0
                if progress["possible_j_completed"] > 0:
                    prune_j_all = 100.0 * (
                        1.0
                        - progress["actual_j_completed"] / progress["possible_j_completed"]
                    )
                prune_k_all = 0.0
                if progress["possible_k_completed"] > 0:
                    prune_k_all = 100.0 * (
                        1.0
                        - progress["actual_k_completed"] / progress["possible_k_completed"]
                    )

                prune_j_cur = prune_j_all
                if delta_possible_j > 0:
                    prune_j_cur = 100.0 * (1.0 - delta_actual_j / delta_possible_j)
                prune_k_cur = prune_k_all
                if delta_possible_k > 0:
                    prune_k_cur = 100.0 * (1.0 - delta_actual_k / delta_possible_k)

                speed_all = (
                    progress["actual_k_completed"] / elapsed if elapsed > 0 else 0.0
                )
                speed_cur = (
                    delta_actual_k / sample_elapsed if sample_elapsed > 0 else speed_all
                )
                remaining_possible_k = max(0, possible_k_total - progress["possible_k_completed"])
                remaining_actual_k = remaining_possible_k * max(0.0, 1.0 - prune_k_cur / 100.0)
                eta_s = remaining_actual_k / speed_cur if speed_cur > 0 else 0.0
                ett_s = elapsed + eta_s
                if initial_ett_s is None:
                    initial_ett_s = ett_s
                l1_pct = _pct(progress["completed_i"], outer_total)
                l2_pct = _pct(progress["possible_j_completed"], possible_j_total)
                l3_pct = _pct(progress["possible_k_completed"], possible_k_total)

                snapshots.append(
                    {
                        "elapsed_s": elapsed,
                        "completed_i": progress["completed_i"],
                        "possible_j": progress["possible_j_completed"],
                        "actual_j": progress["actual_j_completed"],
                        "possible_k": progress["possible_k_completed"],
                        "actual_k": progress["actual_k_completed"],
                        "prune_j_pct": prune_j_all,
                        "prune_k_pct": prune_k_all,
                        "prune_j_cur_pct": prune_j_cur,
                        "prune_k_cur_pct": prune_k_cur,
                        "speed": speed_all,
                        "speed_cur": speed_cur,
                        "eta_s": eta_s,
                        "ett_s": ett_s,
                        "l1_pct": l1_pct,
                        "l2_pct": l2_pct,
                        "l3_pct": l3_pct,
                        "floor": progress["floor"] if progress["floor"] is not None else 0.0,
                    }
                )

                last_floor_delta = 0.0
                last_floor_lift_elapsed = None
                prev_floor = None
                for s in snapshots:
                    cur_floor = s["floor"]
                    if prev_floor is not None and cur_floor > prev_floor:
                        last_floor_delta = cur_floor - prev_floor
                        last_floor_lift_elapsed = s["elapsed_s"]
                    prev_floor = cur_floor

                recent_samples = snapshots[-6:]
                floor_display = progress["floor"] if progress["floor"] is not None else 0.0

                if progress_mode in ("dashboard", "live"):
                    size = shutil.get_terminal_size((80, 24))
                    if size.columns < 80 or size.lines < 16:
                        output = (
                            "Expand terminal for diagnostics "
                            f"(need >=80x16, current {size.columns}x{size.lines})"
                        )
                    else:
                        output = _render_dashboard(
                            worker_count=worker_count,
                            chunk_size=chunk_size,
                            elapsed=elapsed,
                            history_seconds=history_seconds,
                            last_age=0.0,
                            l1_pct=l1_pct,
                            l2_pct=l2_pct,
                            completed_i=progress["completed_i"],
                            outer_total=outer_total,
                            possible_j_completed=progress["possible_j_completed"],
                            possible_j_total=possible_j_total,
                            prune_j_cur=prune_j_cur,
                            prune_j_all=prune_j_all,
                            prune_k_cur=prune_k_cur,
                            prune_k_all=prune_k_all,
                            speed_cur=speed_cur,
                            speed_all=speed_all,
                            eta_s=eta_s,
                            ett_s=ett_s,
                            initial_ett_s=initial_ett_s,
                            checked=progress["actual_k_completed"],
                            floor=floor_display,
                            max_entropy=max_entropy,
                            last_floor_delta=last_floor_delta,
                            last_floor_lift_elapsed=last_floor_lift_elapsed,
                            recent_samples=recent_samples,
                        )
                    sys.stdout.write("\033[2J\033[H")
                    sys.stdout.write(output + "\n")
                    sys.stdout.flush()
                elif progress_mode == "log":
                    print(
                        f"[{elapsed/3600:6.2f}h] "
                        f"L1 {l1_pct:5.1f}% ({progress['completed_i']:,}/{outer_total:,}) | "
                        f"L2 {l2_pct:5.1f}% ({progress['possible_j_completed']:,}/{possible_j_total:,})"
                    )
                    print(
                        f"           P2 cur/all {prune_j_cur:5.1f}/{prune_j_all:5.1f}% | "
                        f"P3 cur/all {prune_k_cur:5.1f}/{prune_k_all:5.1f}% | "
                        f"spd {speed_cur:,.0f}/{speed_all:,.0f}/s | eta {_fmt_ddhhmm(eta_s)} | "
                        f"ett {_fmt_ddhhmm(ett_s)} | "
                        f"floor {floor_display:.4f}"
                    )

                next_snapshot_time += history_seconds
                last_sample_time = now
                last_sample_possible_j = progress["possible_j_completed"]
                last_sample_actual_j = progress["actual_j_completed"]
                last_sample_possible_k = progress["possible_k_completed"]
                last_sample_actual_k = progress["actual_k_completed"]

    best_triples.sort(reverse=True)

    if progress["completed_i"] < outer_total and progress["floor"] is not None:
        print(
            "Outer-level early exits occurred within chunk workers once branch upper bounds "
            f"fell under floor (count: {early_exit_count:,})."
        )

    if snapshots:
        print("\nSearch history summary (sampled):")
        print(
            f"P2 all: {snapshots[0]['prune_j_pct']:.1f}% -> {snapshots[-1]['prune_j_pct']:.1f}% | "
            f"P3 all: {snapshots[0]['prune_k_pct']:.1f}% -> {snapshots[-1]['prune_k_pct']:.1f}%"
        )
        print(
            f"Speed all: {snapshots[0]['speed']:,.0f} -> {snapshots[-1]['speed']:,.0f} triple-evals/s"
        )
        print(
            f"Floor: {snapshots[0]['floor']:.4f} -> {snapshots[-1]['floor']:.4f} bits"
        )

    if stats_file is not None and snapshots:
        with open(stats_file, "w", encoding="ascii") as handle:
            handle.write(
                "elapsed_s,completed_i,possible_j,actual_j,possible_k,actual_k,"
                "prune_j_pct,prune_k_pct,prune_j_cur_pct,prune_k_cur_pct,"
                "speed,speed_cur,eta_s,l1_pct,l2_pct,l3_pct,floor\n"
            )
            for snap in snapshots:
                handle.write(
                    f"{snap['elapsed_s']:.3f},{snap['completed_i']},{snap['possible_j']},"
                    f"{snap['actual_j']},{snap['possible_k']},{snap['actual_k']},"
                    f"{snap['prune_j_pct']:.5f},{snap['prune_k_pct']:.5f},"
                    f"{snap['prune_j_cur_pct']:.5f},{snap['prune_k_cur_pct']:.5f},"
                    f"{snap['speed']:.5f},{snap['speed_cur']:.5f},{snap['eta_s']:.3f},"
                    f"{snap['l1_pct']:.5f},{snap['l2_pct']:.5f},{snap['l3_pct']:.5f},"
                    f"{snap['floor']:.6f}\n"
                )
        print(f"Saved sampled run statistics to {stats_file}.")

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
    parser.add_argument(
        "-workers",
        type=int,
        default=None,
        help="Worker processes for -words 3 (default: CPU count).",
    )
    parser.add_argument(
        "-chunk-size",
        type=int,
        default=1,
        help="Number of first-guess indices per worker task in -words 3 mode.",
    )
    parser.add_argument(
        "-progress",
        choices=("dashboard", "live", "log", "off"),
        default="dashboard",
        help="Progress output style for -words 3 (default: dashboard).",
    )
    parser.add_argument(
        "-stats-file",
        type=str,
        default=None,
        help="Optional CSV path for sampled -words 3 run statistics.",
    )
    parser.add_argument(
        "-history-seconds",
        type=float,
        default=DEFAULT_HISTORY_SECONDS,
        help="Sampling period in seconds for run-history summaries in -words 3 mode.",
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
        run_three_guess(
            answers,
            allowed,
            matrix,
            args.verbose,
            workers=args.workers,
            chunk_size=args.chunk_size,
            progress_mode=args.progress,
            stats_file=args.stats_file,
            history_seconds=args.history_seconds,
        )
        return

    run_two_guess(answers, allowed, matrix, args.verbose)


if __name__ == "__main__":
    main()
