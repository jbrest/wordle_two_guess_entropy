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
from collections import deque
import hashlib
import heapq
import json
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
DASHBOARD_REFRESH_SECONDS = 1.0
INITIAL_ETT_MIN_ELAPSED_S = 30.0
INITIAL_ETT_MIN_CHECKED = 10_000_000


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


def _fmt_pct_or_na(value):
    if value is None:
        return " N/A "
    return f"{value:5.1f}%"


def _render_dashboard(
    *,
    worker_count,
    chunk_size,
    elapsed,
    history_seconds,
    allowed_count,
    answer_count,
    l1_pct,
    l2_pct,
    completed_i,
    outer_total,
    current_mid_done,
    current_mid_total,
    possible_j_completed,
    possible_j_total,
    prune_j_last,
    prune_j_all,
    prune_k_last,
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
    initial_core_ett_s = None if initial_ett_s is None else (initial_ett_s * worker_count)
    mid_cur_pct = _pct(current_mid_done, current_mid_total)
    progress_label_width = 24
    pct_width = 5
    sep = "   "
    outer_done_w = len(f"{outer_total:,}")
    mid_cur_done_w = len(f"{max(1, current_mid_total):,}")
    mid_tot_done_w = len(f"{possible_j_total:,}")
    done_w = max(outer_done_w, mid_cur_done_w, mid_tot_done_w)
    total_w = max(
        len(f"{outer_total:,}"),
        len(f"{max(1, current_mid_total):,}"),
        len(f"{possible_j_total:,}"),
    )
    lines = [
        (
            "Wordle 3-guess entropy | "
            f"{allowed_count:,} allowed, {answer_count:,} answers | "
            f"cores:{worker_count} chunk:{chunk_size} | "
            f"update every {int(history_seconds)}s"
        ),
        "",
        "Progress",
        (
            f"{'Outer loop:':<{progress_label_width}} "
            f"{l1_pct:{pct_width}.1f}%{sep}"
            f"{completed_i:>{done_w},} / {outer_total:>{total_w},}"
        ),
        (
            f"{'Middle loop (current):':<{progress_label_width}} "
            f"{mid_cur_pct:{pct_width}.1f}%{sep}"
            f"{current_mid_done:>{done_w},} / {max(1, current_mid_total):>{total_w},}"
        ),
        (
            f"{'Middle loop (total):':<{progress_label_width}} "
            f"{l2_pct:{pct_width}.1f}%{sep}"
            f"{possible_j_completed:>{done_w},} / {possible_j_total:>{total_w},}"
        ),
        "",
        "Pruning",
        (
            "Middle loop (last / cumulative): "
            f"{_fmt_pct_or_na(prune_j_last)} / {prune_j_all:5.1f}% skipped"
        ),
        (
            "Inner loop (last / cumulative):  "
            f"{_fmt_pct_or_na(prune_k_last)} / {prune_k_all:5.1f}% skipped"
        ),
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
        f"ETT delta vs initial(warm): {_fmt_signed_ddhhmm(ett_s - initial_ett_s)}   "
        f"(initial {_fmt_ddhhmm(initial_ett_s)})"
        if initial_ett_s is not None
        else "ETT delta vs initial(warm): N/A",
        (
            f"Core ETT delta vs initial(warm): "
            f"{_fmt_signed_ddhhmm(worker_ett - initial_core_ett_s)}   "
            f"(initial {_fmt_ddhhmm(initial_core_ett_s)})"
        )
        if initial_core_ett_s is not None
        else "Core ETT delta vs initial(warm): N/A",
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
                "t(min)   P2all   P3all   spd(k/s)   ETA(h)   ETT(h)    floor",
            ]
        )
        for s in recent_samples:
            lines.append(
                f"{int(s['elapsed_s']/60):6d}   {s['prune_j_pct']:5.1f}   "
                f"{s['prune_k_pct']:5.1f}   {s['speed_cur']/1000:8.1f}   "
                f"{s['eta_s']/3600:6.1f}   {s['ett_s']/3600:6.1f}   {s['floor']:7.4f}"
            )

    return "\n".join(lines)


def _pid_log_path(path, pid):
    if "{pid}" in path:
        return path.replace("{pid}", str(pid))
    root, ext = os.path.splitext(path)
    if ext:
        return f"{root}.{pid}{ext}"
    return f"{path}.{pid}"


def _dataset_signature(answers, allowed):
    h = hashlib.sha256()
    h.update(f"{len(answers)}|{len(allowed)}|".encode("ascii"))
    for word in answers:
        h.update(word.encode("ascii"))
        h.update(b"\n")
    h.update(b"|")
    for word in allowed:
        h.update(word.encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()


def _load_checkpoint(path):
    with open(path, "r", encoding="ascii") as handle:
        return json.load(handle)


def _write_checkpoint(path, payload):
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="ascii") as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


def _confirm_resume_checkpoint(path):
    try:
        response = input(f"Checkpoint found at {path}. Resume? [Y/n]: ").strip().lower()
    except EOFError:
        return True
    if response in ("", "y", "yes"):
        return True
    if response in ("n", "no"):
        return False
    return True


def _update_shared_floor(candidate_floor):
    shared_floor = _THREE_WORKER_STATE["shared_floor"]
    with shared_floor.get_lock():
        if candidate_floor > shared_floor.value:
            shared_floor.value = candidate_floor


def _init_k_worker(
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


def _worker_k_branch(task):
    i, pos_j, row12_scaled, h12 = task
    sorted_indices = _THREE_WORKER_STATE["sorted_indices"]
    sorted_entropies = _THREE_WORKER_STATE["sorted_entropies"]
    matrix_u32 = _THREE_WORKER_STATE["matrix_u32"]
    max_entropy = _THREE_WORKER_STATE["max_entropy"]
    top_triples = _THREE_WORKER_STATE["top_triples"]
    shared_floor = _THREE_WORKER_STATE["shared_floor"]
    n_allowed = len(sorted_indices)

    middle_possible_k = max(0, n_allowed - pos_j - 1)
    middle_actual_k = 0
    local_best = []

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
            heapq.heappush(local_best, (h123, i, sorted_indices[pos_j], k))
            if len(local_best) == top_triples:
                _update_shared_floor(local_best[0][0])
        elif h123 > local_best[0][0]:
            heapq.heapreplace(local_best, (h123, i, sorted_indices[pos_j], k))
            _update_shared_floor(local_best[0][0])

    return {
        "pos_j": pos_j,
        "possible_k": middle_possible_k,
        "actual_k": middle_actual_k,
        "local_best": local_best,
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
    chunk_size=2,
    progress_mode="dashboard",
    stats_file=None,
    debug_prune_file=None,
    checkpoint_file=".wordle3_checkpoint.json",
    resume_mode="ask",
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
    debug_prune = os.getenv("WORDLE_DEBUG_PRUNE", "0") == "1"
    debug_prune_log = os.getenv("WORDLE_DEBUG_PRUNE_LOG", "/tmp/wordle_p2_debug.log")
    debug_stop_on_first_p2_i0 = os.getenv("WORDLE_DEBUG_STOP_ON_FIRST_P2_I0", "0") == "1"
    debug_prune_handle = None
    run_first_p2_i = None
    run_first_p2_j = None
    run_first_p2_elapsed_s = None
    dataset_sig = _dataset_signature(answers, allowed)
    resume_pos_i = 0
    last_completed_i_p2_pct = None
    last_completed_j_p3_pct = None

    progress = {
        "completed_i": 0,
        "possible_j_completed": 0,
        "actual_j_completed": 0,
        "possible_k_completed": 0,
        "skipped_k_completed": 0,
        "actual_k_completed": 0,
        "floor": None,
    }
    current_mid_done = 0
    current_mid_total = 1
    snapshots = []
    possible_j_total = (n_allowed - 1) * (n_allowed - 2) // 2
    possible_k_total = n_allowed * (n_allowed - 1) * (n_allowed - 2) // 6
    last_cur_time = start_time
    last_cur_checked = 0
    initial_ett_s = None
    cur_window = deque()
    current_i_classified_j = 0
    current_i_skipped_j = 0
    current_i_possible_k = 0
    current_i_skipped_k = 0
    prev_i_index = None
    prev_i_classified_j = 0
    prev_i_skipped_j = 0
    prev_i_possible_k = 0
    prev_i_skipped_k = 0
    current_j_index = None
    current_j_possible_k = 0
    current_j_skipped_k = 0
    prev_j_index = None
    prev_j_possible_k = 0
    prev_j_skipped_k = 0
    p2_tail_snap_i = None
    p2_tail_snap_progress_pct = 0.0
    p2_tail_snap_skipped = 0
    p2_tail_snap_total = 0
    p2_tail_snap_pct = 0.0
    p2_head_snap_i = None
    p2_head_snap_progress_pct = 0.0
    p2_head_snap_skipped = 0
    p2_head_snap_total = 0
    p2_head_snap_pct = 0.0

    if checkpoint_file is not None and os.path.exists(checkpoint_file):
        do_resume = True
        if resume_mode == "new":
            do_resume = False
            print(f"Ignoring existing checkpoint due to -resume new: {checkpoint_file}")
        elif resume_mode == "ask":
            if sys.stdin.isatty():
                do_resume = _confirm_resume_checkpoint(checkpoint_file)
            else:
                do_resume = True
        if do_resume:
            ckpt = _load_checkpoint(checkpoint_file)
            ckpt_sig = ckpt.get("dataset_sig")
            if ckpt_sig is not None and ckpt_sig != dataset_sig:
                raise SystemExit(
                    "Checkpoint dataset signature does not match current word lists; "
                    "refusing to resume."
                )
            resume_pos_i = int(ckpt.get("next_pos_i", 0))
            if resume_pos_i < 0 or resume_pos_i > outer_total:
                raise SystemExit(
                    f"Checkpoint next_pos_i out of range: {resume_pos_i} (outer_total={outer_total})"
                )
            loaded_progress = ckpt.get("progress", {})
            for key in progress:
                if key in loaded_progress:
                    progress[key] = loaded_progress[key]
            loaded_initial_ett_s = ckpt.get("initial_ett_s")
            if loaded_initial_ett_s is not None:
                initial_ett_s = float(loaded_initial_ett_s)
            loaded_last_i_p2 = ckpt.get("last_completed_i_p2_pct")
            if loaded_last_i_p2 is not None:
                last_completed_i_p2_pct = float(loaded_last_i_p2)
            loaded_last_j_p3 = ckpt.get("last_completed_j_p3_pct")
            if loaded_last_j_p3 is not None:
                last_completed_j_p3_pct = float(loaded_last_j_p3)
            loaded_triples = ckpt.get("best_triples", [])
            best_triples = []
            for triple in loaded_triples:
                if not isinstance(triple, list) or len(triple) != 4:
                    continue
                h123, i, j, k = triple
                best_triples.append((float(h123), int(i), int(j), int(k)))
            heapq.heapify(best_triples)
            if len(best_triples) > TOP_TRIPLES:
                best_triples = heapq.nlargest(TOP_TRIPLES, best_triples)
                heapq.heapify(best_triples)
            if best_triples and progress["floor"] is None:
                progress["floor"] = float(best_triples[0][0])
            early_exit_count = int(ckpt.get("early_exit_count", 0))
            print(
            f"Resuming from checkpoint {checkpoint_file}: "
            f"next i={resume_pos_i}/{outer_total}, completed_i={progress['completed_i']}"
        )
        else:
            print(f"Starting fresh and overwriting checkpoint: {checkpoint_file}")
    elif checkpoint_file is not None:
        print(f"Checkpointing enabled: writing to {checkpoint_file}")

    print(
        f"Starting optimized full three guess search using {worker_count} worker(s), "
        f"chunk size {chunk_size}...\n"
    )

    def build_checkpoint_payload(next_pos_i):
        return {
            "schema": 1,
            "dataset_sig": dataset_sig,
            "outer_total": outer_total,
            "next_pos_i": int(next_pos_i),
            "saved_at_epoch_s": time.time(),
            "initial_ett_s": (
                None
                if initial_ett_s is None
                else float(initial_ett_s)
            ),
            "last_completed_i_p2_pct": (
                None
                if last_completed_i_p2_pct is None
                else float(last_completed_i_p2_pct)
            ),
            "last_completed_j_p3_pct": (
                None
                if last_completed_j_p3_pct is None
                else float(last_completed_j_p3_pct)
            ),
            "progress": {
                "completed_i": int(progress["completed_i"]),
                "possible_j_completed": int(progress["possible_j_completed"]),
                "actual_j_completed": int(progress["actual_j_completed"]),
                "possible_k_completed": int(progress["possible_k_completed"]),
                "skipped_k_completed": int(progress["skipped_k_completed"]),
                "actual_k_completed": int(progress["actual_k_completed"]),
                "floor": (
                    None
                    if progress["floor"] is None
                    else float(progress["floor"])
                ),
            },
            "best_triples": [
                [float(h123), int(i), int(j), int(k)]
                for (h123, i, j, k) in best_triples
            ],
            "early_exit_count": int(early_exit_count),
        }

    if checkpoint_file is not None:
        _write_checkpoint(checkpoint_file, build_checkpoint_payload(resume_pos_i))

    start_methods = mp.get_all_start_methods()
    start_method = "fork" if "fork" in start_methods else "spawn"
    ctx = mp.get_context(start_method)
    shared_floor = ctx.Value("d", -1.0)

    with ctx.Pool(
        processes=worker_count,
        initializer=_init_k_worker,
        initargs=(
            sorted_indices,
            sorted_entropies,
            matrix_u32,
            max_entropy,
            TOP_TRIPLES,
            shared_floor,
        ),
    ) as pool:
        if progress["floor"] is not None:
            with shared_floor.get_lock():
                shared_floor.value = float(progress["floor"])
        if debug_prune_file is not None:
            debug_prune_file = _pid_log_path(debug_prune_file, os.getpid())
            print(f"Writing prune diagnostics to {debug_prune_file}")
            debug_prune_handle = open(debug_prune_file, "w", encoding="ascii", buffering=1)
            debug_prune_handle.write(
                f"# pid={os.getpid()} start_epoch_s={start_time:.3f}\n"
            )
            debug_prune_handle.write(
                "event,elapsed_s,i,j,pct_bucket,i_loop_len,"
                "i_classified_j,i_skipped_j,first_prune_j,first_prune_offset,"
                "tail_len,total_prunes,tail_prune_ratio,"
                "all_possible_j,all_skipped_j,all_possible_k,all_skipped_k\n"
            )
        next_snapshot_time = history_seconds
        next_render_time = 0.0

        def emit_status(now):
            nonlocal next_snapshot_time
            nonlocal next_render_time
            nonlocal last_cur_time
            nonlocal last_cur_checked
            nonlocal initial_ett_s
            nonlocal last_completed_i_p2_pct
            nonlocal last_completed_j_p3_pct
            nonlocal current_i_classified_j
            nonlocal current_i_skipped_j
            nonlocal current_i_possible_k
            nonlocal current_i_skipped_k
            nonlocal prev_i_index
            nonlocal prev_i_classified_j
            nonlocal prev_i_skipped_j
            nonlocal prev_i_possible_k
            nonlocal prev_i_skipped_k
            nonlocal current_j_index
            nonlocal current_j_possible_k
            nonlocal current_j_skipped_k
            nonlocal prev_j_index
            nonlocal prev_j_possible_k
            nonlocal prev_j_skipped_k
            nonlocal p2_tail_snap_i
            nonlocal p2_tail_snap_progress_pct
            nonlocal p2_tail_snap_skipped
            nonlocal p2_tail_snap_total
            nonlocal p2_tail_snap_pct
            nonlocal p2_head_snap_i
            nonlocal p2_head_snap_progress_pct
            nonlocal p2_head_snap_skipped
            nonlocal p2_head_snap_total
            nonlocal p2_head_snap_pct
            elapsed = now - start_time
            prune_total_j = progress["possible_j_completed"]
            actual_total_j = progress["actual_j_completed"]
            prune_skipped_j = max(0, prune_total_j - actual_total_j)
            prune_total_k = progress["possible_k_completed"]
            prune_skipped_k = progress["skipped_k_completed"]
            checked = progress["actual_k_completed"]

            cur_elapsed = max(1e-9, now - last_cur_time)
            delta_checked = checked - last_cur_checked

            prune_j_all = 0.0
            if prune_total_j > 0:
                prune_j_all = 100.0 * (prune_skipped_j / prune_total_j)
            prune_k_all = 0.0
            if prune_total_k > 0:
                prune_k_all = 100.0 * (prune_skipped_k / prune_total_k)

            cur_window.append(
                (
                    now,
                    checked,
                )
            )
            while len(cur_window) >= 2 and now - cur_window[0][0] > history_seconds:
                cur_window.popleft()

            base_t, base_checked = cur_window[0]
            win_elapsed = max(1e-9, now - base_t)
            win_delta_checked = checked - base_checked

            prune_j_cur = 0.0
            if current_i_classified_j > 0:
                prune_j_cur = 100.0 * (current_i_skipped_j / current_i_classified_j)

            prune_k_cur = prune_k_all
            if current_j_index is not None and current_j_possible_k > 0:
                prune_k_cur = 100.0 * (current_j_skipped_k / current_j_possible_k)

            p2_i_prev_pct = 0.0
            prune_j_last = last_completed_i_p2_pct
            if prev_i_classified_j > 0:
                p2_i_prev_pct = 100.0 * (prev_i_skipped_j / prev_i_classified_j)
                prune_j_last = p2_i_prev_pct
            p3_j_cur_pct = 0.0
            if current_j_index is not None and current_j_possible_k > 0:
                p3_j_cur_pct = 100.0 * (current_j_skipped_k / current_j_possible_k)
            prune_k_last = last_completed_j_p3_pct
            if prev_j_index is not None and prev_j_possible_k > 0:
                p3_j_prev_pct = 100.0 * (prev_j_skipped_k / prev_j_possible_k)
                prune_k_last = p3_j_prev_pct

            speed_all = checked / elapsed if elapsed > 0 else 0.0
            speed_cur = win_delta_checked / win_elapsed if win_delta_checked > 0 else (
                delta_checked / cur_elapsed
            )
            remaining_possible_k = max(0, possible_k_total - prune_total_k)
            speed_for_eta = speed_cur if speed_cur > 0 else speed_all
            remaining_actual_k = remaining_possible_k * max(0.0, 1.0 - prune_k_all / 100.0)
            eta_s = remaining_actual_k / speed_for_eta if speed_for_eta > 0 else 0.0
            ett_s = elapsed + eta_s
            if (
                initial_ett_s is None
                and speed_for_eta > 0
                and elapsed >= INITIAL_ETT_MIN_ELAPSED_S
                and checked >= INITIAL_ETT_MIN_CHECKED
            ):
                initial_ett_s = ett_s
            l1_pct = _pct(progress["completed_i"], outer_total)
            l2_pct = _pct(prune_total_j, possible_j_total)
            l3_pct = _pct(prune_total_k, possible_k_total)

            did_snapshot = False
            if elapsed >= next_snapshot_time:
                snapshots.append(
                    {
                        "elapsed_s": elapsed,
                        "completed_i": progress["completed_i"],
                        "possible_j": prune_total_j,
                        "actual_j": max(0, prune_total_j - prune_skipped_j),
                        "possible_k": prune_total_k,
                        "actual_k": checked,
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
                next_snapshot_time += history_seconds
                did_snapshot = True

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

            if now >= next_render_time:
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
                            allowed_count=n_allowed,
                            answer_count=len(answers),
                            l1_pct=l1_pct,
                            l2_pct=l2_pct,
                            completed_i=progress["completed_i"],
                            outer_total=outer_total,
                            current_mid_done=current_mid_done,
                            current_mid_total=current_mid_total,
                            possible_j_completed=prune_total_j,
                            possible_j_total=possible_j_total,
                            prune_j_last=prune_j_last,
                            prune_j_all=prune_j_all,
                            prune_k_last=prune_k_last,
                            prune_k_all=prune_k_all,
                            speed_cur=speed_cur,
                            speed_all=speed_all,
                            eta_s=eta_s,
                            ett_s=ett_s,
                            initial_ett_s=initial_ett_s,
                            checked=checked,
                            floor=floor_display,
                            max_entropy=max_entropy,
                            last_floor_delta=last_floor_delta,
                            last_floor_lift_elapsed=last_floor_lift_elapsed,
                            recent_samples=recent_samples,
                        )
                    sys.stdout.write("\033[2J\033[H")
                    sys.stdout.write(output + "\n")
                    sys.stdout.flush()
                elif progress_mode == "log" and did_snapshot:
                    p2_last_str = _fmt_pct_or_na(prune_j_last).strip()
                    p3_last_str = _fmt_pct_or_na(prune_k_last).strip()
                    print(
                        f"[{elapsed/3600:6.2f}h] "
                        f"L1 {l1_pct:5.1f}% ({progress['completed_i']:,}/{outer_total:,}) | "
                        f"L2 {l2_pct:5.1f}% ({prune_total_j:,}/{possible_j_total:,})"
                    )
                    print(
                        f"           P2 last/all {p2_last_str}/{prune_j_all:5.1f}% | "
                        f"P3 last/all {p3_last_str}/{prune_k_all:5.1f}% | "
                        f"spd {speed_cur:,.0f}/{speed_all:,.0f}/s | eta {_fmt_ddhhmm(eta_s)} | "
                        f"ett {_fmt_ddhhmm(ett_s)} | "
                        f"floor {floor_display:.4f}"
                    )
                last_cur_time = now
                last_cur_checked = checked
                next_render_time = now + DASHBOARD_REFRESH_SECONDS

        for pos_i in range(resume_pos_i, outer_total):
            i = sorted_indices[pos_i]
            h1 = sorted_entropies[pos_i]
            outer_possible_j = max(0, n_allowed - pos_i - 2)
            current_mid_total = max(1, outer_possible_j)
            current_mid_done = 0
            current_i_classified_j = 0
            current_i_skipped_j = 0
            current_i_possible_k = 0
            current_i_skipped_k = 0
            current_j_index = None
            current_j_possible_k = 0
            current_j_skipped_k = 0
            prev_j_index = None
            prev_j_possible_k = 0
            prev_j_skipped_k = 0
            p2_tail_captured_this_i = False
            p2_head_captured_this_i = False
            next_pct_bucket = 1
            row_i = matrix_u32[i]
            row_i_243 = row_i * 243

            def maybe_log_pct_crossings(now):
                nonlocal next_pct_bucket
                if debug_prune_handle is None or current_mid_total <= 0:
                    return
                pct_floor = int((current_i_classified_j * 100) // current_mid_total)
                if pct_floor < next_pct_bucket:
                    return
                all_possible_j = progress["possible_j_completed"]
                all_skipped_j = all_possible_j - progress["actual_j_completed"]
                all_possible_k = progress["possible_k_completed"]
                all_skipped_k = progress["skipped_k_completed"]
                while next_pct_bucket <= pct_floor and next_pct_bucket <= 100:
                    debug_prune_handle.write(
                        f"pct,{now-start_time:.3f},{pos_i},,{next_pct_bucket},"
                        f"{current_mid_total},{current_i_classified_j},{current_i_skipped_j},"
                        ",,,,,"  # first-prune and tail-cutoff fields are loop-end only
                        f"{all_possible_j},{all_skipped_j},"
                        f"{all_possible_k},{all_skipped_k}\n"
                    )
                    next_pct_bucket += 1

            def maybe_capture_p2_tail_snapshot():
                nonlocal p2_tail_snap_i
                nonlocal p2_tail_snap_progress_pct
                nonlocal p2_tail_snap_skipped
                nonlocal p2_tail_snap_total
                nonlocal p2_tail_snap_pct
                nonlocal p2_tail_captured_this_i
                nonlocal p2_head_snap_i
                nonlocal p2_head_snap_progress_pct
                nonlocal p2_head_snap_skipped
                nonlocal p2_head_snap_total
                nonlocal p2_head_snap_pct
                nonlocal p2_head_captured_this_i
                if current_mid_total <= 0:
                    return
                prog_pct = 100.0 * (current_i_classified_j / current_mid_total)
                if (not p2_head_captured_this_i) and prog_pct <= 1.0:
                    p2_head_snap_i = pos_i
                    p2_head_snap_progress_pct = 1.0
                    p2_head_snap_skipped = current_i_skipped_j
                    p2_head_snap_total = current_i_classified_j
                    p2_head_snap_pct = (
                        100.0 * current_i_skipped_j / current_i_classified_j
                        if current_i_classified_j > 0
                        else 0.0
                    )
                    p2_head_captured_this_i = True
                if (not p2_tail_captured_this_i) and prog_pct >= 99.0:
                    p2_tail_snap_i = pos_i
                    p2_tail_snap_progress_pct = 99.0
                    p2_tail_snap_skipped = current_i_skipped_j
                    p2_tail_snap_total = current_i_classified_j
                    p2_tail_snap_pct = (
                        100.0 * current_i_skipped_j / current_i_classified_j
                        if current_i_classified_j > 0
                        else 0.0
                    )
                    p2_tail_captured_this_i = True

            if len(best_triples) == TOP_TRIPLES and pos_i + 2 < n_allowed:
                cutoff = max(best_triples[0][0], shared_floor.value)
                upper_bound = min(
                    max_entropy,
                    h1 + sorted_entropies[pos_i + 1] + sorted_entropies[pos_i + 2],
                )
                if upper_bound <= cutoff:
                    early_exit_count += outer_total - pos_i
                    if checkpoint_file is not None:
                        _write_checkpoint(checkpoint_file, build_checkpoint_payload(outer_total))
                    break

            pending = deque()
            j_evaluated = 0
            next_pos_j = pos_i + 1
            broke_j_loop = False
            first_p2_prune_j = None
            first_p2_prune_reported = False
            # Keep dispatch serial in order while allowing enough in-flight work
            # to saturate workers.
            max_pending = max(1, worker_count * chunk_size)

            while next_pos_j < n_allowed - 1 or pending:
                while not broke_j_loop and next_pos_j < n_allowed - 1 and len(pending) < max_pending:
                    pos_j = next_pos_j
                    j = sorted_indices[pos_j]
                    h2 = sorted_entropies[pos_j]
                    cutoff = (
                        max(best_triples[0][0], shared_floor.value)
                        if len(best_triples) == TOP_TRIPLES
                        else -1.0
                    )

                    if cutoff >= 0.0 and pos_j + 1 < n_allowed:
                        upper_bound = min(max_entropy, h1 + h2 + sorted_entropies[pos_j + 1])
                        if upper_bound <= cutoff:
                            remaining_j = (n_allowed - 1) - pos_j
                            progress["possible_j_completed"] += remaining_j
                            current_i_classified_j += remaining_j
                            current_i_skipped_j += remaining_j
                            current_mid_done = current_i_classified_j
                            maybe_log_pct_crossings(time.time())
                            maybe_capture_p2_tail_snapshot()
                            first_p2_prune_j = pos_j
                            if (
                                debug_prune
                                and pos_i == 0
                                and not first_p2_prune_reported
                            ):
                                line = (
                                    f"[DEBUG-FIRST-P2] i=0 first_prune_j={pos_j} "
                                    f"classified_j={current_i_classified_j:,} "
                                    f"skipped_j={current_i_skipped_j:,}"
                                )
                                print(line, flush=True)
                                try:
                                    with open(debug_prune_log, "a", encoding="ascii") as dbg:
                                        dbg.write(line + "\\n")
                                except OSError:
                                    pass
                                first_p2_prune_reported = True
                                if debug_stop_on_first_p2_i0:
                                    return
                            broke_j_loop = True
                            break

                    h12 = two_guess_entropy(row_i_243, matrix_u32[j])
                    if cutoff >= 0.0 and pos_j + 1 < n_allowed:
                        upper_bound = min(max_entropy, h12 + sorted_entropies[pos_j + 1])
                        if upper_bound <= cutoff:
                            remaining_j = (n_allowed - 1) - pos_j
                            progress["possible_j_completed"] += remaining_j
                            current_i_classified_j += remaining_j
                            current_i_skipped_j += remaining_j
                            current_mid_done = current_i_classified_j
                            maybe_log_pct_crossings(time.time())
                            maybe_capture_p2_tail_snapshot()
                            first_p2_prune_j = pos_j
                            if (
                                debug_prune
                                and pos_i == 0
                                and not first_p2_prune_reported
                            ):
                                line = (
                                    f"[DEBUG-FIRST-P2] i=0 first_prune_j={pos_j} "
                                    f"classified_j={current_i_classified_j:,} "
                                    f"skipped_j={current_i_skipped_j:,}"
                                )
                                print(line, flush=True)
                                try:
                                    with open(debug_prune_log, "a", encoding="ascii") as dbg:
                                        dbg.write(line + "\\n")
                                except OSError:
                                    pass
                                first_p2_prune_reported = True
                                if debug_stop_on_first_p2_i0:
                                    return
                            broke_j_loop = True
                            break

                    row12_scaled = row_i * 59049 + matrix_u32[j] * 243
                    async_result = pool.apply_async(_worker_k_branch, ((i, pos_j, row12_scaled, h12),))
                    pending.append(async_result)
                    j_evaluated += 1
                    next_pos_j += 1
                    emit_status(time.time())

                if pending:
                    result = pending.popleft().get()
                    progress["possible_j_completed"] += 1
                    progress["actual_j_completed"] += 1
                    current_i_classified_j += 1
                    current_mid_done = current_i_classified_j
                    maybe_log_pct_crossings(time.time())
                    maybe_capture_p2_tail_snapshot()
                    progress["possible_k_completed"] += result["possible_k"]
                    progress["actual_k_completed"] += result["actual_k"]
                    progress["skipped_k_completed"] += result["possible_k"] - result["actual_k"]
                    current_i_possible_k += result["possible_k"]
                    current_i_skipped_k += result["possible_k"] - result["actual_k"]
                    prev_j_index = current_j_index
                    prev_j_possible_k = current_j_possible_k
                    prev_j_skipped_k = current_j_skipped_k
                    current_j_index = result["pos_j"]
                    current_j_possible_k = result["possible_k"]
                    current_j_skipped_k = result["possible_k"] - result["actual_k"]
                    if current_j_possible_k > 0:
                        last_completed_j_p3_pct = (
                            100.0 * (current_j_skipped_k / current_j_possible_k)
                        )

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
                    emit_status(time.time())
                elif broke_j_loop:
                    break

            if not broke_j_loop:
                skipped_j = outer_possible_j - j_evaluated
                if skipped_j > 0:
                    if first_p2_prune_j is None:
                        first_p2_prune_j = pos_i + 1 + j_evaluated
                    progress["possible_j_completed"] += skipped_j
                    current_i_classified_j += skipped_j
                    current_i_skipped_j += skipped_j
                    current_mid_done = current_i_classified_j
                    maybe_log_pct_crossings(time.time())
                    maybe_capture_p2_tail_snapshot()

            if first_p2_prune_j is not None and run_first_p2_i is None:
                run_first_p2_i = pos_i
                run_first_p2_j = first_p2_prune_j
                run_first_p2_elapsed_s = time.time() - start_time
                if debug_prune_handle is not None:
                    debug_prune_handle.write(
                        f"run_first_p2,{run_first_p2_elapsed_s:.3f},{run_first_p2_i},{run_first_p2_j},"
                        ",,,,,,,,,,,,\n"
                    )

            if debug_prune_handle is not None:
                first_offset = -1
                tail_len = 0
                tail_prune_ratio = ""
                if first_p2_prune_j is not None:
                    first_offset = max(0, first_p2_prune_j - (pos_i + 1))
                    tail_len = max(0, current_mid_total - first_offset)
                    if tail_len > 0:
                        tail_prune_ratio = f"{(current_i_skipped_j / tail_len):.6f}"
                all_possible_j = progress["possible_j_completed"]
                all_skipped_j = max(0, all_possible_j - progress["actual_j_completed"])
                all_possible_k = progress["possible_k_completed"]
                all_skipped_k = progress["skipped_k_completed"]
                debug_prune_handle.write(
                    f"loop_i,{time.time()-start_time:.3f},{pos_i},,"
                    f",{current_mid_total},{current_i_classified_j},{current_i_skipped_j},"
                    f"{'' if first_p2_prune_j is None else first_p2_prune_j},"
                    f"{'' if first_offset < 0 else first_offset},"
                    f"{tail_len},{current_i_skipped_j},{tail_prune_ratio},"
                    f"{all_possible_j},{all_skipped_j},{all_possible_k},{all_skipped_k}\n"
                )


            progress["completed_i"] += 1
            prev_i_index = pos_i
            prev_i_classified_j = current_i_classified_j
            prev_i_skipped_j = current_i_skipped_j
            prev_i_possible_k = current_i_possible_k
            prev_i_skipped_k = current_i_skipped_k
            if current_i_classified_j > 0:
                last_completed_i_p2_pct = (
                    100.0 * (current_i_skipped_j / current_i_classified_j)
                )
            if debug_prune and pos_i <= 2:
                p2_i = (
                    100.0 * current_i_skipped_j / current_i_classified_j
                    if current_i_classified_j > 0
                    else 0.0
                )
                first_marker = "none" if first_p2_prune_j is None else str(first_p2_prune_j)
                line = (
                    f"[DEBUG] i={pos_i} classified_j={current_i_classified_j:,} "
                    f"skipped_j={current_i_skipped_j:,} p2_i={p2_i:.1f}% first_prune_j={first_marker}"
                )
                print(line, flush=True)
                try:
                    with open(debug_prune_log, "a", encoding="ascii") as dbg:
                        dbg.write(line + "\\n")
                except OSError:
                    pass
            current_mid_done = current_mid_total
            if checkpoint_file is not None:
                _write_checkpoint(checkpoint_file, build_checkpoint_payload(pos_i + 1))
            emit_status(time.time())

        if debug_prune_handle is not None:
            debug_prune_handle.close()

    best_triples.sort(reverse=True)

    if progress["completed_i"] < outer_total and progress["floor"] is not None:
        print(
            "Outer-level early exits occurred once remaining branch upper bounds "
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
        default=2,
        help="In-flight task multiplier per worker in -words 3 mode (default: 2).",
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
        "-debug-prune-file",
        type=str,
        default=None,
        help="Optional line-buffered CSV path for prune diagnostics (1%% crossings + per-i first-prune/total-prunes records).",
    )
    parser.add_argument(
        "-checkpoint-file",
        type=str,
        default=".wordle3_checkpoint.json",
        help="Checkpoint JSON path for -words 3 (default: .wordle3_checkpoint.json).",
    )
    parser.add_argument(
        "-resume",
        choices=("ask", "auto", "new"),
        default="ask",
        help="Checkpoint behavior when file exists: ask (default), auto, or new (ignore old checkpoint).",
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
            debug_prune_file=args.debug_prune_file,
            checkpoint_file=args.checkpoint_file,
            resume_mode=args.resume,
            history_seconds=args.history_seconds,
        )
        return

    run_two_guess(answers, allowed, matrix, args.verbose)


if __name__ == "__main__":
    main()
