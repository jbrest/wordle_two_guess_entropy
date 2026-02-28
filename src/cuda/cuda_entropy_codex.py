"""
cuda_entropy_codex.py

Codex CUDA wrapper for GPU-native two-guess and three-guess search.
"""

import ctypes
import hashlib
import heapq
import json
import os
import shutil
import subprocess
import time
import textwrap
from pathlib import Path

import numpy as np
import torch

CUDA_DIR = Path(__file__).parent
KERNEL_SOURCE = CUDA_DIR / "batched_entropy_codex.cu"
KERNEL_LIB = CUDA_DIR / "batched_entropy_codex.so"
BLOCK_THREADS = 128
CHECKPOINT_SCHEMA_VERSION = 4

_cuda_lib = None


def _fmt_dhhmm(seconds: float) -> str:
    seconds = max(0, int(seconds))
    days = seconds // 86400
    rem = seconds % 86400
    hours = rem // 3600
    minutes = (rem % 3600) // 60
    return f"{days}:{hours:02d}:{minutes:02d}"


def _fmt_hhmmss(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours = seconds // 3600
    rem = seconds % 3600
    minutes = rem // 60
    secs = rem % 60
    return f"{hours}:{minutes:02d}:{secs:02d}"


def _fmt_signed_dhhmm(seconds: float) -> str:
    sign = "-" if seconds < 0 else "+"
    return f"{sign}{_fmt_dhhmm(abs(seconds))}"


def _fmt_adaptive_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 1.0:
        return f"{seconds * 1000.0:.1f} ms"
    if seconds < 60.0:
        return f"{seconds:.1f} s"
    if seconds < 3600.0:
        mm = int(seconds // 60.0)
        ss = int(seconds % 60.0)
        return f"{mm}:{ss:02d}"
    hh = int(seconds // 3600.0)
    mm = int((seconds % 3600.0) // 60.0)
    return f"{hh}:{mm:02d}"


def _fmt_adaptive_duration_labeled(seconds: float) -> str:
    base = _fmt_adaptive_duration(seconds)
    if base.endswith(" ms") or base.endswith(" s"):
        return base
    if seconds >= 3600.0:
        return f"{base} h"
    if seconds >= 60.0:
        return f"{base} m"
    return base


def _emit_progress(
    progress_mode: str,
    title: str,
    chunk_idx: int,
    n_chunks: int,
    n_chunks_original: int | None,
    pct: float,
    floor_bits: float,
    best_bits: float,
    top_k: int,
    candidate_n: int,
    full_prune_blocks: int,
    blocks_in_chunk: int,
    full_prune_pct: float,
    cumulative_full_prune_blocks: int | None,
    cumulative_blocks_considered: int | None,
    pruned_threads: int,
    active_threads: int,
    prune_thread_pct: float,
    cumulative_active_threads: int | None,
    chunk_ms: float,
    start_time: float,
    processed_tasks: int,
    total_tasks: int,
    baseline_chunk_ms: float | None,
    cumulative_pruned_threads: int,
    precompute_note: str | None = None,
    prune_count_line: str | None = None,
    queue_wait_line: str | None = None,
    show_pruning_line: bool = True,
    show_prune_n3_line: bool = True,
    prune_n3_label: str = "Prune N^3",
    chunking_label: str | None = None,
    pruning_floor_mode: str | None = None,
    prune_current_considered: int | None = None,
    cumulative_prune_considered: int | None = None,
    prune_tail_current: int | None = None,
    prune_tail_cumulative: int | None = None,
    prune_singleton_current: int | None = None,
    prune_singleton_cumulative: int | None = None,
    last_elevated_chunk_idx: int | None = None,
    last_global_max_chunk_idx: int | None = None,
):
    if progress_mode == "off":
        return

    # Auto-fallback: when terminal is too narrow, skip dashboard tables.
    # This keeps output readable in small tmux panes.
    min_dashboard_cols = 96
    if progress_mode == "dashboard":
        term_cols = shutil.get_terminal_size(fallback=(120, 30)).columns
        if term_cols < min_dashboard_cols:
            if not getattr(_emit_progress, "_narrow_warned", False):
                print(
                    f"Dashboard disabled: terminal width {term_cols} < {min_dashboard_cols}; "
                    "falling back to compact progress output.",
                    flush=True,
                )
                setattr(_emit_progress, "_narrow_warned", True)
            progress_mode = "line"
        else:
            setattr(_emit_progress, "_narrow_warned", False)

    elapsed_s = max(0.0, time.perf_counter() - start_time)
    done = max(1, chunk_idx)
    avg_chunk_s = elapsed_s / done
    rem_chunks = max(0, n_chunks - chunk_idx)
    eta_s = rem_chunks * avg_chunk_s
    ett_s = elapsed_s + eta_s
    chunk_s = max(chunk_ms / 1000.0, 1e-9)
    cur_rate = int(round((total_tasks / n_chunks) / chunk_s))
    avg_rate = int(round(processed_tasks / max(elapsed_s, 1e-9)))

    baseline_line = "Baseline:   N/A"
    baseline_ett_s = None
    baseline_eta_s = None
    if baseline_chunk_ms is not None and baseline_chunk_ms > 0:
        baseline_chunk_s = baseline_chunk_ms / 1000.0
        baseline_ett_s = baseline_chunk_s * n_chunks
        baseline_eta_s = max(0.0, baseline_ett_s - elapsed_s)
        speedup = baseline_ett_s / max(ett_s, 1e-9)
        baseline_line = (
            f"Baseline:   ETT@chunk1-rate {_fmt_dhhmm(baseline_ett_s)} "
            f"| speedup x{speedup:.2f}"
        )

    if progress_mode == "dashboard":
        print("\033[2J\033[H", end="")
        print(title)
        print("")
        if chunking_label is not None:
            method_line = f"Chunking:   {chunking_label}"
            if pruning_floor_mode is not None:
                method_line += f" | Pruning floor: {pruning_floor_mode}"
            print(method_line)
            print("")
        eliminated_chunks = 0
        eliminated_pct = 0.0
        progress_text = f"chunk {chunk_idx:,}/{n_chunks:,} ({pct:.3f}%)"
        pruned_chunks_text = "0 (0.000%)"
        if n_chunks_original is not None and n_chunks_original != n_chunks:
            eliminated_chunks = max(0, n_chunks_original - n_chunks)
            eliminated_pct = 100.0 * eliminated_chunks / max(1, n_chunks_original)
            pruned_chunks_text = f"{eliminated_chunks:,} ({eliminated_pct:.3f}%) from original {n_chunks_original:,}"

        # Use dispatch-considered accounting when provided (hybrid), else kernel-active accounting.
        current_considered = int(prune_current_considered) if prune_current_considered is not None else int(active_threads)
        cumulative_considered_for_prune = (
            int(cumulative_prune_considered)
            if cumulative_prune_considered is not None else
            int(cumulative_active_threads if cumulative_active_threads is not None else current_considered)
        )
        current_pct_total = 100.0 * pruned_threads / max(total_tasks, 1)
        cumulative_pct_total = 100.0 * cumulative_pruned_threads / max(total_tasks, 1)
        current_pct_considered = 100.0 * pruned_threads / max(current_considered, 1)
        cumulative_pct_considered = 100.0 * cumulative_pruned_threads / max(cumulative_considered_for_prune, 1)

        p_label_w = 20
        p_val_w = 64

        def _ptbl_row(label: str, val: str) -> str:
            return f"| {label:<{p_label_w}} | {val:<{p_val_w}} |"

        def _ptbl_rows_wrapped(label: str, val: str) -> list[str]:
            chunks = textwrap.wrap(val, width=p_val_w) or [""]
            rows = []
            for idx, part in enumerate(chunks):
                row_label = label if idx == 0 else ""
                rows.append(_ptbl_row(row_label, part))
            return rows

        sep_p = "|" + "-" * (len(_ptbl_row("", "")) - 2) + "|"
        print(sep_p)
        print(f"| {'Progress & State':^{len(sep_p) - 4}} |")
        print(sep_p)
        avg_chunk_ms = avg_chunk_s * 1000.0
        for row in _ptbl_rows_wrapped("Progress", progress_text):
            print(row)
        for row in _ptbl_rows_wrapped("Pruned chunks", pruned_chunks_text):
            print(row)
        best_s = f"{best_bits:.4f} bits" if best_bits >= 0.0 else "N/A"
        gate_s = f"{floor_bits:.4f} bits" if floor_bits >= 0.0 else "N/A"
        for row in _ptbl_rows_wrapped("Max entropy", f"#1 {best_s} | #{top_k + 1} gate {gate_s}"):
            print(row)
        elev_chunk_s = f"{last_elevated_chunk_idx:,}" if last_elevated_chunk_idx is not None else "-"
        gmax_chunk_s = f"{last_global_max_chunk_idx:,}" if last_global_max_chunk_idx is not None else "-"
        for row in _ptbl_rows_wrapped("Last update chunk", f"elevated {elev_chunk_s} | global max {gmax_chunk_s}"):
            print(row)
        for row in _ptbl_rows_wrapped("Elevated this chunk", f"{candidate_n:,} candidates"):
            print(row)
        for row in _ptbl_rows_wrapped("Chunk time", f"current {chunk_ms:.1f} ms, average {avg_chunk_ms:.1f} ms"):
            print(row)
        for row in _ptbl_rows_wrapped("Rate", f"current {cur_rate:,} triples/sec, average {avg_rate:,} triples/sec"):
            print(row)
        print(sep_p)

        label_w = 22
        col_w = 20

        def _tbl_row(label: str, cur: str, cum: str) -> str:
            return f"| {label:<{label_w}} | {cur:>{col_w}} | {cum:>{col_w}} |"

        def _tbl_header(label: str, cur: str, cum: str) -> str:
            return f"| {label:^{label_w}} | {cur:^{col_w}} | {cum:^{col_w}} |"

        sep = "|" + "-" * (len(_tbl_row("", "", "")) - 2) + "|"
        print("")
        print(sep)
        print(f"| {'Pruning Results':^{len(sep) - 4}} |")
        print(sep)
        print(_tbl_header("", "Current Chunk", "Cumulative"))
        print(sep)
        # Chunk-level totals in current column; running totals in cumulative column.
        print(_tbl_row("Total Elements", f"{current_considered:,}", f"{cumulative_considered_for_prune:,}"))
        print(_tbl_row("Elements Pruned", f"{pruned_threads:,}", f"{cumulative_pruned_threads:,}"))
        print(_tbl_row("% of Total", f"{current_pct_total:.3f}%", f"{cumulative_pct_total:.3f}%"))
        print(_tbl_row("% of Considered", f"{current_pct_considered:.3f}%", f"{cumulative_pct_considered:.3f}%"))
        if (
            prune_tail_current is not None and
            prune_tail_cumulative is not None and
            prune_singleton_current is not None and
            prune_singleton_cumulative is not None
        ):
            singleton_pct_current = 100.0 * prune_singleton_current / max(pruned_threads, 1)
            singleton_pct_cumulative = 100.0 * prune_singleton_cumulative / max(cumulative_pruned_threads, 1)
            print(sep)
            print(_tbl_header("Prune Mix", "", ""))
            print(sep)
            print(_tbl_row("Tail", f"{prune_tail_current:,}", f"{prune_tail_cumulative:,}"))
            print(_tbl_row("Singleton", f"{prune_singleton_current:,}", f"{prune_singleton_cumulative:,}"))
            print(_tbl_row("% Singleton", f"{singleton_pct_current:.3f}%", f"{singleton_pct_cumulative:.3f}%"))
        print(sep)

        # Pre-dispatch N^3 line is redundant with the pruning table and intentionally omitted in dashboard mode.
        if prune_count_line:
            print(prune_count_line)

        t_label_w = 12
        t_elapsed_w = 10
        t_eta_w = 10
        t_ett_w = 10
        t_dabs_w = 10
        t_dpct_w = 9

        def _ttbl_row(label: str, elapsed_v: str, eta_v: str, ett_v: str, delta_abs_v: str, delta_pct_v: str) -> str:
            return (
                f"| {label:<{t_label_w}} | {elapsed_v:>{t_elapsed_w}} | {eta_v:>{t_eta_w}} | "
                f"{ett_v:>{t_ett_w}} | {delta_abs_v:>{t_dabs_w}} | {delta_pct_v:>{t_dpct_w}} |"
            )

        sep_t = "|" + "-" * (len(_ttbl_row("", "", "", "", "", "")) - 2) + "|"
        print("")
        print(sep_t)
        print(f"| {'Timing (dd:hh:mm)':^{len(sep_t) - 4}} |")
        print(sep_t)
        delta_merge_w = t_dabs_w + t_dpct_w + 3
        print(
            f"| {'':^{t_label_w}} | {'Elapsed':^{t_elapsed_w}} | {'ETA':^{t_eta_w}} | "
            f"{'ETT':^{t_ett_w}} | {'ΔETT':^{delta_merge_w}} |"
        )
        print(sep_t)
        if baseline_ett_s is not None and baseline_eta_s is not None:
            original_eta = _fmt_dhhmm(baseline_eta_s)
            original_ett = _fmt_dhhmm(baseline_ett_s)
            gains_delta_s = ett_s - baseline_ett_s
            gains_delta_pct = 100.0 * gains_delta_s / max(baseline_ett_s, 1e-9)
        else:
            original_eta = "N/A"
            original_ett = "N/A"
            gains_delta_s = None
            gains_delta_pct = None
        gains_delta_abs = _fmt_signed_dhhmm(gains_delta_s) if gains_delta_s is not None else "N/A"
        gains_delta_pct_s = f"{gains_delta_pct:+.2f}%" if gains_delta_pct is not None else "N/A"
        print(_ttbl_row("Original", _fmt_dhhmm(elapsed_s), original_eta, original_ett, "", ""))
        print(_ttbl_row("To Date", _fmt_dhhmm(elapsed_s), _fmt_dhhmm(eta_s), _fmt_dhhmm(ett_s), gains_delta_abs, gains_delta_pct_s))
        if (
            cumulative_prune_considered is not None and
            cumulative_prune_considered > 0 and
            elapsed_s > 0.0
        ):
            prune_rate = cumulative_pruned_threads / max(cumulative_prune_considered, 1)
            prune_rate = max(0.0, min(prune_rate, 0.999999))
            launched_done = max(0.0, cumulative_prune_considered - cumulative_pruned_threads)
            est_total_launched = max(launched_done, total_tasks * (1.0 - prune_rate))
            launch_rate = launched_done / max(elapsed_s, 1e-9)
            rem_launched = max(0.0, est_total_launched - launched_done)
            eta_prune_s = rem_launched / max(launch_rate, 1e-9)
            ett_prune_s = elapsed_s + eta_prune_s
            if baseline_ett_s is not None:
                ett_delta_s = ett_prune_s - baseline_ett_s
                ett_delta_pct = 100.0 * ett_delta_s / max(baseline_ett_s, 1e-9)
                ett_delta_abs = _fmt_signed_dhhmm(ett_delta_s)
                ett_delta_pct_s = f"{ett_delta_pct:+.2f}%"
            else:
                ett_delta_abs = "N/A"
                ett_delta_pct_s = "N/A"
            print(_ttbl_row("Extrapolated", _fmt_dhhmm(elapsed_s), _fmt_dhhmm(eta_prune_s), _fmt_dhhmm(ett_prune_s), ett_delta_abs, ett_delta_pct_s))
        else:
            print(_ttbl_row("Extrapolated", _fmt_dhhmm(elapsed_s), "N/A", "N/A", "N/A", "N/A"))
        print(sep_t)

        if queue_wait_line:
            diag_labels = ("CPU Loop", "GPU Execute", "CPU Build", "CPU Overhead")
            parsed_metrics = []
            for line in queue_wait_line.splitlines():
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 3 and parts[0].startswith("cur ") and parts[1].startswith("tot ") and parts[2].startswith("avg "):
                    parsed_metrics.append((parts[0][4:].strip(), parts[1][4:].strip(), parts[2][4:].strip()))
            if parsed_metrics:
                m_w = 16
                cur_w = 12
                tot_w = 12
                avg_w = 12

                def _rrow(metric: str, cur: str, tot: str, avg: str) -> str:
                    return f"| {metric:<{m_w}} | {cur:>{cur_w}} | {tot:>{tot_w}} | {avg:>{avg_w}} |"

                sep_r = "|" + "-" * (len(_rrow("", "", "", "")) - 2) + "|"
                print("")
                print(sep_r)
                print(f"| {'Runtime Notes':^{len(sep_r) - 4}} |")
                print(sep_r)
                print(_rrow("Metric", "Current", "Total", "Average"))
                print(sep_r)
                for idx, (cur_v, tot_v, avg_v) in enumerate(parsed_metrics):
                    label = diag_labels[idx] if idx < len(diag_labels) else f"Row {idx + 1}"
                    print(_rrow(label, cur_v, tot_v, avg_v))
                print(sep_r)
        else:
            print("")
            print(sep_p)
            print(f"| {'Runtime Notes':^{len(sep_p) - 4}} |")
            print(sep_p)
            for row in _ptbl_rows_wrapped("Status", "No runtime metrics available yet"):
                print(row)
            print(sep_p)
        print("", flush=True)
        return

    best_base = f"{best_bits:.4f}" if best_bits >= 0.0 else "N/A"
    gate_base = f"{floor_bits:.4f}" if floor_bits >= 0.0 else "N/A"
    if n_chunks_original is not None and n_chunks_original != n_chunks:
        eliminated_chunks = max(0, n_chunks_original - n_chunks)
        eliminated_pct = 100.0 * eliminated_chunks / max(1, n_chunks_original)
        base = (
            f"  Kernel chunks: {chunk_idx}/{n_chunks} ({pct:.6f}%) "
            f"(original {n_chunks_original}, eliminated {eliminated_pct:.3f}%) "
            f"max={best_base} gate#{top_k + 1}={gate_base} elevated={candidate_n:,} "
        )
    else:
        base = (
            f"  Kernel chunks: {chunk_idx}/{n_chunks} ({pct:.6f}%) "
            f"max={best_base} gate#{top_k + 1}={gate_base} elevated={candidate_n:,} "
        )
    if show_pruning_line:
        base += (
            f"full_prune_blocks={full_prune_blocks:,}/{blocks_in_chunk:,} "
            f"({full_prune_pct:.1f}%) pruned_threads={pruned_threads:,}/"
            f"{active_threads:,} ({prune_thread_pct:.3f}%) "
            f"prune_total={cumulative_pruned_threads:,}/{total_tasks:,} "
        )
    base += f"chunk_ms={chunk_ms:.1f}"
    print(base, flush=True)
    if prune_count_line:
        print(f"    {prune_count_line}", flush=True)
    if queue_wait_line:
        print(f"    {queue_wait_line}", flush=True)


def compile_cuda_kernel(force=False):
    if KERNEL_LIB.exists() and not force:
        if KERNEL_SOURCE.stat().st_mtime <= KERNEL_LIB.stat().st_mtime:
            print(f"CUDA kernel already compiled: {KERNEL_LIB}")
            return KERNEL_LIB

    print("Compiling CUDA kernel...")
    print(f"  Source: {KERNEL_SOURCE}")
    print(f"  Output: {KERNEL_LIB}")

    try:
        result = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True,
            text=True,
            check=True,
        )
        print(f"  Using: {result.stdout.splitlines()[0]}")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        raise RuntimeError(
            "nvcc not found. Please install CUDA toolkit:\n"
            "  sudo apt-get install cuda-toolkit-12-6"
        ) from e

    cmd = [
        "nvcc",
        "-shared",
        "-Xcompiler",
        "-fPIC",
        "-O3",
        "--use_fast_math",
        "-o",
        str(KERNEL_LIB),
        str(KERNEL_SOURCE),
    ]

    try:
        subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            cwd=str(CUDA_DIR),
        )
        print("  Compilation successful!")
        return KERNEL_LIB
    except subprocess.CalledProcessError as e:
        print("  Compilation failed!")
        print(f"  Command: {' '.join(cmd)}")
        if e.stdout:
            print(f"  stdout:\n{e.stdout}")
        if e.stderr:
            print(f"  stderr:\n{e.stderr}")
        raise RuntimeError("CUDA kernel compilation failed") from e


def load_cuda_kernel():
    global _cuda_lib

    if _cuda_lib is not None:
        return _cuda_lib

    # Always run staleness check so signature/source changes recompile
    # before loading an existing shared object.
    compile_cuda_kernel(force=False)

    try:
        _cuda_lib = ctypes.CDLL(str(KERNEL_LIB))

        _cuda_lib.launch_two_guess_search_codex.argtypes = [
            ctypes.c_void_p,  # d_matrix_u8
            ctypes.c_int,  # n_allowed
            ctypes.c_int,  # n_answers
            ctypes.c_void_p,  # d_sorted_indices
            ctypes.c_void_p,  # d_sorted_entropies
            ctypes.c_longlong,  # task_start
            ctypes.c_int,  # n_tasks
            ctypes.c_void_p,  # d_diag_offsets
            ctypes.c_int,  # n_diags
            ctypes.c_int,  # decode_mode (0=row, 1=antidiag)
            ctypes.c_void_p,  # d_floor_q
            ctypes.c_void_p,  # d_out_count
            ctypes.c_int,  # out_capacity
            ctypes.c_void_p,  # d_out_overflow_flag
            ctypes.c_void_p,  # d_out_overflow_dropped
            ctypes.c_void_p,  # d_out_full_prune_blocks
            ctypes.c_void_p,  # d_out_active_threads
            ctypes.c_void_p,  # d_out_bound_pass_threads
            ctypes.c_void_p,  # d_out_entropy_q
            ctypes.c_void_p,  # d_out_i
            ctypes.c_void_p,  # d_out_j
        ]
        _cuda_lib.launch_two_guess_search_codex.restype = None

        _cuda_lib.launch_three_guess_search_codex.argtypes = [
            ctypes.c_void_p,  # d_matrix_u8
            ctypes.c_int,  # n_allowed
            ctypes.c_int,  # n_answers
            ctypes.c_void_p,  # d_sorted_indices
            ctypes.c_void_p,  # d_sorted_entropies
            ctypes.c_longlong,  # task_start
            ctypes.c_int,  # n_tasks
            ctypes.c_void_p,  # d_floor_q
            ctypes.c_void_p,  # d_out_count
            ctypes.c_int,  # out_capacity
            ctypes.c_void_p,  # d_out_overflow_flag
            ctypes.c_void_p,  # d_out_overflow_dropped
            ctypes.c_void_p,  # d_out_full_prune_blocks
            ctypes.c_void_p,  # d_out_active_threads
            ctypes.c_void_p,  # d_out_bound_pass_threads
            ctypes.c_void_p,  # d_out_entropy_q
            ctypes.c_void_p,  # d_out_i
            ctypes.c_void_p,  # d_out_j
            ctypes.c_void_p,  # d_out_k
        ]
        _cuda_lib.launch_three_guess_search_codex.restype = None

        _cuda_lib.launch_three_guess_search_indexed_codex.argtypes = [
            ctypes.c_void_p,  # d_matrix_u8
            ctypes.c_int,  # n_allowed
            ctypes.c_int,  # n_answers
            ctypes.c_void_p,  # d_sorted_indices
            ctypes.c_void_p,  # d_task_i_sorted
            ctypes.c_void_p,  # d_task_j_sorted
            ctypes.c_void_p,  # d_task_k_sorted
            ctypes.c_int,  # n_tasks
            ctypes.c_void_p,  # d_floor_q
            ctypes.c_void_p,  # d_out_count
            ctypes.c_int,  # out_capacity
            ctypes.c_void_p,  # d_out_overflow_flag
            ctypes.c_void_p,  # d_out_overflow_dropped
            ctypes.c_void_p,  # d_out_full_prune_blocks
            ctypes.c_void_p,  # d_out_active_threads
            ctypes.c_void_p,  # d_out_bound_pass_threads
            ctypes.c_void_p,  # d_out_entropy_q
            ctypes.c_void_p,  # d_out_i
            ctypes.c_void_p,  # d_out_j
            ctypes.c_void_p,  # d_out_k
        ]
        _cuda_lib.launch_three_guess_search_indexed_codex.restype = None

        _cuda_lib.launch_compute_two_guess_entropy_q_codex.argtypes = [
            ctypes.c_void_p,  # d_matrix_u8
            ctypes.c_int,  # n_allowed
            ctypes.c_int,  # n_answers
            ctypes.c_void_p,  # d_sorted_indices
            ctypes.c_longlong,  # task_start
            ctypes.c_int,  # n_tasks
            ctypes.c_void_p,  # d_out_entropy_q
        ]
        _cuda_lib.launch_compute_two_guess_entropy_q_codex.restype = None
        print(f"Loaded CUDA kernel: {KERNEL_LIB}")
        return _cuda_lib
    except Exception as e:
        raise RuntimeError(f"Failed to load CUDA kernel library: {e}") from e


def build_antidiagonal_offsets(n_allowed: int) -> np.ndarray:
    """Prefix offsets for anti-diagonal enumeration of i<j pairs."""
    n_diags = 2 * n_allowed - 3
    offsets = np.zeros(n_diags + 1, dtype=np.int32)

    total = 0
    for d in range(n_diags):
        s = d + 1
        i_min = max(0, s - (n_allowed - 1))
        i_max = min(n_allowed - 2, (s - 1) // 2)
        length = max(0, i_max - i_min + 1)
        total += length
        offsets[d + 1] = total

    expected = n_allowed * (n_allowed - 1) // 2
    if total != expected:
        raise RuntimeError(
            f"Anti-diagonal offsets mismatch: built={total}, expected={expected}"
        )

    return offsets


def _save_checkpoint(checkpoint_file: str, state: dict):
    tmp = f"{checkpoint_file}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, checkpoint_file)


def _load_checkpoint(checkpoint_file: str) -> dict | None:
    if not checkpoint_file or not os.path.exists(checkpoint_file):
        return None
    with open(checkpoint_file, "r", encoding="utf-8") as f:
        return json.load(f)


def pair_row_id(i: int, j: int, n_allowed: int) -> int:
    return i * n_allowed - (i * (i + 1)) // 2 + (j - i - 1)


def load_or_build_h12_q_cache_gpu(
    matrix_u8_gpu: torch.Tensor,
    sorted_indices_gpu: torch.Tensor,
    n_answers: int,
    cache_base: str,
    chunk_tasks: int = 2_000_000,
) -> tuple[np.ndarray, str]:
    lib = load_cuda_kernel()
    n_allowed = int(matrix_u8_gpu.shape[0])
    total_pairs = n_allowed * (n_allowed - 1) // 2

    npy_path = f"{cache_base}.npy"
    meta_path = f"{cache_base}.meta.json"
    sorted_idx_cpu = sorted_indices_gpu.cpu().numpy().astype(np.int32, copy=False)
    sig = hashlib.sha1(sorted_idx_cpu.tobytes()).hexdigest()

    if os.path.exists(npy_path) and os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            ok = (
                int(meta.get("version", -1)) == CHECKPOINT_SCHEMA_VERSION and
                int(meta.get("n_allowed", -1)) == n_allowed and
                int(meta.get("n_answers", -1)) == n_answers and
                int(meta.get("total_pairs", -1)) == total_pairs and
                meta.get("sorted_indices_sha1", "") == sig
            )
            if ok:
                arr = np.load(npy_path, mmap_mode="r")
                if arr.dtype == np.int32 and arr.size == total_pairs:
                    build_s = meta.get("build_seconds")
                    if isinstance(build_s, (int, float)):
                        note = (
                            f"H12 precompute: {_fmt_hhmmss(float(build_s))} "
                            f"(cached; build time from prior run, excluded from this run)"
                        )
                    else:
                        note = (
                            "H12 precompute: unknown "
                            "(cached; build time from prior run, excluded from this run)"
                        )
                    print(f"Loaded H12 cache: {npy_path}")
                    print(note)
                    return arr, note
        except Exception:
            pass

    print(f"Building H12 cache on GPU ({total_pairs:,} pairs)...")
    d_out = torch.empty((total_pairs,), dtype=torch.int32, device="cuda")
    n_chunks = (total_pairs + chunk_tasks - 1) // chunk_tasks
    t0 = time.perf_counter()
    for idx, start in enumerate(range(0, total_pairs, chunk_tasks), start=1):
        n_tasks = min(chunk_tasks, total_pairs - start)
        chunk_t0 = time.perf_counter()
        lib.launch_compute_two_guess_entropy_q_codex(
            matrix_u8_gpu.data_ptr(),
            n_allowed,
            n_answers,
            sorted_indices_gpu.data_ptr(),
            int(start),
            int(n_tasks),
            d_out.data_ptr(),
        )
        torch.cuda.synchronize()
        if idx == 1 or idx % 10 == 0 or idx == n_chunks:
            pct = 100.0 * min(start + n_tasks, total_pairs) / total_pairs
            elapsed_s = time.perf_counter() - t0
            done_pairs = min(start + n_tasks, total_pairs)
            avg_rate = done_pairs / max(elapsed_s, 1e-9)
            rem_pairs = total_pairs - done_pairs
            eta_s = rem_pairs / max(avg_rate, 1e-9)
            chunk_ms = (time.perf_counter() - chunk_t0) * 1000.0
            print(
                f"  H12 cache chunks: {idx}/{n_chunks} ({pct:.1f}%) "
                f"chunk_ms={chunk_ms:.1f} avg_rate={avg_rate:,.0f}/s "
                f"elapsed={_fmt_dhhmm(elapsed_s)} ETA={_fmt_dhhmm(eta_s)}",
                flush=True,
            )

    arr = d_out.cpu().numpy()
    np.save(npy_path, arr)
    meta = {
        "version": CHECKPOINT_SCHEMA_VERSION,
        "n_allowed": n_allowed,
        "n_answers": n_answers,
        "total_pairs": total_pairs,
        "sorted_indices_sha1": sig,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f)
    build_seconds = float(time.perf_counter() - t0)
    meta["build_seconds"] = build_seconds
    meta["built_at_epoch_s"] = time.time()
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f)
    note = (
        f"H12 precompute: {_fmt_hhmmss(build_seconds)} "
        f"(built in this run; included in this run timing)"
    )
    print(f"Built H12 cache in {build_seconds:.1f}s: {npy_path}")
    print(note)
    return np.load(npy_path, mmap_mode="r"), note


def two_guess_search_cuda_codex(
    matrix_u8_gpu: torch.Tensor,
    sorted_indices_gpu: torch.Tensor,
    sorted_entropies_gpu: torch.Tensor,
    top_k: int = 50,
    chunk_tasks: int = 2_000_000,
    pair_order: str = "antidiag",
    launch_mode: str = "chunked",
    append_capacity: int = 2_000_000,
    progress_mode: str = "dashboard",
    checkpoint_file: str | None = None,
    resume: bool = True,
):
    """
    Run GPU-native two-guess search.

    Returns:
        List[(entropy_bits, i_orig, j_orig)] sorted descending.
    """
    lib = load_cuda_kernel()

    assert matrix_u8_gpu.is_cuda and matrix_u8_gpu.dtype == torch.uint8
    assert sorted_indices_gpu.is_cuda and sorted_indices_gpu.dtype == torch.int32
    assert sorted_entropies_gpu.is_cuda and sorted_entropies_gpu.dtype == torch.float32

    n_allowed, n_answers = matrix_u8_gpu.shape
    total_pairs = n_allowed * (n_allowed - 1) // 2
    n_chunks = (total_pairs + chunk_tasks - 1) // chunk_tasks

    if pair_order == "antidiag":
        decode_mode = 1
        offsets_cpu = build_antidiagonal_offsets(n_allowed)
        diag_offsets_gpu = torch.tensor(offsets_cpu, dtype=torch.int32, device="cuda")
        n_diags = len(offsets_cpu) - 1
    elif pair_order == "row":
        decode_mode = 0
        diag_offsets_gpu = torch.zeros((1,), dtype=torch.int32, device="cuda")
        n_diags = 0
    else:
        raise ValueError(f"Unknown pair_order: {pair_order}")

    floor_q = torch.tensor([-1], dtype=torch.int32, device="cuda")
    out_count = torch.zeros((1,), dtype=torch.int32, device="cuda")
    out_overflow_flag = torch.zeros((1,), dtype=torch.int32, device="cuda")
    out_overflow_dropped = torch.zeros((1,), dtype=torch.int32, device="cuda")
    out_full_prune_blocks = torch.zeros((1,), dtype=torch.int32, device="cuda")
    out_active_threads = torch.zeros((1,), dtype=torch.int64, device="cuda")
    out_bound_pass_threads = torch.zeros((1,), dtype=torch.int64, device="cuda")
    out_entropy_q = torch.empty((append_capacity,), dtype=torch.int32, device="cuda")
    out_i = torch.empty((append_capacity,), dtype=torch.int32, device="cuda")
    out_j = torch.empty((append_capacity,), dtype=torch.int32, device="cuda")

    top_heap = []  # min-heap of (entropy_q, i, j)
    current_floor_q = -1
    global_best_q = -1
    last_elevated_chunk_idx = None
    last_global_max_chunk_idx = None
    resume_task_start = 0
    resume_chunk_idx = 1
    elapsed_prev = 0.0
    baseline_chunk_ms = None
    cumulative_pruned_threads = 0
    cumulative_full_prune_blocks = 0
    cumulative_blocks_considered = 0
    cumulative_active_threads = 0
    precompute_note = None

    if checkpoint_file and resume and launch_mode == "chunked":
        ckpt = _load_checkpoint(checkpoint_file)
        if ckpt is not None:
            if (
                int(ckpt.get("version", -1)) == CHECKPOINT_SCHEMA_VERSION and
                ckpt.get("kind") == "two_guess" and
                int(ckpt.get("total_tasks", -1)) == int(total_pairs) and
                int(ckpt.get("chunk_tasks", -1)) == int(chunk_tasks) and
                ckpt.get("launch_mode") == launch_mode and
                ckpt.get("pair_order") == pair_order and
                int(ckpt.get("top_k", -1)) == int(top_k)
            ):
                top_heap = [
                    (int(a), int(b), int(c))
                    for (a, b, c) in ckpt.get("top_heap", [])
                ]
                heapq.heapify(top_heap)
                current_floor_q = int(ckpt.get("current_floor_q", -1))
                global_best_q = int(ckpt.get("global_best_q", max((x[0] for x in top_heap), default=-1)))
                last_elevated_chunk_idx = ckpt.get("last_elevated_chunk_idx")
                last_global_max_chunk_idx = ckpt.get("last_global_max_chunk_idx")
                resume_task_start = int(ckpt.get("next_task_start", 0))
                resume_chunk_idx = int(ckpt.get("next_chunk_idx", 1))
                if (
                    last_global_max_chunk_idx is None
                    and global_best_q >= 0
                    and resume_chunk_idx > 1
                ):
                    # Backfill provenance for checkpoints that predate this field.
                    # Exact origin chunk is unknown; best known bound is prior chunk.
                    last_global_max_chunk_idx = int(resume_chunk_idx - 1)
                elapsed_prev = float(ckpt.get("elapsed_s", 0.0))
                baseline_chunk_ms = ckpt.get("baseline_chunk_ms")
                if baseline_chunk_ms is not None:
                    baseline_chunk_ms = float(baseline_chunk_ms)
                cumulative_pruned_threads = int(ckpt.get("cumulative_pruned_threads", 0))
                cumulative_full_prune_blocks = int(ckpt.get("cumulative_full_prune_blocks", 0))
                cumulative_blocks_considered = int(ckpt.get("cumulative_blocks_considered", 0))
                cumulative_active_threads = int(ckpt.get("cumulative_active_threads", 0))
            else:
                print(
                    f"Ignoring incompatible checkpoint (schema/config mismatch): {checkpoint_file}",
                    flush=True,
                )

    run_t0 = time.perf_counter() - elapsed_prev

    if launch_mode == "single":
        ranges = [(1, 0, total_pairs)]
        n_chunks = 1
    elif launch_mode == "chunked":
        ranges = [
            (idx, start, min(chunk_tasks, total_pairs - start))
            for idx, start in enumerate(
                range(resume_task_start, total_pairs, chunk_tasks),
                start=resume_chunk_idx,
            )
        ]
    else:
        raise ValueError(f"Unknown launch_mode: {launch_mode}")

    for chunk_idx, task_start, n_tasks in ranges:
        chunk_t0 = time.perf_counter()
        floor_q.fill_(current_floor_q)
        out_count.zero_()
        out_overflow_flag.zero_()
        out_overflow_dropped.zero_()
        out_full_prune_blocks.zero_()
        out_active_threads.zero_()
        out_bound_pass_threads.zero_()

        lib.launch_two_guess_search_codex(
            matrix_u8_gpu.data_ptr(),
            n_allowed,
            n_answers,
            sorted_indices_gpu.data_ptr(),
            sorted_entropies_gpu.data_ptr(),
            int(task_start),
            n_tasks,
            diag_offsets_gpu.data_ptr(),
            n_diags,
            decode_mode,
            floor_q.data_ptr(),
            out_count.data_ptr(),
            append_capacity,
            out_overflow_flag.data_ptr(),
            out_overflow_dropped.data_ptr(),
            out_full_prune_blocks.data_ptr(),
            out_active_threads.data_ptr(),
            out_bound_pass_threads.data_ptr(),
            out_entropy_q.data_ptr(),
            out_i.data_ptr(),
            out_j.data_ptr(),
        )
        torch.cuda.synchronize()

        overflow_flag = int(out_overflow_flag.item())
        overflow_dropped = int(out_overflow_dropped.item())
        if overflow_flag != 0:
            raise RuntimeError(
                "GPU append buffer overflow: "
                f"capacity={append_capacity:,}, dropped={overflow_dropped:,}, "
                f"chunk_idx={chunk_idx}, task_start={task_start:,}, n_tasks={n_tasks:,}. "
                "Increase --append-capacity or use --launch-mode chunked."
            )

        candidate_n = int(out_count.item())
        full_prune_blocks = int(out_full_prune_blocks.item())
        active_threads = int(out_active_threads.item())
        bound_pass_threads = int(out_bound_pass_threads.item())
        if candidate_n > append_capacity:
            raise RuntimeError(
                f"Kernel produced invalid candidate count {candidate_n} > append_capacity {append_capacity}"
            )
        if candidate_n > n_tasks and launch_mode == "chunked":
            raise RuntimeError(
                f"Kernel produced invalid candidate count {candidate_n} > n_tasks {n_tasks}"
            )

        if candidate_n > 0:
            last_elevated_chunk_idx = int(chunk_idx)
            q = out_entropy_q[:candidate_n].cpu().numpy()
            ii = out_i[:candidate_n].cpu().numpy()
            jj = out_j[:candidate_n].cpu().numpy()
            for entropy_q_val, i_idx, j_idx in zip(q, ii, jj):
                entry = (int(entropy_q_val), int(i_idx), int(j_idx))
                if len(top_heap) < top_k:
                    heapq.heappush(top_heap, entry)
                elif entry[0] > top_heap[0][0]:
                    heapq.heapreplace(top_heap, entry)

            if len(top_heap) == top_k:
                current_floor_q = top_heap[0][0]
            chunk_best_q = max((entry[0] for entry in top_heap), default=-1)
            if chunk_best_q > global_best_q:
                global_best_q = int(chunk_best_q)
                last_global_max_chunk_idx = int(chunk_idx)

        blocks_in_chunk = (n_tasks + BLOCK_THREADS - 1) // BLOCK_THREADS
        full_prune_pct = (
            (100.0 * full_prune_blocks / blocks_in_chunk) if blocks_in_chunk > 0 else 100.0
        )
        prune_thread_pct = (
            100.0 * (1.0 - (bound_pass_threads / active_threads))
            if active_threads > 0 else 100.0
        )
        pruned_threads = max(0, active_threads - bound_pass_threads)
        cumulative_pruned_threads += pruned_threads
        cumulative_full_prune_blocks += full_prune_blocks
        cumulative_blocks_considered += blocks_in_chunk
        cumulative_active_threads += active_threads
        should_print = (
            progress_mode == "dashboard" or
            (candidate_n > 0) or
            (full_prune_blocks < blocks_in_chunk)
        )
        if should_print:
            pct = 100.0 * min(task_start + n_tasks, total_pairs) / total_pairs
            floor_bits = (current_floor_q / 1_000_000.0) if current_floor_q >= 0 else -1.0
            best_q = max((entry[0] for entry in top_heap), default=-1)
            best_bits = (best_q / 1_000_000.0) if best_q >= 0 else -1.0
            chunk_ms = (time.perf_counter() - chunk_t0) * 1000.0
            if baseline_chunk_ms is None:
                baseline_chunk_ms = chunk_ms
            processed_tasks = int(min(task_start + n_tasks, total_pairs))
            _emit_progress(
                progress_mode=progress_mode,
                title="GPU 2-word entropy search",
                chunk_idx=chunk_idx,
                n_chunks=n_chunks,
                n_chunks_original=None,
                pct=pct,
                floor_bits=floor_bits,
                best_bits=best_bits,
                top_k=top_k,
                candidate_n=candidate_n,
                full_prune_blocks=full_prune_blocks,
                blocks_in_chunk=blocks_in_chunk,
                full_prune_pct=full_prune_pct,
                cumulative_full_prune_blocks=cumulative_full_prune_blocks,
                cumulative_blocks_considered=cumulative_blocks_considered,
                pruned_threads=pruned_threads,
                active_threads=active_threads,
                prune_thread_pct=prune_thread_pct,
                cumulative_active_threads=cumulative_active_threads,
                chunk_ms=chunk_ms,
                start_time=run_t0,
                processed_tasks=processed_tasks,
                total_tasks=total_pairs,
                baseline_chunk_ms=baseline_chunk_ms,
                cumulative_pruned_threads=cumulative_pruned_threads,
                precompute_note=precompute_note,
                prune_current_considered=active_threads,
                cumulative_prune_considered=cumulative_active_threads,
                last_elevated_chunk_idx=last_elevated_chunk_idx,
                last_global_max_chunk_idx=last_global_max_chunk_idx,
            )

        if checkpoint_file and launch_mode == "chunked":
            next_task_start = int(task_start + n_tasks)
            state = {
                "version": CHECKPOINT_SCHEMA_VERSION,
                "kind": "two_guess",
                "total_tasks": int(total_pairs),
                "chunk_tasks": int(chunk_tasks),
                "launch_mode": launch_mode,
                "pair_order": pair_order,
                "top_k": int(top_k),
                "current_floor_q": int(current_floor_q),
                "global_best_q": int(global_best_q),
                "top_heap": [list(x) for x in top_heap],
                "last_elevated_chunk_idx": (
                    int(last_elevated_chunk_idx) if last_elevated_chunk_idx is not None else None
                ),
                "last_global_max_chunk_idx": (
                    int(last_global_max_chunk_idx) if last_global_max_chunk_idx is not None else None
                ),
                "next_task_start": next_task_start,
                "next_chunk_idx": int(chunk_idx + 1),
                "elapsed_s": float(max(0.0, time.perf_counter() - run_t0)),
                "baseline_chunk_ms": baseline_chunk_ms,
                "cumulative_pruned_threads": int(cumulative_pruned_threads),
                "cumulative_full_prune_blocks": int(cumulative_full_prune_blocks),
                "cumulative_blocks_considered": int(cumulative_blocks_considered),
                "cumulative_active_threads": int(cumulative_active_threads),
                "completed": bool(next_task_start >= total_pairs),
            }
            _save_checkpoint(checkpoint_file, state)

    results = []
    for entropy_q_val, i_idx, j_idx in top_heap:
        results.append((float(entropy_q_val) / 1_000_000.0, int(i_idx), int(j_idx)))

    results.sort(reverse=True)
    return results


def three_guess_search_cuda_codex(
    matrix_u8_gpu: torch.Tensor,
    sorted_indices_gpu: torch.Tensor,
    sorted_entropies_gpu: torch.Tensor,
    sorted_entropies_cpu: np.ndarray | None = None,
    top_k: int = 50,
    chunk_tasks: int = 2_000_000,
    launch_mode: str = "chunked",
    append_capacity: int = 2_000_000,
    progress_mode: str = "dashboard",
    checkpoint_file: str | None = None,
    resume: bool = True,
    dispatch_mode: str = "geometric",
    h12_cache_base: str = ".gpu_h12_cache",
    floor_source_mode: str = "t-1",
    return_stats: bool = False,
):
    """
    Run GPU-native three-guess search in row order (i < j < k).

    Returns:
        List[(entropy_bits, i_orig, j_orig, k_orig)] sorted descending, or
        (results, stats) when return_stats=True.
    """
    lib = load_cuda_kernel()

    assert matrix_u8_gpu.is_cuda and matrix_u8_gpu.dtype == torch.uint8
    assert sorted_indices_gpu.is_cuda and sorted_indices_gpu.dtype == torch.int32
    assert sorted_entropies_gpu.is_cuda and sorted_entropies_gpu.dtype == torch.float32

    n_allowed, n_answers = matrix_u8_gpu.shape
    total_triples = n_allowed * (n_allowed - 1) * (n_allowed - 2) // 6
    n_chunks = (total_triples + chunk_tasks - 1) // chunk_tasks

    floor_q = torch.tensor([-1], dtype=torch.int32, device="cuda")
    out_count = torch.zeros((1,), dtype=torch.int32, device="cuda")
    out_overflow_flag = torch.zeros((1,), dtype=torch.int32, device="cuda")
    out_overflow_dropped = torch.zeros((1,), dtype=torch.int32, device="cuda")
    out_full_prune_blocks = torch.zeros((1,), dtype=torch.int32, device="cuda")
    out_active_threads = torch.zeros((1,), dtype=torch.int64, device="cuda")
    out_bound_pass_threads = torch.zeros((1,), dtype=torch.int64, device="cuda")
    out_entropy_q = torch.empty((append_capacity,), dtype=torch.int32, device="cuda")
    out_i = torch.empty((append_capacity,), dtype=torch.int32, device="cuda")
    out_j = torch.empty((append_capacity,), dtype=torch.int32, device="cuda")
    out_k = torch.empty((append_capacity,), dtype=torch.int32, device="cuda")

    top_heap = []  # min-heap of (entropy_q, i, j, k)
    current_floor_q = -1
    global_best_q = -1
    last_elevated_chunk_idx = None
    last_global_max_chunk_idx = None
    resume_task_start = 0
    resume_chunk_idx = 1
    elapsed_prev = 0.0
    baseline_chunk_ms = None
    scheduler_pruned_total = 0
    scheduler_pruned_tail_total = 0
    scheduler_pruned_singleton_total = 0
    precompute_note = None
    cumulative_considered = 0
    dispatch_latency_ms_total = 0.0
    dispatch_latency_samples = 0
    cpu_loop_ms_total = 0.0
    cpu_loop_samples = 0
    cpu_build_ms_total = 0.0
    cpu_build_overhang_ms_total = 0.0
    cpu_build_samples = 0
    gpu_exec_ms_total = 0.0
    gpu_exec_samples = 0
    cpu_overhead_ms_total = 0.0
    cpu_overhead_samples = 0
    chunk_non_gpu_ms_total = 0.0
    chunk_non_gpu_samples = 0

    if dispatch_mode not in ("geometric", "hybrid"):
        raise ValueError(f"Unknown dispatch_mode: {dispatch_mode}")
    if floor_source_mode not in ("t", "t-1"):
        raise ValueError(f"Unknown floor_source_mode: {floor_source_mode}")

    if dispatch_mode == "hybrid":
        if launch_mode != "chunked":
            raise ValueError("dispatch_mode=hybrid requires launch_mode=chunked")
        if sorted_entropies_cpu is None:
            raise ValueError(
                "dispatch_mode=hybrid requires sorted_entropies_cpu"
            )
        # Quantized sorted single entropies, matching kernel floor quantization.
        ent_q_sorted = np.rint(sorted_entropies_cpu.astype(np.float64) * 1_000_000.0).astype(np.int32)
        neg_ent_q = -ent_q_sorted.astype(np.int64)
        h12_q_pairs, precompute_note = load_or_build_h12_q_cache_gpu(
            matrix_u8_gpu=matrix_u8_gpu,
            sorted_indices_gpu=sorted_indices_gpu,
            n_answers=n_answers,
            cache_base=h12_cache_base,
            chunk_tasks=chunk_tasks,
        )
        # Scheduler scan state over sorted index geometry.
        sched_i = 0
        sched_j = 1
        sched_k = 2
    cumulative_pruned_threads = 0
    cumulative_full_prune_blocks = 0
    cumulative_blocks_considered = 0
    cumulative_active_threads = 0

    if checkpoint_file and resume and launch_mode == "chunked":
        ckpt = _load_checkpoint(checkpoint_file)
        if ckpt is not None:
            if (
                int(ckpt.get("version", -1)) == CHECKPOINT_SCHEMA_VERSION and
                ckpt.get("kind") == "three_guess" and
                int(ckpt.get("total_tasks", -1)) == int(total_triples) and
                int(ckpt.get("chunk_tasks", -1)) == int(chunk_tasks) and
                ckpt.get("launch_mode") == launch_mode and
                int(ckpt.get("top_k", -1)) == int(top_k) and
                ckpt.get("dispatch_mode", "geometric") == dispatch_mode and
                ckpt.get("floor_source_mode", "t-1") == floor_source_mode
            ):
                top_heap = [
                    (int(a), int(b), int(c), int(d))
                    for (a, b, c, d) in ckpt.get("top_heap", [])
                ]
                heapq.heapify(top_heap)
                current_floor_q = int(ckpt.get("current_floor_q", -1))
                global_best_q = int(ckpt.get("global_best_q", max((x[0] for x in top_heap), default=-1)))
                last_elevated_chunk_idx = ckpt.get("last_elevated_chunk_idx")
                last_global_max_chunk_idx = ckpt.get("last_global_max_chunk_idx")
                resume_task_start = int(ckpt.get("next_task_start", 0))
                resume_chunk_idx = int(ckpt.get("next_chunk_idx", 1))
                if (
                    last_global_max_chunk_idx is None
                    and global_best_q >= 0
                    and resume_chunk_idx > 1
                ):
                    # Backfill provenance for checkpoints that predate this field.
                    # Exact origin chunk is unknown; best known bound is prior chunk.
                    last_global_max_chunk_idx = int(resume_chunk_idx - 1)
                elapsed_prev = float(ckpt.get("elapsed_s", 0.0))
                baseline_chunk_ms = ckpt.get("baseline_chunk_ms")
                if baseline_chunk_ms is not None:
                    baseline_chunk_ms = float(baseline_chunk_ms)
                scheduler_pruned_total = int(ckpt.get("scheduler_pruned_total", 0))
                scheduler_pruned_tail_total = int(ckpt.get("scheduler_pruned_tail_total", 0))
                scheduler_pruned_singleton_total = int(ckpt.get("scheduler_pruned_singleton_total", 0))
                cumulative_pruned_threads = int(
                    ckpt.get(
                        "cumulative_pruned_threads",
                        scheduler_pruned_total if dispatch_mode == "hybrid" else 0,
                    )
                )
                cumulative_considered = int(ckpt.get("cumulative_considered", 0))
                if dispatch_mode == "hybrid" and cumulative_considered <= 0:
                    # Backward-compatible reconstruction for older checkpoints
                    # that didn't persist cumulative_considered.
                    cumulative_considered = scheduler_pruned_total + resume_task_start
                cumulative_full_prune_blocks = int(ckpt.get("cumulative_full_prune_blocks", 0))
                cumulative_blocks_considered = int(ckpt.get("cumulative_blocks_considered", 0))
                cumulative_active_threads = int(
                    ckpt.get(
                        "cumulative_active_threads",
                        cumulative_considered if dispatch_mode == "hybrid" else 0,
                    )
                )
                dispatch_latency_ms_total = float(ckpt.get("dispatch_latency_ms_total", 0.0))
                dispatch_latency_samples = int(ckpt.get("dispatch_latency_samples", 0))
                cpu_loop_ms_total = float(ckpt.get("cpu_loop_ms_total", 0.0))
                cpu_loop_samples = int(ckpt.get("cpu_loop_samples", 0))
                cpu_build_ms_total = float(ckpt.get("cpu_build_ms_total", 0.0))
                cpu_build_overhang_ms_total = float(ckpt.get("cpu_build_overhang_ms_total", 0.0))
                cpu_build_samples = int(ckpt.get("cpu_build_samples", 0))
                gpu_exec_ms_total = float(ckpt.get("gpu_exec_ms_total", 0.0))
                gpu_exec_samples = int(ckpt.get("gpu_exec_samples", 0))
                cpu_overhead_ms_total = float(ckpt.get("cpu_overhead_ms_total", 0.0))
                cpu_overhead_samples = int(ckpt.get("cpu_overhead_samples", 0))
                if cpu_loop_ms_total <= 0.0 and (gpu_exec_ms_total > 0.0 or cpu_overhead_ms_total > 0.0):
                    # Backfill for checkpoints saved before CPU-loop counters existed.
                    cpu_loop_ms_total = max(0.0, gpu_exec_ms_total + cpu_overhead_ms_total)
                    cpu_loop_samples = max(gpu_exec_samples, cpu_overhead_samples, 0)
                # Chunk non-GPU metric; keep backward compatibility with prior key names.
                chunk_non_gpu_ms_total = float(
                    ckpt.get("chunk_non_gpu_ms_total", ckpt.get("loop_minus_gpu_ms_total", 0.0))
                )
                chunk_non_gpu_samples = int(
                    ckpt.get("chunk_non_gpu_samples", ckpt.get("loop_minus_gpu_samples", 0))
                )
                if chunk_non_gpu_ms_total < 0.0 or chunk_non_gpu_samples < 0:
                    chunk_non_gpu_ms_total = 0.0
                    chunk_non_gpu_samples = 0
                if dispatch_mode == "hybrid":
                    sched_i = int(ckpt.get("sched_i", 0))
                    sched_j = int(ckpt.get("sched_j", 1))
                    sched_k = int(ckpt.get("sched_k", 2))
            else:
                print(
                    f"Ignoring incompatible checkpoint (schema/config mismatch): {checkpoint_file}",
                    flush=True,
                )

    run_t0 = time.perf_counter() - elapsed_prev

    if dispatch_mode == "hybrid":
        ranges = [(idx, -1, -1) for idx in range(resume_chunk_idx, n_chunks + 1)]
    elif launch_mode == "single":
        if total_triples > np.iinfo(np.int32).max:
            raise ValueError(
                "launch_mode=single exceeds kernel n_tasks int32 limit for triples; "
                "use --launch-mode chunked."
            )
        ranges = [(1, 0, total_triples)]
        n_chunks = 1
    elif launch_mode == "chunked":
        ranges = [
            (idx, start, min(chunk_tasks, total_triples - start))
            for idx, start in enumerate(
                range(resume_task_start, total_triples, chunk_tasks),
                start=resume_chunk_idx,
            )
        ]
    else:
        raise ValueError(f"Unknown launch_mode: {launch_mode}")

    def _build_hybrid_chunk(max_tasks: int, floor_q_cur: int):
        nonlocal sched_i, sched_j, sched_k
        nonlocal scheduler_pruned_total, scheduler_pruned_tail_total, scheduler_pruned_singleton_total
        out_i = []
        out_j = []
        out_k = []
        additional_pruned = 0
        additional_tail = 0
        additional_singleton = 0
        n = n_allowed
        while len(out_i) < max_tasks and sched_i < n - 2:
            if sched_j >= n - 1:
                sched_i += 1
                sched_j = sched_i + 1
                sched_k = sched_j + 1
                continue
            if sched_k <= sched_j:
                sched_k = sched_j + 1
            if sched_k >= n:
                sched_j += 1
                sched_k = sched_j + 1
                continue

            h12_q = int(h12_q_pairs[pair_row_id(sched_i, sched_j, n_allowed)])
            thresh_q = floor_q_cur - h12_q
            # ent_q_sorted is descending. pass_end is first k with ent_q <= thresh.
            pass_end = int(np.searchsorted(neg_ent_q, -int(thresh_q), side="left"))
            if pass_end > n:
                pass_end = n
            if pass_end <= sched_k:
                # Remaining k suffix for this pair is fully pruned.
                additional_pruned += (n - sched_k)
                additional_tail += (n - sched_k)
                scheduler_pruned_total += (n - sched_k)
                scheduler_pruned_tail_total += (n - sched_k)
                sched_j += 1
                sched_k = sched_j + 1
                continue

            take = min(pass_end - sched_k, max_tasks - len(out_i))
            if take > 0:
                ks = np.arange(sched_k, sched_k + take, dtype=np.int32)
                # Symmetric entangled pruning:
                # keep only if all three upper bounds can still beat floor.
                row_base_i = sched_i * n_allowed - (sched_i * (sched_i + 1)) // 2
                row_base_j = sched_j * n_allowed - (sched_j * (sched_j + 1)) // 2
                idx_13 = row_base_i + (ks - sched_i - 1)
                idx_23 = row_base_j + (ks - sched_j - 1)
                h13_q = h12_q_pairs[idx_13]
                h23_q = h12_q_pairs[idx_23]
                keep_mask = (
                    (h13_q.astype(np.int64) + int(ent_q_sorted[sched_j]) > floor_q_cur) &
                    (h23_q.astype(np.int64) + int(ent_q_sorted[sched_i]) > floor_q_cur)
                )
                kept_ks = ks[keep_mask]
                dropped = int(ks.size - kept_ks.size)
                if dropped > 0:
                    additional_pruned += dropped
                    additional_singleton += dropped
                    scheduler_pruned_total += dropped
                    scheduler_pruned_singleton_total += dropped
                if kept_ks.size > 0:
                    out_i.extend([sched_i] * int(kept_ks.size))
                    out_j.extend([sched_j] * int(kept_ks.size))
                    out_k.extend(kept_ks.tolist())
                sched_k += int(take)

            if sched_k >= pass_end:
                tail = n - pass_end
                if tail > 0:
                    additional_pruned += tail
                    additional_tail += tail
                    scheduler_pruned_total += tail
                    scheduler_pruned_tail_total += tail
                sched_j += 1
                sched_k = sched_j + 1

        return (
            np.asarray(out_i, dtype=np.int32),
            np.asarray(out_j, dtype=np.int32),
            np.asarray(out_k, dtype=np.int32),
            additional_pruned,
            additional_tail,
            additional_singleton,
        )

    processed_tasks_running = resume_task_start
    pending_chunk = None
    prev_chunk_return_ts = None
    last_dispatch_gap_ms = 0.0
    if dispatch_mode == "hybrid":
        ti0, tj0, tk0, add0, tail0, sing0 = _build_hybrid_chunk(chunk_tasks, current_floor_q)
        if ti0.size > 0:
            pending_chunk = (ti0, tj0, tk0, add0, tail0, sing0)
    for chunk_idx, task_start, n_tasks in ranges:
        chunk_t0 = time.perf_counter()
        out_count.zero_()
        out_overflow_flag.zero_()
        out_overflow_dropped.zero_()
        out_full_prune_blocks.zero_()
        out_active_threads.zero_()
        out_bound_pass_threads.zero_()

        additional_pruned = 0
        additional_tail = 0
        additional_singleton = 0
        chunk_considered = 0
        queue_wait_line = None
        cpu_build_ms_last = 0.0
        cpu_build_overhang_ms_last = 0.0
        chunk_non_gpu_ms_last = 0.0
        gpu_chunk_ms_last = 0.0
        gpu_ev_start = None
        gpu_ev_end = None
        if dispatch_mode == "hybrid":
            if pending_chunk is None:
                break
            ti, tj, tk, additional_pruned, additional_tail, additional_singleton = pending_chunk
            n_tasks_cur = int(ti.size)
            if n_tasks_cur <= 0:
                break
            d_ti = torch.from_numpy(ti).to(device="cuda", dtype=torch.int32)
            d_tj = torch.from_numpy(tj).to(device="cuda", dtype=torch.int32)
            d_tk = torch.from_numpy(tk).to(device="cuda", dtype=torch.int32)
            dispatch_floor_q = current_floor_q
            floor_q.fill_(dispatch_floor_q)
            gpu_ev_start = torch.cuda.Event(enable_timing=True)
            gpu_ev_end = torch.cuda.Event(enable_timing=True)
            gpu_ev_start.record()
            lib.launch_three_guess_search_indexed_codex(
                matrix_u8_gpu.data_ptr(),
                n_allowed,
                n_answers,
                sorted_indices_gpu.data_ptr(),
                d_ti.data_ptr(),
                d_tj.data_ptr(),
                d_tk.data_ptr(),
                n_tasks_cur,
                floor_q.data_ptr(),
                out_count.data_ptr(),
                append_capacity,
                out_overflow_flag.data_ptr(),
                out_overflow_dropped.data_ptr(),
                out_full_prune_blocks.data_ptr(),
                out_active_threads.data_ptr(),
                out_bound_pass_threads.data_ptr(),
                out_entropy_q.data_ptr(),
                out_i.data_ptr(),
                out_j.data_ptr(),
                out_k.data_ptr(),
            )
            gpu_ev_end.record()
            dispatch_done_ts = time.perf_counter()
            if prev_chunk_return_ts is not None:
                last_dispatch_gap_ms = (dispatch_done_ts - prev_chunk_return_ts) * 1000.0
                dispatch_latency_ms_total += max(0.0, last_dispatch_gap_ms)
                dispatch_latency_samples += 1
            processed_tasks_running += n_tasks_cur
            if floor_source_mode == "t-1":
                # Build next chunk on stale floor while GPU executes current chunk.
                build_t0 = time.perf_counter()
                ti_next, tj_next, tk_next, add_next, tail_next, sing_next = _build_hybrid_chunk(chunk_tasks, dispatch_floor_q)
                cpu_build_ms_last = (time.perf_counter() - build_t0) * 1000.0
                cpu_build_ms_total += max(0.0, cpu_build_ms_last)
                cpu_build_samples += 1
                pending_chunk = (
                    (ti_next, tj_next, tk_next, add_next, tail_next, sing_next)
                    if ti_next.size > 0 else None
                )
            else:
                # floor(t): must wait for current chunk completion and floor update.
                pending_chunk = None
            n_tasks = n_tasks_cur
            task_start = processed_tasks_running - n_tasks_cur
            chunk_considered = n_tasks_cur + additional_pruned
        else:
            floor_q.fill_(current_floor_q)
            lib.launch_three_guess_search_codex(
                matrix_u8_gpu.data_ptr(),
                n_allowed,
                n_answers,
                sorted_indices_gpu.data_ptr(),
                sorted_entropies_gpu.data_ptr(),
                int(task_start),
                n_tasks,
                floor_q.data_ptr(),
                out_count.data_ptr(),
                append_capacity,
                out_overflow_flag.data_ptr(),
                out_overflow_dropped.data_ptr(),
                out_full_prune_blocks.data_ptr(),
                out_active_threads.data_ptr(),
                out_bound_pass_threads.data_ptr(),
                out_entropy_q.data_ptr(),
                out_i.data_ptr(),
                out_j.data_ptr(),
                out_k.data_ptr(),
            )
            processed_tasks_running = int(min(task_start + n_tasks, total_triples))
        torch.cuda.synchronize()
        if dispatch_mode == "hybrid" and gpu_ev_start is not None and gpu_ev_end is not None:
            gpu_chunk_ms_last = float(gpu_ev_start.elapsed_time(gpu_ev_end))
            if floor_source_mode == "t-1":
                cpu_build_overhang_ms_last = max(0.0, cpu_build_ms_last - gpu_chunk_ms_last)
                cpu_build_overhang_ms_total += cpu_build_overhang_ms_last
        if dispatch_mode == "hybrid":
            curr_chunk_return_ts = time.perf_counter()
            if prev_chunk_return_ts is not None and gpu_chunk_ms_last > 0.0:
                chunk_interval_ms = max(0.0, (curr_chunk_return_ts - prev_chunk_return_ts) * 1000.0)
                chunk_non_gpu_ms_last = max(0.0, chunk_interval_ms - gpu_chunk_ms_last)
                chunk_non_gpu_ms_total += chunk_non_gpu_ms_last
                chunk_non_gpu_samples += 1
            prev_chunk_return_ts = curr_chunk_return_ts

        overflow_flag = int(out_overflow_flag.item())
        overflow_dropped = int(out_overflow_dropped.item())
        if overflow_flag != 0:
            raise RuntimeError(
                "GPU append buffer overflow (triples): "
                f"capacity={append_capacity:,}, dropped={overflow_dropped:,}, "
                f"chunk_idx={chunk_idx}, task_start={task_start:,}, n_tasks={n_tasks:,}. "
                "Increase --append-capacity or reduce --chunk-tasks."
            )

        candidate_n = int(out_count.item())
        full_prune_blocks = int(out_full_prune_blocks.item())
        active_threads = int(out_active_threads.item())
        bound_pass_threads = int(out_bound_pass_threads.item())
        if dispatch_mode != "hybrid":
            chunk_considered = active_threads
        if candidate_n > append_capacity:
            raise RuntimeError(
                f"Kernel produced invalid candidate count {candidate_n} > append_capacity {append_capacity}"
            )
        if candidate_n > n_tasks and launch_mode == "chunked":
            raise RuntimeError(
                f"Kernel produced invalid candidate count {candidate_n} > n_tasks {n_tasks}"
            )

        if candidate_n > 0:
            last_elevated_chunk_idx = int(chunk_idx)
            q = out_entropy_q[:candidate_n].cpu().numpy()
            ii = out_i[:candidate_n].cpu().numpy()
            jj = out_j[:candidate_n].cpu().numpy()
            kk = out_k[:candidate_n].cpu().numpy()
            for entropy_q_val, i_idx, j_idx, k_idx in zip(q, ii, jj, kk):
                entry = (int(entropy_q_val), int(i_idx), int(j_idx), int(k_idx))
                if len(top_heap) < top_k:
                    heapq.heappush(top_heap, entry)
                elif entry[0] > top_heap[0][0]:
                    heapq.heapreplace(top_heap, entry)

            if len(top_heap) == top_k:
                current_floor_q = top_heap[0][0]
            chunk_best_q = max((entry[0] for entry in top_heap), default=-1)
            if chunk_best_q > global_best_q:
                global_best_q = int(chunk_best_q)
                last_global_max_chunk_idx = int(chunk_idx)

        if dispatch_mode == "hybrid" and floor_source_mode == "t":
            build_t0 = time.perf_counter()
            ti_next, tj_next, tk_next, add_next, tail_next, sing_next = _build_hybrid_chunk(chunk_tasks, current_floor_q)
            cpu_build_ms_last = (time.perf_counter() - build_t0) * 1000.0
            cpu_build_ms_total += max(0.0, cpu_build_ms_last)
            cpu_build_overhang_ms_last = max(0.0, cpu_build_ms_last)  # floor(t) build is fully serialized.
            cpu_build_overhang_ms_total += cpu_build_overhang_ms_last
            cpu_build_samples += 1
            pending_chunk = (
                (ti_next, tj_next, tk_next, add_next, tail_next, sing_next)
                if ti_next.size > 0 else None
            )

        blocks_in_chunk = (n_tasks + BLOCK_THREADS - 1) // BLOCK_THREADS
        full_prune_pct = (
            (100.0 * full_prune_blocks / blocks_in_chunk) if blocks_in_chunk > 0 else 100.0
        )
        prune_thread_pct = (
            100.0 * (1.0 - (bound_pass_threads / active_threads))
            if active_threads > 0 else 100.0
        )
        pruned_threads = max(0, active_threads - bound_pass_threads)
        if dispatch_mode == "hybrid":
            pruned_threads = additional_pruned
            active_threads = n_tasks + additional_pruned
            prune_thread_pct = (100.0 * additional_pruned / active_threads) if active_threads > 0 else 0.0
        else:
            scheduler_pruned_total += 0
        cumulative_pruned_threads += pruned_threads
        cumulative_full_prune_blocks += full_prune_blocks
        cumulative_blocks_considered += blocks_in_chunk
        cumulative_active_threads += active_threads
        cumulative_considered += chunk_considered
        if dispatch_mode == "hybrid" and dispatch_latency_samples > 0:
            queue_wait_line = None
        should_print = (
            progress_mode == "dashboard" or
            (candidate_n > 0) or
            (full_prune_blocks < blocks_in_chunk)
        )
        if should_print:
            if dispatch_mode == "hybrid":
                covered = min(total_triples, processed_tasks_running + scheduler_pruned_total)
                remaining = max(0, total_triples - covered)
                n_chunks_display = max(
                    chunk_idx,
                    chunk_idx + (remaining + max(1, chunk_tasks) - 1) // max(1, chunk_tasks),
                )
            else:
                covered = min(task_start + n_tasks, total_triples)
                n_chunks_display = n_chunks
            pct = 100.0 * covered / total_triples
            floor_bits = (current_floor_q / 1_000_000.0) if current_floor_q >= 0 else -1.0
            best_q = max((entry[0] for entry in top_heap), default=-1)
            best_bits = (best_q / 1_000_000.0) if best_q >= 0 else -1.0
            chunk_ms = (time.perf_counter() - chunk_t0) * 1000.0
            if dispatch_mode == "hybrid" and gpu_chunk_ms_last > 0.0:
                cpu_loop_ms_total += chunk_ms
                cpu_loop_samples += 1
                gpu_exec_ms_total += gpu_chunk_ms_last
                gpu_exec_samples += 1
                cpu_overhead_ms_last = max(0.0, chunk_ms - gpu_chunk_ms_last)
                cpu_overhead_ms_total += cpu_overhead_ms_last
                cpu_overhead_samples += 1
            if baseline_chunk_ms is None:
                baseline_chunk_ms = chunk_ms
            processed_tasks = int(covered)
            if dispatch_mode == "hybrid" and gpu_exec_samples > 0:
                cpu_loop_current_ms = chunk_ms
                cpu_loop_total_s = cpu_loop_ms_total / 1000.0
                cpu_loop_avg_ms = cpu_loop_ms_total / max(cpu_loop_samples, 1)

                gpu_exec_current_ms = max(0.0, gpu_chunk_ms_last)
                gpu_exec_total_s = gpu_exec_ms_total / 1000.0
                gpu_exec_avg_ms = gpu_exec_ms_total / max(gpu_exec_samples, 1)

                cpu_build_current_ms = max(0.0, cpu_build_ms_last)
                cpu_build_total_s = cpu_build_ms_total / 1000.0
                cpu_build_avg_ms = cpu_build_ms_total / max(cpu_build_samples, 1) if cpu_build_samples > 0 else 0.0

                cpu_overhead_current_ms = max(0.0, cpu_loop_current_ms - gpu_exec_current_ms)
                cpu_overhead_total_s = cpu_overhead_ms_total / 1000.0
                cpu_overhead_avg_ms = (
                    cpu_overhead_ms_total / max(cpu_overhead_samples, 1)
                    if cpu_overhead_samples > 0 else 0.0
                )

                queue_wait_line = (
                    f"cur {cpu_loop_current_ms:.1f} ms | tot {_fmt_adaptive_duration_labeled(cpu_loop_total_s)} | "
                    f"avg {cpu_loop_avg_ms:.1f} ms"
                    f"\ncur {gpu_exec_current_ms:.1f} ms | tot {_fmt_adaptive_duration_labeled(gpu_exec_total_s)} | "
                    f"avg {gpu_exec_avg_ms:.1f} ms"
                    f"\ncur {cpu_build_current_ms:.1f} ms | tot {_fmt_adaptive_duration_labeled(cpu_build_total_s)} | "
                    f"avg {cpu_build_avg_ms:.1f} ms"
                    f"\ncur {cpu_overhead_current_ms:.1f} ms | tot {_fmt_adaptive_duration_labeled(cpu_overhead_total_s)} | "
                    f"avg {cpu_overhead_avg_ms:.1f} ms"
                )
            _emit_progress(
                progress_mode=progress_mode,
                title=(
                    "GPU 3-word entropy search (hybrid entangled dispatch)"
                    if dispatch_mode == "hybrid" else
                    "GPU 3-word entropy search"
                ),
                chunk_idx=chunk_idx,
                n_chunks=n_chunks_display,
                n_chunks_original=(n_chunks if dispatch_mode == "hybrid" else None),
                pct=pct,
                floor_bits=floor_bits,
                best_bits=best_bits,
                top_k=top_k,
                candidate_n=candidate_n,
                full_prune_blocks=full_prune_blocks,
                blocks_in_chunk=blocks_in_chunk,
                full_prune_pct=full_prune_pct,
                cumulative_full_prune_blocks=cumulative_full_prune_blocks,
                cumulative_blocks_considered=cumulative_blocks_considered,
                pruned_threads=pruned_threads,
                active_threads=active_threads,
                prune_thread_pct=prune_thread_pct,
                cumulative_active_threads=cumulative_active_threads,
                chunk_ms=chunk_ms,
                start_time=run_t0,
                processed_tasks=processed_tasks,
                total_tasks=total_triples,
                baseline_chunk_ms=baseline_chunk_ms,
                cumulative_pruned_threads=cumulative_pruned_threads,
                precompute_note=precompute_note,
                prune_count_line=None,
                queue_wait_line=queue_wait_line,
                show_pruning_line=(dispatch_mode != "hybrid"),
                show_prune_n3_line=(dispatch_mode == "hybrid"),
                prune_n3_label=("Pre-dispatch N^3" if dispatch_mode == "hybrid" else "Prune N^3"),
                chunking_label=("Pruned" if dispatch_mode == "hybrid" else "Geometric"),
                pruning_floor_mode=(floor_source_mode if dispatch_mode == "hybrid" else None),
                prune_current_considered=(
                    chunk_considered if dispatch_mode == "hybrid" else active_threads
                ),
                cumulative_prune_considered=(
                    cumulative_considered if dispatch_mode == "hybrid" else cumulative_active_threads
                ),
                prune_tail_current=(additional_tail if dispatch_mode == "hybrid" else None),
                prune_tail_cumulative=(scheduler_pruned_tail_total if dispatch_mode == "hybrid" else None),
                prune_singleton_current=(additional_singleton if dispatch_mode == "hybrid" else None),
                prune_singleton_cumulative=(
                    scheduler_pruned_singleton_total if dispatch_mode == "hybrid" else None
                ),
                last_elevated_chunk_idx=last_elevated_chunk_idx,
                last_global_max_chunk_idx=last_global_max_chunk_idx,
            )

        if checkpoint_file and launch_mode == "chunked":
            if dispatch_mode == "hybrid":
                next_task_start = int(processed_tasks_running)
            else:
                next_task_start = int(task_start + n_tasks)
            state = {
                "version": CHECKPOINT_SCHEMA_VERSION,
                "kind": "three_guess",
                "dispatch_mode": dispatch_mode,
                "floor_source_mode": floor_source_mode,
                "total_tasks": int(total_triples),
                "chunk_tasks": int(chunk_tasks),
                "launch_mode": launch_mode,
                "top_k": int(top_k),
                "current_floor_q": int(current_floor_q),
                "global_best_q": int(global_best_q),
                "top_heap": [list(x) for x in top_heap],
                "last_elevated_chunk_idx": (
                    int(last_elevated_chunk_idx) if last_elevated_chunk_idx is not None else None
                ),
                "last_global_max_chunk_idx": (
                    int(last_global_max_chunk_idx) if last_global_max_chunk_idx is not None else None
                ),
                "next_task_start": next_task_start,
                "next_chunk_idx": int(chunk_idx + 1),
                "elapsed_s": float(max(0.0, time.perf_counter() - run_t0)),
                "baseline_chunk_ms": baseline_chunk_ms,
                "cumulative_pruned_threads": int(cumulative_pruned_threads),
                "cumulative_full_prune_blocks": int(cumulative_full_prune_blocks),
                "cumulative_blocks_considered": int(cumulative_blocks_considered),
                "cumulative_active_threads": int(cumulative_active_threads),
                "scheduler_pruned_total": int(scheduler_pruned_total),
                "scheduler_pruned_tail_total": int(scheduler_pruned_tail_total),
                "scheduler_pruned_singleton_total": int(scheduler_pruned_singleton_total),
                "cumulative_considered": int(cumulative_considered),
                "dispatch_latency_ms_total": float(dispatch_latency_ms_total),
                "dispatch_latency_samples": int(dispatch_latency_samples),
                "cpu_loop_ms_total": float(cpu_loop_ms_total),
                "cpu_loop_samples": int(cpu_loop_samples),
                "cpu_build_ms_total": float(cpu_build_ms_total),
                "cpu_build_overhang_ms_total": float(cpu_build_overhang_ms_total),
                "cpu_build_samples": int(cpu_build_samples),
                "gpu_exec_ms_total": float(gpu_exec_ms_total),
                "gpu_exec_samples": int(gpu_exec_samples),
                "cpu_overhead_ms_total": float(cpu_overhead_ms_total),
                "cpu_overhead_samples": int(cpu_overhead_samples),
                "chunk_non_gpu_ms_total": float(chunk_non_gpu_ms_total),
                "chunk_non_gpu_samples": int(chunk_non_gpu_samples),
                "sched_i": int(sched_i) if dispatch_mode == "hybrid" else None,
                "sched_j": int(sched_j) if dispatch_mode == "hybrid" else None,
                "sched_k": int(sched_k) if dispatch_mode == "hybrid" else None,
                "completed": bool(next_task_start >= total_triples),
            }
            _save_checkpoint(checkpoint_file, state)

    if dispatch_mode == "hybrid" and dispatch_latency_samples > 0 and progress_mode != "off":
        elapsed_s = max(0.0, time.perf_counter() - run_t0)
        dispatch_latency_pct = 100.0 * dispatch_latency_ms_total / max(elapsed_s * 1000.0, 1.0)
        print(
            f"Dispatch latency summary [floor({floor_source_mode})]: "
            f"{_fmt_dhhmm(dispatch_latency_ms_total / 1000.0)} "
            f"({dispatch_latency_pct:.2f}% runtime)",
            flush=True,
        )

    results = []
    for entropy_q_val, i_idx, j_idx, k_idx in top_heap:
        results.append(
            (
                float(entropy_q_val) / 1_000_000.0,
                int(i_idx),
                int(j_idx),
                int(k_idx),
            )
        )

    results.sort(reverse=True)
    elapsed_s_final = max(0.0, time.perf_counter() - run_t0)
    prune_considered_final = (
        int(cumulative_considered) if dispatch_mode == "hybrid" else int(cumulative_active_threads)
    )
    prune_pct_final = 100.0 * cumulative_pruned_threads / max(prune_considered_final, 1)
    dispatch_latency_pct_final = 100.0 * dispatch_latency_ms_total / max(elapsed_s_final * 1000.0, 1.0)
    stats = {
        "dispatch_mode": dispatch_mode,
        "floor_source_mode": floor_source_mode,
        "total_triples": int(total_triples),
        "floor_bits": (current_floor_q / 1_000_000.0) if current_floor_q >= 0 else -1.0,
        "elapsed_s": float(elapsed_s_final),
        "cumulative_pruned_threads": int(cumulative_pruned_threads),
        "cumulative_considered": int(prune_considered_final),
        "cumulative_prune_pct": float(prune_pct_final),
        "cumulative_full_prune_blocks": int(cumulative_full_prune_blocks),
        "cumulative_blocks_considered": int(cumulative_blocks_considered),
        "dispatch_latency_ms_total": float(dispatch_latency_ms_total),
        "dispatch_latency_pct_runtime": float(dispatch_latency_pct_final),
    }
    if return_stats:
        return results, stats
    return results
