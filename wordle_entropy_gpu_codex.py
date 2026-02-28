#!/usr/bin/env python3
"""
wordle_entropy_gpu_codex.py

GPU-accelerated Wordle entropy optimizer using CUDA via PyTorch.

This script finds optimal non-adaptive guess sequences using GPU parallelization.
Supports both 2-word and 3-word searches.
"""

import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch

from src.words import load_words
from src.patterns import load_or_build_matrix
from src.entropy import entropy_from_counts
from src.entropy_gpu import (
    check_gpu_available,
    get_gpu_info,
)
from src.cuda.cuda_entropy_codex import two_guess_search_cuda_codex
from src.cuda.cuda_entropy_codex import three_guess_search_cuda_codex

# Constants
TOP_PAIRS = 50
TOP_TRIPLES = 50


def single_guess_entropy(matrix_row):
    """Compute entropy of one guess across all answers (CPU)."""
    counts = np.bincount(matrix_row)
    return entropy_from_counts(counts)


def run_two_guess_gpu(
    answers,
    allowed,
    matrix,
    verbose=False,
    pair_order="antidiag",
    launch_mode="chunked",
    append_capacity=2_000_000,
    chunk_tasks=2_000_000,
    progress_mode="dashboard",
    checkpoint_file=None,
    resume=True,
):
    """
    GPU-accelerated 2-word search.

    Strategy:
    1. Compute single entropies on CPU (fast)
    2. Transfer matrix + entropies to GPU (one-time)
    3. Process pairs in batches on GPU
    4. Each batch:
       - Generate (i,j) pairs to evaluate
       - Prune using h1 + h2 bound
       - Compute h12 on GPU
       - Update heap
    5. Return top 50 pairs
    """
    answer_set = set(answers)
    n_allowed = len(allowed)

    print("Computing single guess entropies (CPU)...")
    start = time.time()
    single_entropies = np.array(
        [single_guess_entropy(matrix[i]) for i in range(n_allowed)]
    )
    elapsed = time.time() - start
    print(f"  Computed {n_allowed:,} entropies in {elapsed:.2f}s")

    best_single_entropy = float(np.max(single_entropies))

    # Sort by entropy (descending)
    sorted_indices = np.argsort(single_entropies)[::-1].copy()
    sorted_entropies = single_entropies[sorted_indices].copy()

    print("\nTransferring data to GPU...")
    gpu_start = time.time()

    # Keep matrix as uint8 on device to reduce memory traffic.
    gpu_matrix = torch.tensor(matrix.astype(np.uint8), device='cuda', dtype=torch.uint8)
    gpu_sorted_indices = torch.tensor(sorted_indices, device='cuda', dtype=torch.int32)
    gpu_sorted_entropies = torch.tensor(sorted_entropies, device='cuda', dtype=torch.float32)

    elapsed = time.time() - gpu_start
    matrix_size_mb = gpu_matrix.element_size() * gpu_matrix.nelement() / 1e6
    print(f"  Transferred {matrix_size_mb:.1f} MB to GPU in {elapsed:.2f}s")

    total_pairs = n_allowed * (n_allowed - 1) // 2
    print(f"\nStarting GPU 2-word search over {total_pairs:,} pairs...")
    print(f"Target: top {TOP_PAIRS} pairs\n")

    search_start = time.time()
    best_pairs = two_guess_search_cuda_codex(
        gpu_matrix,
        gpu_sorted_indices,
        gpu_sorted_entropies,
        top_k=TOP_PAIRS,
        chunk_tasks=chunk_tasks,
        pair_order=pair_order,
        launch_mode=launch_mode,
        append_capacity=append_capacity,
        progress_mode=progress_mode,
        checkpoint_file=checkpoint_file,
        resume=resume,
    )

    total_elapsed = time.time() - search_start
    print(f"\nSearch completed in {total_elapsed:.2f}s")
    print(f"Scanned task space: {total_pairs:,} pairs")
    print(f"Average rate: {total_pairs / total_elapsed:.0f} pairs/s")

    # Display results
    print("\nTop two-guess pairs (non-adaptive, GPU-accelerated):")
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


def run_three_guess_gpu(
    answers,
    allowed,
    matrix,
    verbose=False,
    launch_mode="chunked",
    append_capacity=2_000_000,
    chunk_tasks=2_000_000,
    progress_mode="dashboard",
    checkpoint_file=None,
    final_summary_file=None,
    resume=True,
    dispatch_mode="geometric",
    h12_cache_base=".gpu_h12_cache",
    floor_source_mode="t-1",
):
    """GPU-accelerated 3-word search in row order (i<j<k)."""
    answer_set = set(answers)
    n_allowed = len(allowed)

    print("Computing single guess entropies (CPU)...")
    start = time.time()
    single_entropies = np.array(
        [single_guess_entropy(matrix[i]) for i in range(n_allowed)]
    )
    elapsed = time.time() - start
    print(f"  Computed {n_allowed:,} entropies in {elapsed:.2f}s")

    best_single_entropy = float(np.max(single_entropies))

    sorted_indices = np.argsort(single_entropies)[::-1].copy()
    sorted_entropies = single_entropies[sorted_indices].copy()

    print("\nTransferring data to GPU...")
    gpu_start = time.time()
    gpu_matrix = torch.tensor(matrix.astype(np.uint8), device='cuda', dtype=torch.uint8)
    gpu_sorted_indices = torch.tensor(sorted_indices, device='cuda', dtype=torch.int32)
    gpu_sorted_entropies = torch.tensor(sorted_entropies, device='cuda', dtype=torch.float32)
    elapsed = time.time() - gpu_start
    matrix_size_mb = gpu_matrix.element_size() * gpu_matrix.nelement() / 1e6
    print(f"  Transferred {matrix_size_mb:.1f} MB to GPU in {elapsed:.2f}s")

    total_triples = n_allowed * (n_allowed - 1) * (n_allowed - 2) // 6
    print(f"\nStarting GPU 3-word search over {total_triples:,} triples...")
    print(f"Target: top {TOP_TRIPLES} triples\n")

    search_start = time.time()
    best_triples, run_stats = three_guess_search_cuda_codex(
        gpu_matrix,
        gpu_sorted_indices,
        gpu_sorted_entropies,
        sorted_entropies_cpu=sorted_entropies.astype(np.float32, copy=False),
        top_k=TOP_TRIPLES,
        chunk_tasks=chunk_tasks,
        launch_mode=launch_mode,
        append_capacity=append_capacity,
        progress_mode=progress_mode,
        checkpoint_file=checkpoint_file,
        resume=resume,
        dispatch_mode=dispatch_mode,
        h12_cache_base=h12_cache_base,
        floor_source_mode=floor_source_mode,
        return_stats=True,
    )
    total_elapsed = time.time() - search_start
    print(f"\nSearch completed in {total_elapsed:.2f}s")
    print(f"Scanned task space: {total_triples:,} triples")
    print(f"Average rate: {total_triples / total_elapsed:.0f} triples/s")
    if run_stats is not None:
        print(
            "Final pre-dispatch prune: "
            f"{run_stats['cumulative_pruned_threads']:,}/{run_stats['cumulative_considered']:,} "
            f"({run_stats['cumulative_prune_pct']:.3f}%)"
        )
        print(f"Final floor: {run_stats['floor_bits']:.4f} bits")

        summary = {
            "mode": 3,
            "dispatch_mode": dispatch_mode,
            "floor_source_mode": floor_source_mode,
            "answers_n": len(answers),
            "allowed_n": len(allowed),
            "total_triples": int(total_triples),
            "search_elapsed_s": float(total_elapsed),
            "average_rate_triples_s": float(total_triples / max(total_elapsed, 1e-9)),
            "checkpoint_file": checkpoint_file,
            "run_stats": run_stats,
        }
        summary_path = final_summary_file
        if summary_path is None and checkpoint_file:
            summary_path = f"{checkpoint_file}.final.json"
        if summary_path is not None:
            Path(summary_path).write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(f"Final summary JSON: {summary_path}")

    print("\nTop three-guess triples (non-adaptive, GPU-accelerated):")
    if verbose:
        print(
            "Legend: word1 + word2 + word3 [flags]: H123 bits (H1, H2, H3) | "
            "Cost: (H_best_single - H1) bits"
        )
    else:
        print("Legend: word1 + word2 + word3 [flags]: H123 bits")
    print(
        "flags: [+++] all answers, [++-] first two only, [+-+] first/third, "
        "[+--] first only, [-++] last two only, [-+-] second only, [--+] third only, [---] none"
    )

    for h123, i, j, k in best_triples:
        w1 = allowed[i]
        w2 = allowed[j]
        w3 = allowed[k]
        f1 = "+" if w1 in answer_set else "-"
        f2 = "+" if w2 in answer_set else "-"
        f3 = "+" if w3 in answer_set else "-"
        if verbose:
            h1 = single_entropies[i]
            h2 = single_entropies[j]
            h3 = single_entropies[k]
            cost = best_single_entropy - h1
            print(
                f"{w1} + {w2} + {w3} [{f1}{f2}{f3}]: {h123:.4f} bits "
                f"({h1:.4f}, {h2:.4f}, {h3:.4f}) | Cost: {cost:.4f} bits"
            )
        else:
            print(f"{w1} + {w2} + {w3} [{f1}{f2}{f3}]: {h123:.4f} bits")


def main():
    parser = argparse.ArgumentParser(
        description="GPU-accelerated Wordle entropy optimizer"
    )
    parser.add_argument(
        "--mode",
        choices=["2", "3"],
        default="2",
        help="Search mode: 2 = two-word search, 3 = three-word search (default: 2)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed entropy breakdown",
    )
    parser.add_argument(
        "--pair-order",
        choices=["antidiag", "row"],
        default="antidiag",
        help="Pair traversal order for mode 2 inside CUDA kernel (default: antidiag).",
    )
    parser.add_argument(
        "--launch-mode",
        choices=["chunked", "single"],
        default="chunked",
        help="Kernel launch mode: chunked or single (host-merged).",
    )
    parser.add_argument(
        "--append-capacity",
        type=int,
        default=2_000_000,
        help="GPU append-buffer capacity for staged candidates per launch.",
    )
    parser.add_argument(
        "--chunk-tasks",
        type=int,
        default=2_000_000,
        help="Number of tasks per chunk when --launch-mode chunked.",
    )
    parser.add_argument(
        "--progress",
        choices=["dashboard", "log", "off"],
        default="dashboard",
        help="Progress output style for GPU search.",
    )
    parser.add_argument(
        "--checkpoint-file",
        type=str,
        default=None,
        help="Optional checkpoint JSON path for chunked GPU mode.",
    )
    parser.add_argument(
        "--resume",
        choices=["auto", "new"],
        default="auto",
        help="Checkpoint behavior: auto=resume if compatible, new=ignore prior checkpoint.",
    )
    parser.add_argument(
        "--dispatch-mode",
        choices=["geometric", "hybrid"],
        default="hybrid",
        help="Mode 3 dispatch: geometric (old) or hybrid entangled-prune tasklist.",
    )
    parser.add_argument(
        "--floor-source-mode",
        choices=["t", "t-1"],
        default="t-1",
        help="Hybrid mode floor used to build next chunk: t=updated floor, t-1=previous floor.",
    )
    parser.add_argument(
        "--h12-cache-base",
        type=str,
        default=".gpu_h12_cache",
        help="Base path (without extension) for GPU-computed H12 cache files.",
    )
    parser.add_argument(
        "--answers-file",
        type=str,
        default=None,
        help="Path to newline-separated answers word list (default: data/answers.txt).",
    )
    parser.add_argument(
        "--allowed-file",
        type=str,
        default=None,
        help="Path to newline-separated allowed word list (default: data/allowed.txt).",
    )
    parser.add_argument(
        "--final-summary-file",
        type=str,
        default=None,
        help="Optional JSON file for end-of-run summary metrics.",
    )

    args = parser.parse_args()

    # Default checkpoint path so restart works without extra flags.
    checkpoint_file = args.checkpoint_file
    if checkpoint_file is None and args.launch_mode == "chunked":
        if args.mode == "2":
            checkpoint_file = f".gpu_codex_mode2_{args.pair_order}.checkpoint.json"
        else:
            if args.dispatch_mode == "hybrid":
                checkpoint_file = (
                    f".gpu_codex_mode3_hybrid_{args.floor_source_mode}.checkpoint.json"
                )
            else:
                checkpoint_file = ".gpu_codex_mode3_geometric.checkpoint.json"
    if checkpoint_file is not None:
        checkpoint_file = str(Path(checkpoint_file))

    # Check GPU availability
    print("Checking GPU availability...")
    try:
        check_gpu_available()
        # Get GPU info
        gpu_info = get_gpu_info()
        print(f"GPU detected: {gpu_info['name']}")
        print(f"Compute capability: {gpu_info['compute_capability']}")
        print(f"GPU memory: {gpu_info['memory_total_gb']:.1f} GB total, {gpu_info['memory_free_gb']:.1f} GB free\n")
    except RuntimeError as e:
        print(f"Error: {e}")
        print("\nGPU acceleration requires:")
        print("  1. NVIDIA GPU with CUDA support")
        print("  2. PyTorch with CUDA installed: pip install torch")
        return 1

    # Load word lists
    print("Loading word lists...")
    answers, allowed = load_words(
        answers_path=args.answers_file,
        allowed_path=args.allowed_file,
    )
    print(f"  Answers: {len(answers):,}")
    print(f"  Allowed: {len(allowed):,}\n")

    # Load or build pattern matrix
    matrix = load_or_build_matrix(
        allowed,
        answers,
        answers_path=args.answers_file,
        allowed_path=args.allowed_file,
    )
    print(f"Matrix shape: {matrix.shape}\n")

    # Run search
    if args.mode == "2":
        resume_enabled = args.resume == "auto"
        print("Using custom CUDA kernel for entropy computation\n")
        if checkpoint_file is not None and args.launch_mode == "chunked":
            action = "resume/continue" if resume_enabled else "start new"
            print(f"Checkpoint: {checkpoint_file} ({action})")
        run_two_guess_gpu(
            answers,
            allowed,
            matrix,
            verbose=args.verbose,
            pair_order=args.pair_order,
            launch_mode=args.launch_mode,
            append_capacity=args.append_capacity,
            chunk_tasks=args.chunk_tasks,
            progress_mode=args.progress,
            checkpoint_file=checkpoint_file,
            resume=resume_enabled,
        )
    elif args.mode == "3":
        resume_enabled = args.resume == "auto"
        print("Using custom CUDA kernel for entropy computation\n")
        if args.pair_order != "row":
            print("Note: mode 3 currently supports row order only; ignoring --pair-order.")
        if checkpoint_file is not None and args.launch_mode == "chunked":
            action = "resume/continue" if resume_enabled else "start new"
            print(f"Checkpoint: {checkpoint_file} ({action})")
        run_three_guess_gpu(
            answers,
            allowed,
            matrix,
            verbose=args.verbose,
            launch_mode=args.launch_mode,
            append_capacity=args.append_capacity,
            chunk_tasks=args.chunk_tasks,
            progress_mode=args.progress,
            checkpoint_file=checkpoint_file,
            final_summary_file=args.final_summary_file,
            resume=resume_enabled,
            dispatch_mode=args.dispatch_mode,
            h12_cache_base=args.h12_cache_base,
            floor_source_mode=args.floor_source_mode,
        )
    else:
        print(f"Mode {args.mode} not yet implemented")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
