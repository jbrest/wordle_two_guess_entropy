# Changelog

## Current (2026-02-28)

This changelog now tracks the current maintained implementation line only.

### Scope

- Primary CPU CLI: `wordle_entropy.py`
- Primary GPU CLI: `wordle_entropy_gpu_codex.py`
- Primary CUDA wrapper/kernel: `src/cuda/cuda_entropy_codex.py`, `src/cuda/batched_entropy_codex.cu`

### Notable Current-State Items

- CPU 3-guess path uses strict exact pruning with checkpoint/resume support.
- GPU path supports:
  - mode 2 (pair search),
  - mode 3 geometric dispatch,
  - mode 3 hybrid entangled dispatch with `t` / `t-1` floor modes.
- Runtime/progress dashboards include normalized progress, pruning, and timing summaries.
- `numba` is a required dependency for CPU triple-entropy computation.
- Pattern-matrix cache validation now includes word-list signature and the
  actual `--answers-file` / `--allowed-file` sources.
- GPU CLI no longer advertises a nonexistent non-kernel fallback.

### Legacy Policy

Legacy/experimental GPU scripts and kernels were removed from this maintained
line. Project documentation references only the current implementation path.
