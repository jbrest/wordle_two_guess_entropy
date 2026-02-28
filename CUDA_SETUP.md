# CUDA Setup (Current Codex Path)

This project's current GPU search path is:

- CLI: `wordle_entropy_gpu_codex.py`
- Wrapper: `src/cuda/cuda_entropy_codex.py`
- Kernel: `src/cuda/batched_entropy_codex.cu`

Legacy GPU scripts/kernels are not part of the active workflow.
There is no non-kernel fallback mode in the maintained GPU CLI.

## Prerequisites

- NVIDIA GPU with CUDA support
- Linux/WSL2 with GPU passthrough
- CUDA toolkit (`nvcc` available)
- Python env with CUDA-enabled `torch`

## Verify Environment

```bash
nvcc --version
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

Expected: `torch.cuda.is_available()` prints `True`.

## Install Build Tools (Ubuntu/WSL)

```bash
sudo apt-get update
sudo apt-get install -y build-essential
```

## Install CUDA Toolkit (example: 12.6)

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-6
```

## PATH Setup

Add to `~/.bashrc`:

```bash
export PATH=/usr/local/cuda-12.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH
```

Reload:

```bash
source ~/.bashrc
```

## Run GPU Searches

```bash
# Mode 2 (pairs)
python wordle_entropy_gpu_codex.py --mode 2

# Mode 3 (triples), hybrid scheduler, t-1 floor
python wordle_entropy_gpu_codex.py --mode 3 --dispatch-mode hybrid --floor-source-mode t-1 --resume auto
```

## Useful Runtime Controls

```bash
python wordle_entropy_gpu_codex.py --mode 3 \
  --dispatch-mode hybrid \
  --floor-source-mode t-1 \
  --chunk-tasks 2000000 \
  --append-capacity 2000000 \
  --progress dashboard \
  --resume auto
```

## Notes

- Kernel compilation is handled automatically by `src/cuda/cuda_entropy_codex.py`.
- If append buffer overflow occurs, increase `--append-capacity` or reduce `--chunk-tasks`.
- Checkpoint files are selected automatically per mode/dispatch unless `--checkpoint-file` is set.
