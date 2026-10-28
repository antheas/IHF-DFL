import logging
import multiprocessing as mp
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.paths import repo_root
from utils.colors import green, red, yellow

logging.getLogger("torch._inductor").setLevel(logging.ERROR)
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_CUDA = (DEVICE.type == "cuda")
PIN_MEMORY = USE_CUDA
NON_BLOCKING = USE_CUDA
# The loaders built in data.py (one val loader per participant + the test loader) are iterated many times per
# round over small in-memory datasets: starting workers for each pass costs more CPU than it saves, and
# persistent ones would keep ~700MB per worker alive for the whole run. Training builds its own loaders.
NUM_WORKERS = 0
PERSISTENT_WORKERS = False

AMP = USE_CUDA  # Optional: mixed precision on CUDA
DATASET_ROOT = repo_root(Path(__file__)) / "data" / "datasets"
# cuDNN autotune for fixed-size inputs (both MNIST 28x28 and CIFAR-10 32x32)
torch._dynamo.config.cache_size_limit = 512
torch.backends.cudnn.benchmark = USE_CUDA

if DEVICE.type == "cuda":
    torch.backends.cuda.matmul.fp32_precision = "tf32"
    torch.backends.cudnn.conv.fp32_precision = "tf32"


def model_to_device(net: nn.Module) -> nn.Module:
    # Move model once; keep it on the chosen device
    return net.to(DEVICE, non_blocking=NON_BLOCKING)


def loader_mp_context(num_workers: int):
    # Parallel experiment runs are spawned, which makes spawn their default too: every DataLoader worker would
    # then re-import torch and get its own pickled copy of the dataset (a Subset pickles the whole parent
    # dataset), ~700MB each. Forked workers share the parent's memory; they only build CPU tensors, never CUDA.
    if num_workers > 0 and "fork" in mp.get_all_start_methods():
        return "fork"
    return None


def cuda_safe_dataloader(ds, batch_size, shuffle=False):
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=PIN_MEMORY,
        num_workers=NUM_WORKERS,
        persistent_workers=PERSISTENT_WORKERS,
        multiprocessing_context=loader_mp_context(NUM_WORKERS),
    )


def print_training_mode(num_gpus: int, num_processes: int):
    """Prints a clean status message describing how training will run."""
    if num_gpus >= 2:
        print(green(f"Detected {num_gpus} GPU(s) → Parallel multi-GPU training"))

    elif num_gpus == 1:
        print(green("Detected 1 GPU → Sequential GPU training"))

    else:  # CPU-only
        if num_processes > 1:
            print(yellow(
                f"Detected 0 GPU(s) → Parallel CPU training ({num_processes} workers)"
            ))
        else:
            print(red("Detected 0 GPU(s) → Sequential CPU mode"))
