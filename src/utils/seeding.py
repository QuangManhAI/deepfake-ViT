"""Deterministic RNG seeding for reproducible runs.

Sets python, numpy, and torch (CPU/CUDA) seeds. DataLoader workers are
auto-seeded by PyTorch from the main-process RNG, so a single main-process
seed keeps shuffles reproducible across runs even with ``num_workers > 0``.
Note: ``PYTHONHASHSEED`` must be set *before* the interpreter starts to take
effect, so it is not set here.
"""
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed all RNGs (python, numpy, torch CPU/CUDA) for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # no-op when CUDA is unavailable
