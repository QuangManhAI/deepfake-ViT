"""Deterministic RNG seeding for reproducible runs.

Note: ``PYTHONHASHSEED`` must be set *before* the interpreter starts to take
effect, so it is not set here. The global torch/numpy/python seeds below are
sufficient for the DataLoaders used in this repo (``num_workers=0``, global
shuffle RNG).
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
