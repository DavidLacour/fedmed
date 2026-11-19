from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np

from fedmed.data import ArraySplit, dirichlet_partition, load_npz


def _resolve(path: str) -> Path:
    p = Path(path)
    if p.exists():
        return p
    # flwr runs apps from a staging dir; fall back to the repo root next to this app
    return Path(os.environ.get("FEDMED_ROOT", Path(__file__).resolve().parents[2])) / path


@lru_cache(maxsize=4)
def load_partition(data_path: str, partition_id: int, num_partitions: int, alpha: float) -> ArraySplit:
    """Each SuperNode only materialises its own hospital's slice."""
    train = load_npz(_resolve(data_path))["train"]
    idx = dirichlet_partition(train.y, num_partitions, alpha, seed=0)[partition_id]
    return ArraySplit(train.x[idx], train.y[idx])


@lru_cache(maxsize=1)
def load_test(data_path: str) -> ArraySplit:
    return load_npz(_resolve(data_path))["test"]


def label_counts(data_path: str, num_partitions: int, alpha: float) -> np.ndarray:
    return np.sum([np.bincount(load_partition(data_path, i, num_partitions, alpha).y, minlength=7)
                   for i in range(num_partitions)], axis=0)
