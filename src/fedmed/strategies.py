"""Server-side aggregation. Operates on flat float32 vectors only.

The server never sees data, only (possibly masked / noised) model deltas.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Update:
    """What the coordinator receives from one site (or the secure sum of all sites)."""

    weighted_delta: np.ndarray  # n_i * delta_i  (summable, so it works with secure aggregation)
    n_samples: float
    delta_c: np.ndarray | None = None  # SCAFFOLD control-variate delta
    n_clients: int = 1  # >1 when this is an already secure-aggregated sum


class Strategy:
    name = "base"

    def __init__(self, server_lr: float = 1.0):
        self.server_lr = server_lr

    def client_payload(self) -> dict[str, np.ndarray]:
        """Extra arrays broadcast with the global model (e.g. SCAFFOLD's c)."""
        return {}

    def aggregate(self, w: np.ndarray, updates: list[Update]) -> np.ndarray:
        total_n = sum(u.n_samples for u in updates)
        if total_n <= 0:
            return w
        avg = sum(u.weighted_delta for u in updates) / total_n
        return (w + self.server_lr * avg).astype(np.float32)


class FedAvg(Strategy):
    """McMahan et al. 2017. Sample-weighted average of client deltas."""

    name = "fedavg"


class FedProx(Strategy):
    """Li et al. 2020. Same aggregation as FedAvg; the proximal term lives on the client."""

    name = "fedprox"


class Scaffold(Strategy):
    """Karimireddy et al. 2020. Control variates correct client drift on non-IID data."""

    name = "scaffold"

    def __init__(self, dim: int, n_total_clients: int, server_lr: float = 1.0):
        super().__init__(server_lr)
        self.c = np.zeros(dim, dtype=np.float32)
        self.n_total = n_total_clients

    def client_payload(self) -> dict[str, np.ndarray]:
        return {"c_global": self.c}

    def aggregate(self, w: np.ndarray, updates: list[Update]) -> np.ndarray:
        new_w = super().aggregate(w, updates)
        n_part = sum(u.n_clients for u in updates)
        dc_sum = sum(u.delta_c for u in updates if u.delta_c is not None)
        if n_part:
            # c <- c + (|S|/N) * mean_i(delta_c_i) = c + sum_i(delta_c_i) / N
            self.c = (self.c + dc_sum / self.n_total).astype(np.float32)
        return new_w


def make_strategy(name: str, dim: int, n_clients: int, server_lr: float = 1.0) -> Strategy:
    if name == "fedavg":
        return FedAvg(server_lr)
    if name == "fedprox":
        return FedProx(server_lr)
    if name == "scaffold":
        return Scaffold(dim, n_clients, server_lr)
    raise ValueError(name)
