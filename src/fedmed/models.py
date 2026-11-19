"""Models and flat-vector helpers.

The model is intentionally small: the point of the project is the federated system.
GroupNorm is used instead of BatchNorm because BatchNorm mixes samples within a batch,
which breaks per-sample gradient clipping (DP-SGD) and behaves badly on non-IID data.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from . import NUM_CLASSES


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, width: int = 32):
        super().__init__()

        def block(cin: int, cout: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.GroupNorm(8, cout),
                nn.ReLU(),
            )

        self.features = nn.Sequential(
            block(3, width),
            nn.MaxPool2d(2),  # 14x14
            block(width, 2 * width),
            nn.MaxPool2d(2),  # 7x7
            block(2 * width, 4 * width),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(4 * width, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(torch.flatten(self.features(x), 1))


def build_model(seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    return SmallCNN()


def get_flat(model: nn.Module) -> np.ndarray:
    """All trainable parameters as one float32 vector (the unit we aggregate)."""
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()]).cpu().numpy().astype(np.float32)


def set_flat(model: nn.Module, flat: np.ndarray) -> None:
    flat_t = torch.from_numpy(np.asarray(flat, dtype=np.float32))
    offset = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            p.copy_(flat_t[offset : offset + n].view_as(p))
            offset += n
    if offset != flat_t.numel():
        raise ValueError(f"Vector size {flat_t.numel()} does not match model ({offset})")


def num_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
