"""Local (on-site) training and evaluation.

One function, :func:`local_train`, implements the client side of FedAvg, FedProx and
SCAFFOLD, optionally under DP-SGD (Opacus). The algorithm-specific corrections are
applied as parameter-space steps *after* each optimizer step. They depend only on the
global model / control variates, never on the data, so they cost no privacy budget
and compose cleanly with Opacus' per-sample clipping and noise.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch import nn

from .data import ArraySplit, make_loader
from .models import build_model, get_flat, set_flat

ALGORITHMS = ("fedavg", "fedprox", "scaffold")


@dataclass
class DPConfig:
    noise_multiplier: float
    max_grad_norm: float = 1.0
    delta: float = 1e-5


@dataclass
class LocalConfig:
    algorithm: str = "fedavg"
    epochs: int = 1
    batch_size: int = 32
    lr: float = 0.05
    mu: float = 0.01  # FedProx proximal strength
    momentum: float = 0.0
    local_steps: int | None = None  # fixed number of SGD steps per round (overrides epochs)
    class_weights: list[float] | None = None  # from the federated label histogram (see quality.py)
    dp: DPConfig | None = None

    def __post_init__(self) -> None:
        if self.algorithm not in ALGORITHMS:
            raise ValueError(f"unknown algorithm {self.algorithm!r}")


@dataclass
class ClientState:
    """State that persists on a hospital node between rounds. Never sent in full."""

    c_local: np.ndarray | None = None  # SCAFFOLD control variate c_i
    privacy_engine: object | None = None  # Opacus engine; its accountant spans all rounds
    residual: np.ndarray | None = None  # error-feedback memory for lossy compression


@dataclass
class LocalResult:
    delta: np.ndarray  # w_local - w_global
    n_samples: int
    train_loss: float
    steps: int
    delta_c: np.ndarray | None = None  # SCAFFOLD: c_i_new - c_i
    epsilon: float | None = None  # cumulative (epsilon, delta)-DP spent by this site
    extra: dict = field(default_factory=dict)


def local_train(
    global_w: np.ndarray,
    data: ArraySplit,
    cfg: LocalConfig,
    state: ClientState,
    seed: int = 0,
    c_global: np.ndarray | None = None,
) -> LocalResult:
    warnings.filterwarnings("ignore", message=".*backward hook.*")
    warnings.filterwarnings("ignore", message=".*Secure RNG turned off.*")
    torch.manual_seed(seed)
    model = build_model()
    set_flat(model, global_w)
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum)
    loader = make_loader(data, cfg.batch_size, shuffle=True, seed=seed)

    if cfg.dp is not None:
        model, opt, loader = _make_private(model, opt, loader, cfg.dp, state)

    params = [p for p in model.parameters()]
    w0 = [p.detach().clone() for p in params]

    if cfg.algorithm == "scaffold":
        if c_global is None:
            c_global = np.zeros_like(global_w)
        if state.c_local is None:
            state.c_local = np.zeros_like(global_w)
        corr = _unflatten(c_global - state.c_local, params)  # c - c_i

    weight = torch.tensor(cfg.class_weights, dtype=torch.float32) if cfg.class_weights else None
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    steps, loss_sum, seen = 0, 0.0, 0
    for xb, yb in _batches(loader, cfg.epochs, cfg.local_steps):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()
        with torch.no_grad():
            if cfg.algorithm == "fedprox":
                # exact proximal step for (mu/2)||w - w_global||^2
                k = cfg.lr * cfg.mu
                for p, p0 in zip(params, w0):
                    p.add_(p0, alpha=k).div_(1 + k)
            elif cfg.algorithm == "scaffold":
                for p, cc in zip(params, corr):
                    p.sub_(cc, alpha=cfg.lr)
        steps += 1
        loss_sum += loss.item() * len(yb)
        seen += len(yb)

    w_new = torch.cat([p.detach().reshape(-1) for p in params]).numpy().astype(np.float32)
    delta = w_new - global_w
    result = LocalResult(delta=delta, n_samples=len(data), train_loss=loss_sum / max(seen, 1), steps=steps)

    if cfg.algorithm == "scaffold" and steps > 0:
        # SCAFFOLD option II: c_i+ = c_i - c + (x - y_i) / (K * lr)
        c_new = state.c_local - c_global - delta / (steps * cfg.lr)
        result.delta_c = (c_new - state.c_local).astype(np.float32)
        state.c_local = c_new.astype(np.float32)

    if cfg.dp is not None:
        result.epsilon = float(state.privacy_engine.get_epsilon(cfg.dp.delta))
    return result


def _batches(loader, epochs: int, local_steps: int | None):
    """Yield non-empty batches: ``epochs`` passes, or exactly ``local_steps`` batches."""
    if local_steps is None:
        for _ in range(epochs):
            for xb, yb in loader:
                if len(yb):  # Poisson sampling can produce empty batches
                    yield xb, yb
        return
    done = 0
    while done < local_steps:
        for xb, yb in loader:
            if len(yb):
                yield xb, yb
                done += 1
                if done == local_steps:
                    return


def _make_private(model, opt, loader, dp: DPConfig, state: ClientState):
    from opacus import PrivacyEngine

    if state.privacy_engine is None:
        state.privacy_engine = PrivacyEngine(accountant="rdp")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return state.privacy_engine.make_private(
            module=model,
            optimizer=opt,
            data_loader=loader,
            noise_multiplier=dp.noise_multiplier,
            max_grad_norm=dp.max_grad_norm,
            poisson_sampling=True,
        )


def _unflatten(vec: np.ndarray, like: list[torch.Tensor]) -> list[torch.Tensor]:
    out, off = [], 0
    t = torch.from_numpy(np.asarray(vec, dtype=np.float32))
    for p in like:
        out.append(t[off : off + p.numel()].view_as(p).clone())
        off += p.numel()
    return out


@torch.no_grad()
def evaluate(weights: np.ndarray, data: ArraySplit, batch_size: int = 512) -> dict:
    model = build_model()
    set_flat(model, weights)
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total, preds = 0.0, []
    for xb, yb in make_loader(data, batch_size, shuffle=False):
        logits = model(xb)
        total += loss_fn(logits, yb).item()
        preds.append(logits.argmax(1).numpy())
    pred = np.concatenate(preds)
    return {
        "loss": total / len(data),
        "accuracy": float((pred == data.y).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(data.y, pred)),
        "macro_f1": float(f1_score(data.y, pred, average="macro", zero_division=0)),
        "n": len(data),
    }


def class_weights_from_counts(counts, power: float = 0.5) -> list[float]:
    """Inverse-frequency weights, damped by ``power`` and normalised to mean 1."""
    c = np.maximum(np.asarray(counts, dtype=np.float64), 1.0)
    w = (c.sum() / c) ** power
    return (w / w.mean()).round(4).tolist()


def initial_weights(seed: int = 0) -> np.ndarray:
    return get_flat(build_model(seed))
