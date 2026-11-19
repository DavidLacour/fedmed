"""In-process federated simulation used for the benchmarks.

It runs exactly the same client (``local_train``), compression, secure-aggregation and
strategy code as the networked deployment. Only the transport is replaced by function
calls, but every message is still serialised so the byte counts are real.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from . import compression
from .data import ArraySplit, dirichlet_partition, load_npz, site_names
from .secagg import RosterEntry, SecAggClient, unmask_sum
from .serialization import pack, unpack
from .strategies import Update, make_strategy
from .train import (
    ClientState,
    DPConfig,
    LocalConfig,
    class_weights_from_counts,
    evaluate,
    initial_weights,
    local_train,
)


@dataclass
class SimConfig:
    name: str = "run"
    algorithm: str = "fedavg"
    n_sites: int = 5
    alpha: float = 0.5
    rounds: int = 50
    local_epochs: int = 1
    local_steps: int | None = 50
    batch_size: int = 32
    lr: float = 0.1
    mu: float = 0.01
    noise_multiplier: float | None = None  # None = no DP
    max_grad_norm: float = 1.0
    delta: float = 1e-5
    class_weight_power: float | None = 0.5  # None = plain cross-entropy
    compression: str = "none"
    secagg: bool = False
    seed: int = 0
    data_path: str = "data/raw/dermamnist.npz"
    track_energy: bool = False
    torch_threads: int = 4

    def local_config(self, label_counts=None) -> LocalConfig:
        dp = DPConfig(self.noise_multiplier, self.max_grad_norm, self.delta) if self.noise_multiplier else None
        cw = None
        if self.class_weight_power and label_counts is not None:
            cw = class_weights_from_counts(label_counts, self.class_weight_power)
        return LocalConfig(class_weights=cw, algorithm=self.algorithm, epochs=self.local_epochs, local_steps=self.local_steps, batch_size=self.batch_size,
                           lr=self.lr, mu=self.mu, dp=dp)


@dataclass
class RunLog:
    config: dict
    rounds: list[dict] = field(default_factory=list)
    summary: dict = field(default_factory=dict)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=1))


def federated_label_counts(sites: list[ArraySplit]) -> np.ndarray:
    """Sum of per-site label histograms, as the coordinator receives them in quality reports."""
    return np.sum([np.bincount(s.y, minlength=7) for s in sites], axis=0)


def partition(cfg: SimConfig) -> tuple[list[ArraySplit], ArraySplit]:
    data = load_npz(Path(cfg.data_path))
    tr = data["train"]
    parts = dirichlet_partition(tr.y, cfg.n_sites, cfg.alpha, seed=cfg.seed)
    return [ArraySplit(tr.x[p], tr.y[p]) for p in parts], data["test"]


def run(cfg: SimConfig, verbose: bool = True) -> RunLog:
    torch.set_num_threads(cfg.torch_threads)
    sites, test = partition(cfg)
    names = site_names(cfg.n_sites)
    lcfg = cfg.local_config(federated_label_counts(sites))
    w = initial_weights(cfg.seed)
    strategy = make_strategy(cfg.algorithm, len(w), cfg.n_sites)
    states = [ClientState() for _ in sites]
    log = RunLog(config=asdict(cfg))
    log.summary["site_sizes"] = [len(s) for s in sites]
    log.summary["site_label_counts"] = [np.bincount(s.y, minlength=7).tolist() for s in sites]

    sa_clients = [SecAggClient(n) for n in names] if cfg.secagg else []
    roster = [RosterEntry(c.site, c.public_key) for c in sa_clients]

    tracker = _start_tracker(cfg) if cfg.track_energy else None
    t_start = time.perf_counter()
    for rnd in range(1, cfg.rounds + 1):
        t0 = time.perf_counter()
        down_msg = pack({"round": rnd}, {"w": w, **strategy.client_payload()})
        bytes_down = len(down_msg) * len(sites)
        bytes_up, client_time, masked, updates, eps, losses = 0, 0.0, [], [], [], []
        for i, (site, st) in enumerate(zip(sites, states)):
            _, arrs = unpack(down_msg)
            tc = time.perf_counter()
            res = local_train(arrs["w"], site, lcfg, st, seed=cfg.seed * 100003 + rnd * 101 + i,
                              c_global=arrs.get("c_global"))
            losses.append(res.train_loss)
            if res.epsilon is not None:
                eps.append(res.epsilon)
            if cfg.secagg:
                parts = [res.delta * res.n_samples, np.array([res.n_samples], np.float32)]
                if res.delta_c is not None:
                    parts.append(res.delta_c)
                m = sa_clients[i].mask(np.concatenate(parts), roster, cfg.name, rnd)
                msg = pack({"site": names[i], "round": rnd}, {"masked": m})
                client_time += time.perf_counter() - tc
                bytes_up += len(msg)
                masked.append(unpack(msg)[1]["masked"])
            else:
                arrays, meta, st.residual = compression.encode(res.delta, cfg.compression, st.residual)
                extra = {"delta_c": res.delta_c} if res.delta_c is not None else {}
                msg = pack({"site": names[i], "round": rnd, "n": res.n_samples, "comp": meta}, {**arrays, **extra})
                client_time += time.perf_counter() - tc
                bytes_up += len(msg)
                meta_r, arr_r = unpack(msg)
                delta = compression.decode(arr_r, meta_r["comp"])
                updates.append(Update(delta * meta_r["n"], meta_r["n"], arr_r.get("delta_c")))
        if cfg.secagg:
            total = unmask_sum(masked)
            d = len(w)
            dc = total[d + 1 :] if cfg.algorithm == "scaffold" else None
            updates = [Update(total[:d], float(total[d]), dc, n_clients=len(sites))]
        w = strategy.aggregate(w, updates)
        ev = evaluate(w, test)
        rec = {
            "round": rnd,
            **{k: ev[k] for k in ("loss", "accuracy", "balanced_accuracy", "macro_f1")},
            "train_loss": float(np.mean(losses)),
            "bytes_up": bytes_up,
            "bytes_down": bytes_down,
            "round_time_s": time.perf_counter() - t0,
            "client_time_s": client_time,
            "epsilon": max(eps) if eps else None,
        }
        log.rounds.append(rec)
        if verbose:
            e = f" eps={rec['epsilon']:.2f}" if rec["epsilon"] else ""
            print(f"[{cfg.name}] round {rnd:3d} acc={ev['accuracy']:.3f} bal_acc={ev['balanced_accuracy']:.3f}"
                  f" up={bytes_up / 1e6:.2f}MB{e}", flush=True)
    log.summary.update(_summarise(log.rounds))
    log.summary["wall_time_s"] = time.perf_counter() - t_start
    log.summary["n_params"] = len(w)
    if tracker is not None:
        log.summary.update(_stop_tracker(tracker))
    return log


def run_local_only(cfg: SimConfig) -> dict:
    """Baseline: every hospital trains alone on its own data (no collaboration)."""
    sites, test = partition(cfg)
    lcfg = cfg.local_config(federated_label_counts(sites))
    out = []
    for site in sites:
        w, st = initial_weights(cfg.seed), ClientState()
        for e in range(cfg.rounds):
            w = w + local_train(w, site, lcfg, st, seed=cfg.seed * 7 + e).delta
        out.append(evaluate(w, test))
    return {"per_site": out, "mean_accuracy": float(np.mean([o["accuracy"] for o in out])),
            "mean_balanced_accuracy": float(np.mean([o["balanced_accuracy"] for o in out]))}


def run_centralised(cfg: SimConfig) -> dict:
    """Upper-bound baseline: all data pooled in one place (what FL avoids)."""
    sites, test = partition(cfg)
    pooled = ArraySplit(np.concatenate([s.x for s in sites]), np.concatenate([s.y for s in sites]))
    lcfg = cfg.local_config(federated_label_counts(sites))
    w, st, hist = initial_weights(cfg.seed), ClientState(), []
    for e in range(cfg.rounds):
        w = w + local_train(w, pooled, lcfg, st, seed=cfg.seed * 7 + e).delta
        hist.append(evaluate(w, test))
    return {"final": hist[-1], "history": hist}


def _summarise(rounds: list[dict]) -> dict:
    accs = [r["accuracy"] for r in rounds]
    tail = rounds[-5:]
    return {
        "final_accuracy": float(np.mean([r["accuracy"] for r in tail])),
        "final_balanced_accuracy": float(np.mean([r["balanced_accuracy"] for r in tail])),
        "final_macro_f1": float(np.mean([r["macro_f1"] for r in tail])),
        "best_accuracy": float(max(accs)),
        "total_bytes_up": int(sum(r["bytes_up"] for r in rounds)),
        "total_bytes_down": int(sum(r["bytes_down"] for r in rounds)),
        "final_epsilon": rounds[-1]["epsilon"],
    }


def rounds_to_target(rounds: list[dict], target: float, key: str = "accuracy") -> int | None:
    for r in rounds:
        if r[key] >= target:
            return r["round"]
    return None


def _start_tracker(cfg: SimConfig):
    try:
        import logging

        from codecarbon import OfflineEmissionsTracker

        logging.getLogger("codecarbon").setLevel(logging.ERROR)
        t = OfflineEmissionsTracker(country_iso_code="CHE", project_name=cfg.name, save_to_file=False,
                                    log_level="error", tracking_mode="process",
                                    allow_multiple_runs=True)
        t.start()
        return t
    except Exception as exc:  # energy tracking is best effort
        print(f"codecarbon unavailable: {exc}")
        return None


def _stop_tracker(t) -> dict:
    try:
        kg = t.stop()
        data = t.final_emissions_data
        return {"energy_kwh": float(data.energy_consumed), "co2_kg": float(kg or 0.0)}
    except Exception as exc:
        return {"energy_error": str(exc)}
