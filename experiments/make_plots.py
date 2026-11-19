"""Turn results/runs/*.json into the figures and the numbers table used in RESULTS.md."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RUNS = Path("results/runs")
OUT = Path("results/plots")
# Fixed categorical order (validated palette): blue, orange, aqua, yellow, magenta, green, violet, red
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, MUTED, GRID = "#1f2328", "#59636e", "#e6e8eb"

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 160, "font.size": 9.5, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "lines.linewidth": 2.0, "legend.frameon": False, "axes.titleweight": "bold", "axes.titlesize": 10.5,
})


def load(prefix: str) -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text()) for p in sorted(RUNS.glob(f"{prefix}*.json"))}


def curve(run: dict, key: str = "balanced_accuracy") -> np.ndarray:
    return np.array([r[key] for r in run["rounds"]])


def smooth(y: np.ndarray, k: int = 5) -> np.ndarray:
    if len(y) < k:
        return y
    pad = np.pad(y, (k // 2, k - 1 - k // 2), mode="edge")
    return np.convolve(pad, np.ones(k) / k, mode="valid")


def tail_mean(run: dict, key: str = "balanced_accuracy", n: int = 10) -> float:
    return float(np.mean(curve(run, key)[-n:]))


METHODS = [("fedavg", "FedAvg"), ("fedprox-mu0.01", "FedProx μ=0.01"), ("fedprox-mu0.1", "FedProx μ=0.1"),
           ("scaffold", "SCAFFOLD")]
ALPHAS = [("0.1", "α = 0.1 (strongly non-IID)"), ("0.5", "α = 0.5 (moderately non-IID)"), ("100.0", "α = 100 (≈ IID)")]


def plot_aggregation(table: list[str]) -> None:
    runs = load("agg_")
    central = [tail_mean({"rounds": r["result"]["history"]}) for r in load("central_").values()]
    local = {a: [r["result"]["mean_balanced_accuracy"] for k, r in load(f"local_a{a}").items()] for a in ("0.1", "0.5")}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    table += ["| Method | " + " | ".join(t for _, t in ALPHAS) + " |", "|---|" + "---|" * len(ALPHAS)]
    rows = defaultdict(list)
    for ax, (a, title) in zip(axes, ALPHAS):
        for ci, (m, label) in enumerate(METHODS):
            seeds = [r for k, r in runs.items() if k.startswith(f"agg_{m}_a{a}_s")]
            if not seeds:
                rows[label].append("–")
                continue
            ys = np.stack([smooth(curve(r)) for r in seeds])
            x = np.arange(1, ys.shape[1] + 1)
            ax.plot(x, ys.mean(0), color=C[ci], label=label)
            if len(seeds) > 1:
                ax.fill_between(x, ys.min(0), ys.max(0), color=C[ci], alpha=0.12, linewidth=0)
            finals = [tail_mean(r) for r in seeds]
            rows[label].append(f"{np.mean(finals):.3f} ± {np.std(finals):.3f}")
        if central:
            ax.axhline(np.mean(central), color=INK, ls="--", lw=1.2)
            ax.text(2, np.mean(central) + 0.008, "pooled data (upper bound)", color=MUTED, fontsize=8)
        if a in local and local[a]:
            ax.axhline(np.mean(local[a]), color=MUTED, ls=":", lw=1.4)
            ax.text(99, np.mean(local[a]) - 0.012, "each hospital alone (mean)", ha="right", va="top", color=MUTED, fontsize=8)
        ax.set_title(title)
        ax.set_xlabel("communication round")
    axes[0].set_ylabel("balanced accuracy (test, 5-round smoothing)")
    axes[-1].legend(loc="lower right")
    fig.suptitle("Aggregation algorithms under label skew, mean of 2 seeds (band = min to max)", x=0.01,
                 ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "aggregation_comparison.png")
    plt.close(fig)
    for label, vals in rows.items():
        table.append(f"| {label} | " + " | ".join(vals) + " |")
    if central:
        table.append(f"| Pooled data (no federation) | {np.mean(central):.3f} | | |")
    for a in ("0.1", "0.5"):
        if local.get(a):
            table.append(f"| Each hospital alone, α={a} (mean of 5) | {np.mean(local[a]):.3f} | | |")


def _dp_points(prefix: str):
    runs = {k: r for k, r in load(prefix).items() if r["config"]["noise_multiplier"]}
    pts = sorted(((r["summary"]["final_epsilon"], tail_mean(r), tail_mean(r, "accuracy"),
                   r["config"]["noise_multiplier"]) for r in runs.values()), key=lambda t: t[0])
    return list(map(np.array, zip(*pts))) if pts else None


def plot_dp(table: list[str]) -> None:
    tuned, naive = _dp_points("dp2_"), _dp_points("dpC1_")
    ref = load("dp2_nodp")
    if tuned is None and naive is None:
        return
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    if tuned is not None:
        eps, bal, _, sig = tuned
        ax.plot(eps, bal, "-o", color=C[0], ms=7, mec="white", mew=2, label="DP-SGD, clip C=10, batch 128")
        for e, b, s_ in zip(eps, bal, sig):
            ax.annotate(f"σ={s_:g}", (e, b), textcoords="offset points", xytext=(6, 6), fontsize=8, color=MUTED)
    if naive is not None:
        ax.plot(naive[0], naive[1], "-s", color=C[1], ms=6, mec="white", mew=2,
                label="DP-SGD, clip C=1, batch 32 (fails)")
    if ref:
        nb = np.mean([tail_mean(r) for r in ref.values()])
        ax.axhline(nb, color=INK, ls="--", lw=1.2)
        ax.text(0.01, nb + 0.006, "same settings, no DP (ε = ∞)", color=MUTED, fontsize=8,
                transform=ax.get_yaxis_transform())
    ax.axhline(1 / 7, color=MUTED, ls=":", lw=1)
    ax.text(0.01, 1 / 7 + 0.004, "chance (1/7)", color=MUTED, fontsize=8, transform=ax.get_yaxis_transform())
    ax.set_xscale("log")
    ax.set_xlabel("ε spent by the worst-off hospital (δ = 1e-5, log scale)")
    ax.set_ylabel("balanced accuracy (test, last 10 rounds)")
    ax.set_title("Privacy–utility trade-off (record-level DP at every hospital)", loc="left")
    ax.legend(loc="center right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "dp_privacy_utility.png")
    plt.close(fig)
    table += ["", "| setting | noise σ | ε (worst site) | balanced acc | accuracy |", "|---|---|---|---|---|"]
    for name, pts in (("C=10, B=128, 50 rounds", tuned), ("C=1, B=32, 100 rounds", naive)):
        if pts is not None:
            table += [f"| {name} | {s_:g} | {e:.1f} | {b:.3f} | {a:.3f} |" for e, b, a, s_ in zip(*pts)]
    if ref:
        r = list(ref.values())
        table.append(f"| C=10 settings, no DP | – | ∞ | {np.mean([tail_mean(x) for x in r]):.3f} | "
                     f"{np.mean([tail_mean(x, 'accuracy') for x in r]):.3f} |")


def plot_communication(table: list[str]) -> None:
    runs = load("comp_")
    if not runs:
        return
    order = [("comp_none_s0", "float32 (baseline)"), ("comp_fp16_s0", "fp16"), ("comp_int8_s0", "int8 blockwise"),
             ("comp_topk0.1_s0", "top-10% + error feedback"), ("comp_topk0.01_s0", "top-1% + error feedback"),
             ("comp_secagg_s0", "secure aggregation (masked)")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 3.9), gridspec_kw={"width_ratios": [1.5, 1]})
    table += ["", "| Uplink encoding | MB / site / round | total uplink (100 rounds) | balanced acc | "
              "rounds to 0.30 bal. acc | energy (kWh, CodeCarbon est.) |", "|---|---|---|---|---|---|"]
    names, per_round = [], []
    for ci, (k, label) in enumerate(order):
        if k not in runs:
            continue
        r = runs[k]
        up = np.cumsum([x["bytes_up"] for x in r["rounds"]]) / 1e6
        a1.plot(up, smooth(curve(r)), color=C[ci], label=label)
        n_sites = r["config"]["n_sites"]
        mb = r["rounds"][0]["bytes_up"] / n_sites / 1e6
        names.append(label)
        per_round.append(mb)
        target = next((x["round"] for x in r["rounds"] if x["balanced_accuracy"] >= 0.30), None)
        kwh = r["summary"].get("energy_kwh")
        table.append(f"| {label} | {mb:.3f} | {up[-1]:.0f} MB | {tail_mean(r):.3f} | {target or '–'} | "
                     f"{kwh:.4f} |" if kwh else f"| {label} | {mb:.3f} | {up[-1]:.0f} MB | {tail_mean(r):.3f} | "
                     f"{target or '–'} | – |")
    he = Path("results/he_benchmark.json")
    if he.exists():
        h = json.loads(he.read_text())["he"]
        names.append("CKKS homomorphic enc.")
        per_round.append(h["he_bytes_per_site"] / 1e6)
        table.append(f"| CKKS homomorphic encryption (benchmark only) | {h['he_bytes_per_site'] / 1e6:.2f} | – | – | – | – |")
    a1.set_xscale("log")
    a1.set_xlabel("cumulative uplink traffic, all sites (MB, log scale)")
    a1.set_ylabel("balanced accuracy (test)")
    a1.set_title("Accuracy per byte sent", loc="left")
    a1.legend(loc="lower right", fontsize=8)
    y = np.arange(len(names))
    a2.barh(y, per_round, color=C[0], height=0.6)
    for yi, v in zip(y, per_round):
        a2.text(v * 1.08, yi, f"{v:.2f} MB" if v >= 0.1 else f"{v * 1000:.0f} kB", va="center", fontsize=8, color=INK)
    a2.set_yticks(y, names)
    a2.invert_yaxis()
    a2.set_xscale("log")
    a2.set_xlim(right=max(per_round) * 6)
    a2.set_xlabel("uplink per site per round (MB, log scale)")
    a2.set_title("Message size", loc="left")
    a2.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "communication.png")
    plt.close(fig)


def plot_scaling(table: list[str]) -> None:
    runs = load("scale_")
    if not runs:
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.4))
    table += ["", "| sites | client time / round, plain (s) | client time / round, secagg (s) | "
              "uplink / round, plain (MB) | uplink / round, secagg (MB) |", "|---|---|---|---|---|"]
    data = {}
    for ci, (mode, label) in enumerate([("plain", "plain updates"), ("sa", "secure aggregation")]):
        pts = sorted((r["config"]["n_sites"], np.mean([x["client_time_s"] for x in r["rounds"]]),
                      np.mean([x["bytes_up"] for x in r["rounds"]]) / 1e6)
                     for k, r in runs.items() if k.endswith(mode))
        n, t, b = map(np.array, zip(*pts))
        data[mode] = dict(zip(n, zip(t, b)))
        a1.plot(n, t / n, "-o", color=C[ci], ms=7, mec="white", mew=2, label=label)
        a2.plot(n, b, "-o", color=C[ci], ms=7, mec="white", mew=2, label=label)
    a1.set_xlabel("number of hospitals")
    a1.set_ylabel("seconds per site per round")
    a1.set_title("Client compute (training + encoding)", loc="left")
    a2.set_xlabel("number of hospitals")
    a2.set_ylabel("MB per round (all sites)")
    a2.set_title("Coordinator ingress", loc="left")
    a2.legend()
    fig.tight_layout()
    fig.savefig(OUT / "scalability.png")
    plt.close(fig)
    for n in sorted(data["plain"]):
        p, s = data["plain"][n], data["sa"].get(n, (float("nan"),) * 2)
        table.append(f"| {n} | {p[0] / n:.2f} | {s[0] / n:.2f} | {p[1]:.1f} | {s[1]:.1f} |")


def plot_partition() -> None:
    from fedmed.simulation import SimConfig, partition

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.3), sharey=False)
    classes = ["AK", "BCC", "BKL", "DF", "MEL", "NV", "VASC"]
    for ax, a in zip(axes, (0.1, 0.5)):
        sites, _ = partition(SimConfig(alpha=a))
        counts = np.array([np.bincount(s.y, minlength=7) for s in sites])
        frac = counts / counts.sum(1, keepdims=True)
        left = np.zeros(len(sites))
        for c in range(7):
            ax.barh(range(len(sites)), frac[:, c], left=left, color=C[c], height=0.65, label=classes[c],
                    edgecolor="white", linewidth=1.5)
            left += frac[:, c]
        ax.set_yticks(range(len(sites)), [f"hospital-{chr(97 + i)} (n={len(s)})" for i, s in enumerate(sites)])
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xlabel("share of the hospital's images")
        ax.set_title(f"Label mix per hospital, α = {a}", loc="left")
        ax.grid(False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, ncol=7, loc="lower center", fontsize=8)
    fig.savefig(OUT / "partition.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    table: list[str] = ["## Aggregation (balanced accuracy, mean of last 10 rounds, mean ± sd over seeds)", ""]
    plot_aggregation(table)
    table += ["", "## Differential privacy"]
    plot_dp(table)
    table += ["", "## Communication"]
    plot_communication(table)
    table += ["", "## Scalability (10 rounds, α = 0.5)"]
    plot_scaling(table)
    plot_partition()
    Path("results/summary.md").write_text("\n".join(table) + "\n")
    print("\n".join(table))


if __name__ == "__main__":
    main()
