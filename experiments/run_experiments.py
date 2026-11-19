"""Run every benchmark in the write-up. Results land in results/runs/*.json.

    python experiments/run_experiments.py --suite all --workers 8
    python experiments/run_experiments.py --suite smoke          # 2-minute CI version

Each run is independent, so they are spread across worker processes.
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path

OUT = Path("results/runs")


def suites(rounds: int) -> dict[str, list[dict]]:
    agg = [
        dict(name=f"agg_{alg}{'' if mu is None else f'-mu{mu}'}_a{a}_s{s}", algorithm=alg, alpha=a, seed=s,
             rounds=rounds, track_energy=True, **({} if mu is None else {"mu": mu}))
        for a in (0.1, 0.5, 100.0)
        for s in (0, 1)
        for alg, mu in (("fedavg", None), ("fedprox", 0.01), ("fedprox", 0.1), ("scaffold", None))
    ]
    base = [
        dict(name=f"local_a{a}_s{s}", kind="local", alpha=a, seed=s, rounds=rounds)
        for a in (0.1, 0.5) for s in (0, 1)
    ] + [dict(name=f"central_s{s}", kind="central", seed=s, rounds=rounds, local_steps=250) for s in (0, 1)]
    # DP-SGD needs its own hyper-parameters: per-sample gradient norms here are ~5-20, so C=1
    # clips away most of the signal (results/runs/dpC1_* kept as the documented failure).
    dp_hp = dict(batch_size=128, local_steps=20, lr=0.2, max_grad_norm=10.0, rounds=50)
    dp = [
        dict(name=f"dp2_sigma{sig}_s0", algorithm="fedavg", alpha=0.5, noise_multiplier=sig, **dp_hp)
        for sig in (0.5, 0.7, 1.0, 1.5, 2.0, 3.0)
    ] + [dict(name="dp2_nodp_s0", algorithm="fedavg", alpha=0.5, **dp_hp)]
    comp = [
        dict(name=f"comp_{c.replace(':', '')}_s0", algorithm="fedavg", alpha=0.5, compression=c, rounds=rounds,
             track_energy=True)
        for c in ("none", "fp16", "int8", "topk:0.1", "topk:0.01")
    ] + [dict(name="comp_secagg_s0", algorithm="fedavg", alpha=0.5, secagg=True, rounds=rounds, track_energy=True)]
    scale = [
        dict(name=f"scale_n{n}_{'sa' if sa else 'plain'}", algorithm="fedavg", alpha=0.5, n_sites=n,
             secagg=sa, rounds=10)
        for n in (3, 5, 10, 20) for sa in (False, True)
    ]
    smoke = [
        dict(name="smoke_fedavg", algorithm="fedavg", rounds=2, local_steps=5, track_energy=True),
        dict(name="smoke_scaffold_secagg", algorithm="scaffold", rounds=2, local_steps=5, secagg=True),
        dict(name="smoke_dp_topk", algorithm="fedprox", rounds=2, local_steps=5, noise_multiplier=1.0,
             compression="topk:0.1"),
    ]
    return {"agg": agg, "baselines": base, "dp": dp, "comp": comp, "scale": scale, "smoke": smoke}


def _job(spec: dict) -> str:
    warnings.filterwarnings("ignore")
    from fedmed.simulation import SimConfig, run, run_centralised, run_local_only

    spec = dict(spec)
    kind = spec.pop("kind", "fl")
    cfg = replace(SimConfig(torch_threads=2), **spec)
    path = OUT / f"{cfg.name}.json"
    if kind == "fl":
        run(cfg, verbose=False).save(path)
    elif kind == "local":
        path.write_text(json.dumps({"config": asdict(cfg), "result": run_local_only(cfg)}, indent=1))
    else:
        path.write_text(json.dumps({"config": asdict(cfg), "result": run_centralised(cfg)}, indent=1))
    return cfg.name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="all")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--force", action="store_true", help="re-run even if the result file exists")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    all_suites = suites(args.rounds)
    names = [s for s in all_suites if s != "smoke"] if args.suite == "all" else args.suite.split(",")
    jobs = [j for n in names for j in all_suites[n]]
    if not args.force:
        jobs = [j for j in jobs if not (OUT / f"{j['name']}.json").exists()]
    # longest jobs first so the pool drains evenly
    jobs.sort(key=lambda j: (j.get("noise_multiplier") is None, -j.get("rounds", 0)))
    print(f"{len(jobs)} runs on {args.workers} workers", flush=True)
    with ProcessPoolExecutor(args.workers) as ex:
        futs = {ex.submit(_job, j): j["name"] for j in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                print(f"[{i}/{len(jobs)}] done {f.result()}", flush=True)
            except Exception as exc:  # keep going, report at the end
                print(f"[{i}/{len(jobs)}] FAILED {futs[f]}: {exc!r}", flush=True)


if __name__ == "__main__":
    main()
