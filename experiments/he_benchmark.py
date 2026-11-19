"""CKKS homomorphic aggregation vs pairwise-mask secure aggregation: cost per site."""

import json
import time
from pathlib import Path

import numpy as np

from fedmed.he import benchmark
from fedmed.secagg import RosterEntry, SecAggClient, unmask_sum
from fedmed.serialization import pack

out = {"he": benchmark()}
dim, n = out["he"]["dim"], out["he"]["n_sites"]
cs = [SecAggClient(f"s{i}") for i in range(n)]
roster = [RosterEntry(c.site, c.public_key) for c in cs]
vecs = [np.random.default_rng(i).normal(0, 0.01, dim).astype(np.float32) for i in range(n)]
t = time.perf_counter()
masked = [c.mask(v, roster, "bench", 1) for c, v in zip(cs, vecs)]
t_mask = (time.perf_counter() - t) / n
t = time.perf_counter()
s = unmask_sum(masked)
out["secagg"] = {
    "bytes_per_site": len(pack({}, {"m": masked[0]})), "mask_s_per_site": t_mask,
    "aggregate_s": time.perf_counter() - t, "max_abs_error": float(np.abs(s - np.sum(vecs, 0)).max()),
}
out["plain_bytes_per_site"] = len(pack({}, {"v": vecs[0]}))
Path("results").mkdir(exist_ok=True)
Path("results/he_benchmark.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
