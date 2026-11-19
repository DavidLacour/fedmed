"""Uplink compression of model deltas: fp16, blockwise int8 quantisation and top-k
sparsification with error feedback. Byte counts are measured on the real wire format.
"""

from __future__ import annotations

import numpy as np

BLOCK = 1024


def parse(method: str) -> tuple[str, float]:
    if method.startswith("topk"):
        _, frac = method.split(":")
        return "topk", float(frac)
    if method in ("none", "fp16", "int8"):
        return method, 0.0
    raise ValueError(method)


def encode(vec: np.ndarray, method: str, residual: np.ndarray | None = None):
    """Returns (arrays, meta, new_residual). ``residual`` implements error feedback:
    whatever compression dropped this round is added back next round, which is what
    keeps aggressive top-k from stalling convergence."""
    kind, frac = parse(method)
    vec = vec.astype(np.float32)
    if kind == "none":
        return {"v": vec}, {"c": "none", "d": len(vec)}, None
    if kind == "fp16":
        return {"v": vec.astype(np.float16)}, {"c": "fp16", "d": len(vec)}, None
    acc = vec if residual is None else vec + residual
    if kind == "int8":
        pad = (-len(acc)) % BLOCK
        blocks = np.pad(acc, (0, pad)).reshape(-1, BLOCK)
        scale = np.abs(blocks).max(axis=1) / 127.0
        scale[scale == 0] = 1.0
        q = np.clip(np.round(blocks / scale[:, None]), -127, 127).astype(np.int8)
        dec = (q.astype(np.float32) * scale[:, None]).reshape(-1)[: len(acc)]
        return {"q": q.reshape(-1), "s": scale.astype(np.float32)}, {"c": "int8", "d": len(acc)}, acc - dec
    # top-k
    k = max(1, int(round(frac * len(acc))))
    idx = np.argpartition(np.abs(acc), -k)[-k:].astype(np.uint32)
    vals = acc[idx]
    new_res = acc.copy()
    new_res[idx] = 0.0
    return {"i": idx, "v": vals.astype(np.float32)}, {"c": "topk", "d": len(acc)}, new_res


def decode(arrays: dict[str, np.ndarray], meta: dict) -> np.ndarray:
    d = int(meta["d"])
    c = meta["c"]
    if c in ("none", "fp16"):
        out = arrays["v"].astype(np.float32)
    elif c == "int8":
        q = arrays["q"].astype(np.float32).reshape(-1, BLOCK)
        out = (q * arrays["s"][:, None]).reshape(-1)[:d]
    elif c == "topk":
        out = np.zeros(d, dtype=np.float32)
        idx = arrays["i"].astype(np.int64)
        if idx.size and (idx.max() >= d):
            raise ValueError("top-k index out of range")
        out[idx] = arrays["v"]
    else:
        raise ValueError(c)
    if out.shape != (d,):
        raise ValueError("decoded size mismatch")
    return out
