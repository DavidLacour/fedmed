"""Safe binary wire format for model weights and updates.

Never use pickle / torch.load on bytes from the network: unpickling untrusted input is
remote code execution. This format is a JSON header plus raw little-endian arrays,
with a dtype allowlist and size checks.
"""

from __future__ import annotations

import json
import struct

import numpy as np

MAGIC = b"FMD1"
ALLOWED_DTYPES = {"float32", "float16", "int8", "uint32", "uint64", "int64"}
MAX_HEADER = 1 << 20


def pack(meta: dict, arrays: dict[str, np.ndarray] | None = None) -> bytes:
    arrays = arrays or {}
    specs, chunks, offset = [], [], 0
    for name, arr in arrays.items():
        a = np.ascontiguousarray(arr)
        if a.dtype.name not in ALLOWED_DTYPES:
            raise TypeError(f"dtype {a.dtype} not allowed on the wire")
        b = a.astype(a.dtype.newbyteorder("<"), copy=False).tobytes()
        specs.append({"name": name, "dtype": a.dtype.name, "shape": list(a.shape), "offset": offset, "nbytes": len(b)})
        chunks.append(b)
        offset += len(b)
    header = json.dumps({"meta": meta, "arrays": specs}, separators=(",", ":")).encode()
    return MAGIC + struct.pack("<I", len(header)) + header + b"".join(chunks)


def unpack(buf: bytes) -> tuple[dict, dict[str, np.ndarray]]:
    if len(buf) < 8 or buf[:4] != MAGIC:
        raise ValueError("bad magic")
    (hlen,) = struct.unpack("<I", buf[4:8])
    if hlen > MAX_HEADER or 8 + hlen > len(buf):
        raise ValueError("bad header length")
    header = json.loads(buf[8 : 8 + hlen])
    body = memoryview(buf)[8 + hlen :]
    arrays = {}
    for s in header["arrays"]:
        if s["dtype"] not in ALLOWED_DTYPES:
            raise ValueError(f"dtype {s['dtype']} not allowed")
        dt = np.dtype(s["dtype"]).newbyteorder("<")
        expected = int(np.prod(s["shape"], dtype=np.int64)) * dt.itemsize
        if s["nbytes"] != expected or s["offset"] < 0 or s["offset"] + s["nbytes"] > len(body):
            raise ValueError(f"array {s['name']} out of bounds")
        raw = body[s["offset"] : s["offset"] + s["nbytes"]]
        arrays[s["name"]] = np.frombuffer(raw, dtype=dt).reshape(s["shape"]).astype(dt.newbyteorder("="))
    return header["meta"], arrays
