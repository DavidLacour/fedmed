"""Homomorphic-encryption aggregation demo (CKKS via TenSEAL).

Each site encrypts its weighted update under a shared CKKS public key. The coordinator
adds ciphertexts without being able to read them, and only the holder of the secret key
can decrypt the *sum*.

This is a demo of the mechanism and its cost, not the production path:
* With a single secret key, whoever holds it could also decrypt individual updates,
  so it must not be the coordinator. Here it is a designated "key-holder" site. Real
  deployments use threshold / multi-key CKKS, where several parties must cooperate.
* CKKS is approximate arithmetic: the decrypted sum carries ~1e-6 relative error.
The benchmark reports ciphertext expansion and time, next to pairwise-mask secure aggregation.
"""

from __future__ import annotations

import time

import numpy as np


def make_context(poly_modulus_degree: int = 8192):
    import tenseal as ts

    ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=poly_modulus_degree,
                     coeff_mod_bit_sizes=[60, 40, 40, 60])
    ctx.global_scale = 2**40
    ctx.generate_galois_keys()
    return ctx


def encrypt_vector(ctx, vec: np.ndarray) -> list:
    import tenseal as ts

    slots = 8192 // 2  # CKKS packs N/2 values per ciphertext
    return [ts.ckks_vector(ctx, vec[i:i + slots].tolist()) for i in range(0, len(vec), slots)]


def benchmark(n_sites: int = 5, dim: int = 94_599, seed: int = 0) -> dict:
    import tenseal as ts

    rng = np.random.default_rng(seed)
    updates = [rng.normal(0, 0.01, dim).astype(np.float32) for _ in range(n_sites)]
    plain_sum = np.sum(updates, axis=0)

    t = time.perf_counter()
    secret_ctx = make_context()
    public_ctx = ts.context_from(secret_ctx.serialize(save_secret_key=False))  # what sites + server get
    t_keygen = time.perf_counter() - t

    t = time.perf_counter()
    enc = [encrypt_vector(public_ctx, u) for u in updates]
    t_enc = (time.perf_counter() - t) / n_sites
    bytes_per_site = sum(len(c.serialize()) for c in enc[0])

    t = time.perf_counter()  # coordinator: ciphertext addition only
    agg = enc[0]
    for e in enc[1:]:
        agg = [a + b for a, b in zip(agg, e)]
    t_agg = time.perf_counter() - t

    t = time.perf_counter()  # key holder decrypts the sum
    for c in agg:
        c.link_context(secret_ctx)
    dec = np.concatenate([np.array(c.decrypt()) for c in agg])[:dim]
    t_dec = time.perf_counter() - t

    err = float(np.max(np.abs(dec - plain_sum)))
    return {
        "n_sites": n_sites, "dim": dim,
        "plain_bytes_per_site": dim * 4,
        "he_bytes_per_site": bytes_per_site,
        "expansion": bytes_per_site / (dim * 4),
        "keygen_s": t_keygen, "encrypt_s_per_site": t_enc, "aggregate_s": t_agg, "decrypt_s": t_dec,
        "max_abs_error": err,
    }
