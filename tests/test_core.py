"""Unit tests for partitioning, wire format, compression, aggregation and local training."""

from __future__ import annotations

import numpy as np
import pytest

from fedmed import compression
from fedmed.data import ArraySplit, dirichlet_partition
from fedmed.serialization import pack, unpack
from fedmed.strategies import FedAvg, Scaffold, Update
from fedmed.train import (
    ClientState,
    DPConfig,
    LocalConfig,
    class_weights_from_counts,
    initial_weights,
    local_train,
)


# ----------------------------------------------------------------- partitioning
def test_dirichlet_partition_covers_everything_once():
    y = np.repeat(np.arange(7), 200)
    parts = dirichlet_partition(y, 5, alpha=0.3, seed=1)
    allidx = np.concatenate(parts)
    assert len(allidx) == len(y) and len(np.unique(allidx)) == len(y)


def test_small_alpha_is_more_skewed_than_large_alpha():
    y = np.repeat(np.arange(7), 300)

    def skew(alpha):
        parts = dirichlet_partition(y, 5, alpha, seed=0)
        dists = [np.bincount(y[p], minlength=7) / len(p) for p in parts]
        return np.mean([np.abs(d - 1 / 7).sum() for d in dists])

    assert skew(0.1) > 2 * skew(100.0)


# ----------------------------------------------------------------- wire format
def test_pack_roundtrip():
    arrs = {"w": np.random.randn(100).astype(np.float32), "m": np.arange(5, dtype=np.uint64)}
    meta, out = unpack(pack({"round": 3}, arrs))
    assert meta == {"round": 3}
    for k in arrs:
        np.testing.assert_array_equal(out[k], arrs[k])


def test_pack_refuses_object_arrays_and_unpack_rejects_tampering():
    with pytest.raises(TypeError):
        pack({}, {"x": np.array([object()])})
    buf = bytearray(pack({}, {"w": np.zeros(4, np.float32)}))
    with pytest.raises(ValueError):
        unpack(bytes(buf[:-4]))  # truncated body
    with pytest.raises(ValueError):
        unpack(b"PK\x03\x04" + bytes(buf[4:]))  # not our format (e.g. a pickle/zip)


# ----------------------------------------------------------------- compression
@pytest.mark.parametrize("method,tol", [("none", 0), ("fp16", 1e-3), ("int8", 0.02)])
def test_dense_compression_error(method, tol):
    v = np.random.default_rng(0).normal(0, 0.1, 5000).astype(np.float32)
    arrays, meta, _ = compression.encode(v, method)
    meta2, arrays2 = unpack(pack(meta, arrays))
    assert np.max(np.abs(compression.decode(arrays2, meta2) - v)) <= tol * np.abs(v).max() + 1e-9


def test_topk_error_feedback_preserves_mass():
    v = np.random.default_rng(1).normal(size=1000).astype(np.float32)
    arrays, meta, res = compression.encode(v, "topk:0.05")
    dec = compression.decode(arrays, meta)
    assert (dec != 0).sum() == 50
    np.testing.assert_allclose(dec + res, v, atol=1e-6)  # nothing is lost, only delayed


def test_topk_rejects_out_of_range_index():
    with pytest.raises(ValueError):
        compression.decode({"i": np.array([10], np.uint32), "v": np.ones(1, np.float32)}, {"c": "topk", "d": 5})


# ----------------------------------------------------------------- aggregation
def test_fedavg_is_sample_weighted():
    w = np.zeros(3, np.float32)
    ups = [Update(np.ones(3) * 1 * 10, 10), Update(np.ones(3) * 4 * 30, 30)]
    np.testing.assert_allclose(FedAvg().aggregate(w, ups), np.full(3, (10 * 1 + 30 * 4) / 40))


def test_scaffold_updates_server_control_variate():
    s = Scaffold(dim=3, n_total_clients=4)
    s.aggregate(np.zeros(3, np.float32), [Update(np.zeros(3), 1, delta_c=np.ones(3, np.float32) * 2)])
    np.testing.assert_allclose(s.c, np.full(3, 0.5))


def test_class_weights_upweight_rare_classes():
    w = class_weights_from_counts([1000, 10, 100])
    assert w[1] > w[2] > w[0] and abs(np.mean(w) - 1) < 1e-3


# ----------------------------------------------------------------- local training
@pytest.fixture(scope="module")
def tiny():
    rng = np.random.default_rng(0)
    return ArraySplit(rng.integers(0, 255, (64, 28, 28, 3), dtype=np.uint8), rng.integers(0, 7, 64))


@pytest.mark.parametrize("alg", ["fedavg", "fedprox", "scaffold"])
def test_local_train_runs(alg, tiny):
    w0 = initial_weights()
    res = local_train(w0, tiny, LocalConfig(algorithm=alg, local_steps=3, batch_size=16), ClientState())
    assert res.delta.shape == w0.shape and np.isfinite(res.delta).all() and res.steps == 3
    assert (res.delta_c is not None) == (alg == "scaffold")


def test_fedprox_large_mu_stays_close_to_global(tiny):
    w0 = initial_weights()
    free = local_train(w0, tiny, LocalConfig("fedavg", local_steps=5, lr=0.1), ClientState(), seed=1)
    prox = local_train(w0, tiny, LocalConfig("fedprox", local_steps=5, lr=0.1, mu=50.0), ClientState(), seed=1)
    assert np.linalg.norm(prox.delta) < 0.5 * np.linalg.norm(free.delta)


def test_dp_budget_accumulates_across_rounds(tiny):
    st = ClientState()
    cfg = LocalConfig(local_steps=4, batch_size=16, dp=DPConfig(noise_multiplier=1.0))
    w = initial_weights()
    e1 = local_train(w, tiny, cfg, st).epsilon
    e2 = local_train(w, tiny, cfg, st).epsilon
    assert e1 is not None and e2 > e1 > 0


def test_more_noise_means_smaller_epsilon(tiny):
    w = initial_weights()
    lo = local_train(w, tiny, LocalConfig(local_steps=4, dp=DPConfig(0.7)), ClientState()).epsilon
    hi = local_train(w, tiny, LocalConfig(local_steps=4, dp=DPConfig(2.0)), ClientState()).epsilon
    assert hi < lo
