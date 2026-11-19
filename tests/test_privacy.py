"""Secure aggregation and data-quality privacy properties."""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from fedmed.data import _write_labels
from fedmed.pki import load_key
from fedmed.quality import validate_site
from fedmed.secagg import RosterEntry, SecAggClient, decode_fixed, encode_fixed, unmask_sum, verify_roster

from .conftest import make_site


def _clients(n):
    cs = [SecAggClient(f"hospital-{chr(97 + i)}") for i in range(n)]
    return cs, [RosterEntry(c.site, c.public_key) for c in cs]


def test_fixed_point_roundtrip_handles_negatives():
    v = np.array([-3.5, 0.0, 1e-6, 1234.25], np.float32)
    np.testing.assert_allclose(decode_fixed(encode_fixed(v)), v, atol=1e-6)


@pytest.mark.parametrize("n", [2, 3, 7])
def test_masks_cancel_in_the_sum(n):
    rng = np.random.default_rng(n)
    vecs = [rng.normal(0, 5, 1000).astype(np.float32) for _ in range(n)]
    cs, roster = _clients(n)
    masked = [c.mask(v, roster, "run", 1) for c, v in zip(cs, vecs)]
    np.testing.assert_allclose(unmask_sum(masked), np.sum(vecs, 0), atol=1e-4)


def test_single_masked_update_reveals_nothing_useful():
    v = np.ones(2000, np.float32)
    cs, roster = _clients(3)
    single = decode_fixed(cs[0].mask(v, roster, "run", 1))
    # A masked vector is uniform noise over Z_2^64: no correlation with the plaintext
    assert abs(np.corrcoef(single, v + np.random.default_rng(0).normal(size=2000))[0, 1]) < 0.1
    assert np.abs(single).mean() > 1e6


def test_masks_change_every_round():
    cs, roster = _clients(2)
    v = np.zeros(10, np.float32)
    assert not np.array_equal(cs[0].mask(v, roster, "run", 1), cs[0].mask(v, roster, "run", 2))


def test_missing_site_breaks_unmasking():
    """No dropout recovery: documents the known limitation."""
    v = [np.ones(50, np.float32)] * 3
    cs, roster = _clients(3)
    masked = [c.mask(x, roster, "r", 1) for c, x in zip(cs, v)]
    assert np.abs(unmask_sum(masked[:2]) - 2).max() > 1e3


def _signed_roster(pki, sites):
    cs = [SecAggClient(s, signing_key=load_key(pki / f"{s}.key")) for s in sites]
    return cs, [RosterEntry(c.site, c.public_key, (pki / f"{c.site}.crt").read_bytes(), c.signature) for c in cs]


def test_roster_signed_with_mtls_keys_verifies(pki):
    _, roster = _signed_roster(pki, ["hospital-a", "hospital-b"])
    verify_roster(roster, (pki / "ca.crt").read_bytes())


def test_coordinator_cannot_substitute_keys(pki):
    """A malicious coordinator swaps hospital-b's key for one it controls (MITM)."""
    _, roster = _signed_roster(pki, ["hospital-a", "hospital-b"])
    attacker = SecAggClient("hospital-b")
    roster[1] = RosterEntry("hospital-b", attacker.public_key, roster[1].cert_pem, roster[1].signature)
    with pytest.raises(Exception):  # noqa: B017 - cryptography raises InvalidSignature
        verify_roster(roster, (pki / "ca.crt").read_bytes())


def test_roster_rejects_cert_for_other_site(pki):
    _, roster = _signed_roster(pki, ["hospital-a", "hospital-b"])
    roster[1] = RosterEntry("hospital-c", roster[1].public_key, roster[1].cert_pem, roster[1].signature)
    with pytest.raises(ValueError):
        verify_roster(roster, (pki / "ca.crt").read_bytes())


# ----------------------------------------------------------------- data quality
def test_quality_tool_finds_planted_defects(tmp_path):
    site = make_site(tmp_path, "hospital-x", 60, seed=3)
    imgs = site / "images"
    (imgs / "img_0000.png").write_bytes(b"\x89PNG broken")
    Image.fromarray(np.zeros((32, 32, 3), np.uint8) + 9).save(imgs / "img_0001.png")
    Image.fromarray(np.zeros((28, 28), np.uint8) + 9).save(imgs / "img_0002.png")
    (imgs / "copy.png").write_bytes((imgs / "img_0003.png").read_bytes())
    (imgs / "img_0004.png").unlink()
    rows = [r.split(",") for r in (site / "labels.csv").read_text().splitlines()[1:]]
    rows[5][1] = "42"
    rows.append(["copy.png", rows[3][1]])
    rows.append(["../../etc/passwd", "1"])
    _write_labels(site / "labels.csv", [tuple(r) for r in rows])

    res = validate_site(site, k_min=1)
    iss = res.local_report["issues"]
    assert iss["corrupt"] == ["img_0000.png", "../../etc/passwd"]
    assert iss["wrong_size"] == ["img_0001.png"] and iss["wrong_mode"] == ["img_0002.png"]
    assert iss["missing"] == ["img_0004.png"] and iss["bad_label"] == ["img_0005.png"]
    assert len(iss["exact_duplicate"]) == 1
    assert len(res.clean) == 60 - 5  # 5 broken originals; the duplicate and traversal rows were extra


def test_shared_quality_report_leaks_no_filenames_and_suppresses_small_cells(tmp_path):
    site = make_site(tmp_path, "hospital-y", 40, seed=4)
    (site / "images" / "img_0000.png").write_bytes(b"junk")
    res = validate_site(site, k_min=5)
    blob = json.dumps(res.shared_report)
    assert ".png" not in blob and "img_" not in blob
    assert res.shared_report["issue_counts"]["corrupt"] == "<5"
    assert all(isinstance(v, int) and (v == 0 or v >= 5) or v == "<5" for v in res.shared_report["label_histogram"])
