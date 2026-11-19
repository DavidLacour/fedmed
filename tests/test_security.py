"""Network security: mTLS, certificate-bound tokens, authorisation, revocation, and a full
federated run with secure aggregation against the real coordinator over TLS 1.3."""

from __future__ import annotations

import ssl
import threading
import time

import httpx
import jwt
import numpy as np
import pytest

from fedmed.auth import Authenticator, Policy, issue_token
from fedmed.node.agent import Node, NodeConfig
from fedmed.pki import init_pki, revoke
from fedmed.serialization import pack
from fedmed.server.app import Coordinator, FederationConfig, make_app, server_ssl_context

from .conftest import SITES, ServerThread


def start(pki, tmp_path, **cfg):
    fc = FederationConfig(participants=SITES, out_dir=str(tmp_path / "coord"), **cfg)
    coord = Coordinator(fc)
    policy = Policy.load(None, default_sites={**{s: "trainer" for s in SITES}, "auditor": "observer"})
    app = make_app(coord, Authenticator(pki / "token_signing.pub", policy), tmp_path / "audit.jsonl")
    return coord, ServerThread(app, server_ssl_context(pki))


def client(pki, site=None, token=None, ca=None):
    ctx = ssl.create_default_context(cafile=str(ca or pki / "ca.crt"))
    if site:
        ctx.load_cert_chain(pki / f"{site}.crt", pki / f"{site}.key")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return ctx, headers


def get(srv, pki, path, site=None, token=None, ca=None):
    ctx, h = client(pki, site, token, ca)
    with httpx.Client(verify=ctx, headers=h) as c:
        return c.get(srv.url + "/api/v1" + path)


def tok(pki, site):
    return (pki / "tokens" / f"{site}.jwt").read_text()


# ------------------------------------------------------------------- transport layer
def test_no_client_certificate_fails_handshake(pki, tmp_path):
    _, srv = start(pki, tmp_path)
    with srv, pytest.raises((httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError)):
        get(srv, pki, "/health")


def test_certificate_from_rogue_ca_is_rejected(pki, tmp_path):
    rogue = tmp_path / "rogue"
    init_pki(rogue, ["hospital-a"], ["localhost"])  # same names, different CA
    _, srv = start(pki, tmp_path)
    ctx = ssl.create_default_context(cafile=str(pki / "ca.crt"))
    ctx.load_cert_chain(rogue / "hospital-a.crt", rogue / "hospital-a.key")
    with srv, pytest.raises((httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError)):
        httpx.get(srv.url + "/api/v1/health", verify=ctx)


def test_client_refuses_impostor_server(pki, tmp_path):
    rogue = tmp_path / "rogue2"
    init_pki(rogue, ["hospital-a"], ["localhost"])
    coord = Coordinator(FederationConfig(participants=SITES, out_dir=str(tmp_path / "c")))
    app = make_app(coord, Authenticator(pki / "token_signing.pub", Policy.load(None)))
    with ServerThread(app, server_ssl_context(rogue)) as srv, pytest.raises(httpx.ConnectError):
        get(srv, pki, "/health", "hospital-a", tok(pki, "hospital-a"))


def test_revoked_certificate_is_rejected(tmp_path):
    pki = tmp_path / "pki"
    init_pki(pki, ["hospital-a", "hospital-b"], ["localhost"])
    revoke(pki, pki / "hospital-b.crt")
    _, srv = start(pki, tmp_path)
    ok_tok = issue_token(pki / "token_signing.key", "hospital-a", "trainer", pki / "hospital-a.crt")
    bad_tok = issue_token(pki / "token_signing.key", "hospital-b", "trainer", pki / "hospital-b.crt")
    with srv:
        assert get(srv, pki, "/health", "hospital-a", ok_tok).status_code == 200
        with pytest.raises((httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError)):
            get(srv, pki, "/health", "hospital-b", bad_tok)


# ------------------------------------------------------------------- application layer
def test_valid_cert_and_token(pki, tmp_path):
    _, srv = start(pki, tmp_path)
    with srv:
        r = get(srv, pki, "/task", "hospital-a", tok(pki, "hospital-a"))
    assert r.status_code == 200 and r.json()["phase"] == "waiting_quality"


def test_missing_token_is_401(pki, tmp_path):
    _, srv = start(pki, tmp_path)
    with srv:
        assert get(srv, pki, "/task", "hospital-a").status_code == 401


def test_stolen_token_is_useless_with_another_certificate(pki, tmp_path):
    """hospital-b presents hospital-a's token: the token is bound to a's certificate."""
    _, srv = start(pki, tmp_path)
    with srv:
        r = get(srv, pki, "/task", "hospital-b", tok(pki, "hospital-a"))
    assert r.status_code == 401 and "certificate" in r.json()["error"]


def test_forged_token_is_rejected(pki, tmp_path):
    _, srv = start(pki, tmp_path)
    claims = jwt.decode(tok(pki, "hospital-a"), options={"verify_signature": False})
    claims["role"] = "admin"
    forged = jwt.encode(claims, "guessed-secret", algorithm="HS256")  # alg confusion attempt
    with srv:
        assert get(srv, pki, "/task", "hospital-a", forged).status_code == 401


def test_expired_token_is_rejected(pki, tmp_path):
    _, srv = start(pki, tmp_path)
    old = issue_token(pki / "token_signing.key", "hospital-a", "trainer", pki / "hospital-a.crt", ttl_hours=-1)
    with srv:
        assert get(srv, pki, "/task", "hospital-a", old).status_code == 401


def test_observer_can_read_status_but_not_model(pki, tmp_path):
    _, srv = start(pki, tmp_path)
    with srv:
        assert get(srv, pki, "/status", "auditor", tok(pki, "auditor")).status_code == 200
        assert get(srv, pki, "/model", "auditor", tok(pki, "auditor")).status_code == 403
        assert get(srv, pki, "/status", "hospital-a", tok(pki, "hospital-a")).status_code == 403


def test_audit_log_records_denials(pki, tmp_path):
    _, srv = start(pki, tmp_path)
    with srv:
        get(srv, pki, "/model", "auditor", tok(pki, "auditor"))
    log = (tmp_path / "audit.jsonl").read_text()
    assert '"site": "auditor"' in log and '"status": 403' in log


def test_update_validation(pki, tmp_path):
    coord, srv = start(pki, tmp_path, max_update_norm=1.0)
    coord.phase, coord.round = "training", 1
    ctx, h = client(pki, "hospital-a", tok(pki, "hospital-a"))
    huge = pack({"run_id": coord.cfg.run_id, "round": 1, "n": 10, "comp": {"c": "none", "d": coord.dim}},
                {"v": np.full(coord.dim, 5.0, np.float32)})
    stale = pack({"run_id": coord.cfg.run_id, "round": 7, "n": 10, "comp": {"c": "none", "d": coord.dim}},
                 {"v": np.zeros(coord.dim, np.float32)})
    with srv, httpx.Client(verify=ctx, headers=h) as c:
        assert c.post(srv.url + "/api/v1/update", content=huge).status_code == 422
        assert c.post(srv.url + "/api/v1/update", content=stale).status_code == 409
        assert c.post(srv.url + "/api/v1/update", content=b"\x80\x04pickle").status_code == 400


# ------------------------------------------------------------------- end to end
@pytest.mark.parametrize("algorithm,secagg,dp", [("fedavg", True, None), ("scaffold", False, {"noise_multiplier": 1.0})])
def test_full_federation_over_mtls(pki, sites, tmp_path, algorithm, secagg, dp):
    coord, srv = start(pki, tmp_path, rounds=2, local_steps=3, algorithm=algorithm, secagg=secagg, dp=dp,
                       compression="int8")
    errors = []

    def run_node(site):
        try:
            Node(NodeConfig(srv.url, sites / site, pki / f"{site}.crt", pki / f"{site}.key", pki / "ca.crt",
                            pki / "tokens" / f"{site}.jwt", poll_s=0.1, torch_threads=1, timeout_s=120)).run()
        except Exception as e:  # surfaced below
            errors.append(e)

    with srv:
        threads = [threading.Thread(target=run_node, args=(s,)) for s in SITES]
        for t in threads:
            t.start()
        for t in threads:
            t.join(150)
        time.sleep(0.2)
    assert not errors, errors
    assert coord.phase == "finished" and len(coord.history) == 2
    assert set(coord.quality) == set(SITES)
    if dp:
        assert coord.history[-1]["max_epsilon"] > 0
