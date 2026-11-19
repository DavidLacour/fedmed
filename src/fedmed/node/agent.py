"""Hospital node agent. Runs inside the hospital, next to the data.

1. Validate the local dataset and send only an aggregate, privacy-filtered summary.
2. (Secure aggregation) Register a signed X25519 key, then verify every peer's key
   against the federation CA before using it.
3. Each round: fetch the global model, evaluate it on the local validation split, train
   locally (optionally with DP-SGD), and upload a compressed or masked update.

All traffic is HTTPS with mutual TLS (TLS 1.3) and a certificate-bound bearer token.
"""

from __future__ import annotations

import base64
import logging
import ssl
import time
import warnings
import zlib
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np
import torch
from cryptography.x509.oid import NameOID

from .. import compression
from ..data import train_val_split
from ..pki import load_cert, load_key
from ..quality import validate_site
from ..secagg import RosterEntry, SecAggClient, verify_roster
from ..serialization import pack, unpack
from ..train import ClientState, DPConfig, LocalConfig, evaluate, local_train

log = logging.getLogger("fedmed.node")
logging.getLogger("httpx").setLevel(logging.WARNING)
API = "/api/v1"


@dataclass
class NodeConfig:
    server: str
    site_dir: Path
    cert: Path
    key: Path
    ca: Path
    token_file: Path
    poll_s: float = 1.0
    torch_threads: int = 2
    val_frac: float = 0.15
    timeout_s: float = 3600.0
    work_dir: Path | None = None  # where site-private reports go (data dir may be read-only)


def client_ssl_context(cfg: NodeConfig) -> ssl.SSLContext:
    ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile=str(cfg.ca))  # verifies hostname too
    ctx.minimum_version = ssl.TLSVersion.TLSv1_3
    ctx.load_cert_chain(str(cfg.cert), str(cfg.key))
    return ctx


class Node:
    def __init__(self, cfg: NodeConfig):
        self.cfg = cfg
        # identity comes from the certificate, never from paths or config
        self.site = load_cert(cfg.cert).subject.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value
        token = Path(cfg.token_file).read_text().strip()
        self.http = httpx.Client(
            base_url=cfg.server, verify=client_ssl_context(cfg), timeout=120.0,
            headers={"Authorization": f"Bearer {token}", "User-Agent": "fedmed-node/0.1"},
        )
        self.state = ClientState()
        self.secagg: SecAggClient | None = None
        self.roster: list[RosterEntry] = []

    def _get(self, path: str) -> httpx.Response:
        r = self.http.get(API + path)
        r.raise_for_status()
        return r

    def _post(self, path: str, **kw) -> httpx.Response:
        r = self.http.post(API + path, **kw)
        if r.status_code >= 400:
            raise RuntimeError(f"POST {path} -> {r.status_code}: {r.text}")
        return r

    # ---------------------------------------------------------------- phases
    def prepare_data(self):
        q = validate_site(self.cfg.site_dir, report_dir=self.cfg.work_dir, site_name=self.site)
        log.info("%s: %d clean images, %s excluded (details stay local)", self.site,
                 len(q.clean), q.local_report["n_excluded"])
        self._post("/quality", json=q.shared_report)
        self.train_data, self.val_data = train_val_split(q.clean, self.cfg.val_frac, seed=0)

    def register_secagg(self) -> None:
        self.secagg = SecAggClient(self.site, signing_key=load_key(self.cfg.key))
        self._post("/secagg/key", json={
            "public_key": base64.b64encode(self.secagg.public_key).decode(),
            "signature": base64.b64encode(self.secagg.signature).decode(),
        })
        while True:
            roster = self._get("/secagg/roster").json()
            if roster["complete"]:
                break
            time.sleep(self.cfg.poll_s)
        self.roster = [
            RosterEntry(e["site"], base64.b64decode(e["public_key"]), e["cert_pem"].encode(),
                        base64.b64decode(e["signature"]))
            for e in roster["entries"]
        ]
        verify_roster(self.roster, Path(self.cfg.ca).read_bytes())  # refuse a tampered roster
        log.info("%s: verified secagg roster of %d sites", self.site, len(self.roster))

    def run(self) -> None:
        torch.set_num_threads(self.cfg.torch_threads)
        warnings.filterwarnings("ignore")
        self.prepare_data()
        done_round, t0 = 0, time.time()
        while time.time() - t0 < self.cfg.timeout_s:
            task = self._get("/task").json()
            if task["phase"] == "finished":
                log.info("%s: federation finished after %d rounds", self.site, done_round)
                return
            if task["phase"] == "waiting_keys" and task["train"]["secagg"] and self.secagg is None:
                self.register_secagg()
                continue
            if task["phase"] != "training" or task["round"] <= done_round or self.site in task["submitted"]:
                time.sleep(self.cfg.poll_s)
                continue
            self.train_round(task)
            done_round = task["round"]
        raise TimeoutError("federation did not finish in time")

    def train_round(self, task: dict) -> None:
        t, rnd = task["train"], task["round"]
        meta, arrs = unpack(self._get("/model").content)
        if meta["round"] != rnd:
            return
        w = arrs["w"]
        ev = evaluate(w, self.val_data)  # federated evaluation of the incoming global model
        dp = DPConfig(**t["dp"]) if t.get("dp") else None
        lcfg = LocalConfig(algorithm=t["algorithm"], lr=t["lr"], batch_size=t["batch_size"],
                           local_steps=t["local_steps"], mu=t["mu"], dp=dp, class_weights=t["class_weights"])
        tic = time.time()
        res = local_train(w, self.train_data, lcfg, self.state, seed=rnd * 7919 + zlib.crc32(self.site.encode()) % 1000,
                          c_global=arrs.get("c_global"))
        if t["secagg"]:
            parts = [res.delta * res.n_samples, np.array([res.n_samples], np.float32)]
            if res.delta_c is not None:
                parts.append(res.delta_c)
            masked = self.secagg.mask(np.concatenate(parts), self.roster, task["run_id"], rnd)
            body = pack({"run_id": task["run_id"], "round": rnd}, {"masked": masked})
        else:
            arrays, cmeta, self.state.residual = compression.encode(res.delta, t["compression"], self.state.residual)
            extra = {"delta_c": res.delta_c} if res.delta_c is not None else {}
            body = pack({"run_id": task["run_id"], "round": rnd, "n": res.n_samples, "comp": cmeta},
                        {**arrays, **extra})
        self._post("/metrics", json={"round": rnd, "accuracy": ev["accuracy"], "n": ev["n"],
                                     "balanced_accuracy": ev["balanced_accuracy"], "epsilon": res.epsilon,
                                     "train_loss": res.train_loss})
        self._post("/update", content=body, headers={"Content-Type": "application/octet-stream"})
        eps = f" eps={res.epsilon:.2f}" if res.epsilon is not None else ""
        log.info("%s: round %d  local-val acc of global model=%.3f  train %.1fs  sent %.1f kB%s", self.site, rnd,
                 ev["accuracy"], time.time() - tic, len(body) / 1e3, eps)
