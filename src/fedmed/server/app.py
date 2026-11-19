"""Federation coordinator: an aiohttp service that runs only behind mTLS.

It holds the global model and orchestrates rounds. It never receives images, file names
or labels. It receives model updates (masked when secure aggregation is on), local
evaluation metrics, and privacy-filtered data-quality summaries.

Phases:  waiting_quality -> [waiting_keys] -> training -> finished
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import ssl
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import yaml
from aiohttp import web
from cryptography import x509
from cryptography.hazmat.primitives import serialization

from .. import NUM_CLASSES, compression
from ..auth import Authenticator, AuthError, Policy, Principal
from ..data import ArraySplit
from ..secagg import unmask_sum
from ..serialization import pack, unpack
from ..strategies import Update, make_strategy
from ..train import class_weights_from_counts, evaluate, initial_weights

log = logging.getLogger("fedmed.server")
API = "/api/v1"
PRINCIPAL = web.RequestKey("principal", Principal)

# (method, path) -> required permission; None = any authenticated principal
ROUTES: dict[tuple[str, str], str | None] = {
    ("GET", f"{API}/health"): None,
    ("GET", f"{API}/task"): "task:read",
    ("GET", f"{API}/model"): "model:read",
    ("POST", f"{API}/quality"): "quality:submit",
    ("POST", f"{API}/secagg/key"): "secagg:register",
    ("GET", f"{API}/secagg/roster"): "secagg:read",
    ("POST", f"{API}/update"): "update:submit",
    ("POST", f"{API}/metrics"): "metrics:submit",
    ("GET", f"{API}/status"): "status:read",
}
QUALITY_KEYS = {"site", "n_clean", "n_excluded", "issue_counts", "label_histogram", "pixel_mean_rgb",
                "privacy", "pass"}


@dataclass
class FederationConfig:
    run_id: str = "derm-poc"
    participants: list[str] = field(default_factory=lambda: [f"hospital-{c}" for c in "abc"])
    rounds: int = 20
    algorithm: str = "fedavg"
    mu: float = 0.01
    lr: float = 0.1
    batch_size: int = 32
    local_steps: int = 50
    compression: str = "none"
    secagg: bool = False
    dp: dict | None = None  # {"noise_multiplier":1.0, "max_grad_norm":1.0, "delta":1e-5}
    class_weight_power: float | None = 0.5
    max_update_norm: float = 200.0  # reject absurd (buggy or malicious) updates; unusable with secagg
    seed: int = 0
    benchmark: str | None = None  # optional public test set held by the coordinator
    out_dir: str = "runs/coordinator"

    @classmethod
    def load(cls, path: Path) -> FederationConfig:
        return cls(**(yaml.safe_load(Path(path).read_text()) or {}))


class Coordinator:
    def __init__(self, cfg: FederationConfig):
        self.cfg = cfg
        self.w = initial_weights(cfg.seed)
        self.dim = len(self.w)
        self.strategy = make_strategy(cfg.algorithm, self.dim, len(cfg.participants))
        self.round = 0
        self.phase = "waiting_quality"
        self.quality: dict[str, dict] = {}
        self.roster: dict[str, dict] = {}
        self.pending: dict[str, object] = {}
        self.client_metrics: dict[int, dict[str, dict]] = {}
        self.history: list[dict] = []
        self.class_weights: list[float] | None = None
        self.lock = asyncio.Lock()
        self.bytes_in = 0
        self.bytes_out = 0
        self.t_round = time.time()
        self.benchmark = None
        if cfg.benchmark and Path(cfg.benchmark).exists():
            d = np.load(cfg.benchmark)
            self.benchmark = ArraySplit(d["x"], d["y"].astype(np.int64))
        self.out = Path(cfg.out_dir)
        self.out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ phases
    def _advance_phase(self) -> None:
        parts = set(self.cfg.participants)
        if self.phase == "waiting_quality" and parts <= set(self.quality):
            counts = np.zeros(NUM_CLASSES)
            for r in self.quality.values():  # "<k" cells are unknown small counts -> 0
                counts += [v if isinstance(v, (int, float)) else 0 for v in r["label_histogram"]]
            if self.cfg.class_weight_power:
                self.class_weights = class_weights_from_counts(counts, self.cfg.class_weight_power)
            self.phase = "waiting_keys" if self.cfg.secagg else "training"
            log.info("all quality reports in; phase -> %s", self.phase)
        if self.phase == "waiting_keys" and parts <= set(self.roster):
            self.phase = "training"
            log.info("secagg roster complete")
        if self.phase == "training" and self.round == 0:
            self.round = 1
            self.t_round = time.time()

    def task(self) -> dict:
        c = self.cfg
        return {
            "run_id": c.run_id, "phase": self.phase, "round": self.round, "total_rounds": c.rounds,
            "participants": c.participants,
            "submitted": sorted(self.pending),
            "train": {
                "algorithm": c.algorithm, "mu": c.mu, "lr": c.lr, "batch_size": c.batch_size,
                "local_steps": c.local_steps, "dp": c.dp, "compression": c.compression,
                "secagg": c.secagg, "class_weights": self.class_weights, "dim": self.dim,
            },
        }

    def model_bytes(self) -> bytes:
        return pack({"run_id": self.cfg.run_id, "round": self.round},
                    {"w": self.w, **self.strategy.client_payload()})

    # ------------------------------------------------------------------ updates
    def accept_update(self, site: str, body: bytes) -> None:
        meta, arrays = unpack(body)
        if meta.get("run_id") != self.cfg.run_id or meta.get("round") != self.round:
            raise AuthError(409, f"update is for round {meta.get('round')}, current round is {self.round}")
        if site in self.pending:
            raise AuthError(409, "update already submitted for this round")
        extra = 1 + (self.dim if self.cfg.algorithm == "scaffold" else 0)
        if self.cfg.secagg:
            m = arrays.get("masked")
            if m is None or m.dtype != np.uint64 or m.shape != (self.dim + extra,):
                raise AuthError(422, "malformed masked update")
            self.pending[site] = m
        else:
            delta = compression.decode(arrays, meta["comp"])
            n = float(meta["n"])
            if delta.shape != (self.dim,) or not np.isfinite(delta).all() or not (0 < n < 1e7):
                raise AuthError(422, "malformed update")
            norm = float(np.linalg.norm(delta))
            if norm > self.cfg.max_update_norm:
                raise AuthError(422, f"update norm {norm:.1f} exceeds bound")
            dc = arrays.get("delta_c")
            if self.cfg.algorithm == "scaffold" and (dc is None or dc.shape != (self.dim,)):
                raise AuthError(422, "missing control variate")
            self.pending[site] = Update(delta * n, n, dc)
        self.bytes_in += len(body)

    def ready(self) -> bool:
        return self.phase == "training" and set(self.cfg.participants) <= set(self.pending)

    def aggregate(self) -> dict:
        """Blocking; called in a worker thread."""
        if self.cfg.secagg:
            total = unmask_sum([self.pending[s] for s in self.cfg.participants])
            d = self.dim
            dc = total[d + 1:] if self.cfg.algorithm == "scaffold" else None
            updates = [Update(total[:d], float(total[d]), dc, n_clients=len(self.cfg.participants))]
        else:
            updates = list(self.pending.values())
        self.w = self.strategy.aggregate(self.w, updates)
        rec = {"round": self.round, "time_s": round(time.time() - self.t_round, 2), "bytes_in": self.bytes_in}
        if self.benchmark is not None:
            rec["benchmark"] = evaluate(self.w, self.benchmark)
        site_m = self.client_metrics.get(self.round, {})
        if site_m:
            n = sum(m["n"] for m in site_m.values())
            rec["federated_eval"] = {
                k: sum(m[k] * m["n"] for m in site_m.values()) / n for k in ("accuracy", "balanced_accuracy")
            }
            eps = [m["epsilon"] for m in site_m.values() if m.get("epsilon") is not None]
            rec["max_epsilon"] = max(eps) if eps else None
        return rec

    def finish_round(self, rec: dict) -> None:
        self.history.append(rec)
        self.pending.clear()
        b = rec.get("benchmark", {})
        log.info("round %d aggregated  benchmark acc=%s bal_acc=%s", self.round,
                 f"{b.get('accuracy', float('nan')):.3f}", f"{b.get('balanced_accuracy', float('nan')):.3f}")
        if self.round >= self.cfg.rounds:
            self.phase = "finished"
            np.save(self.out / "global_model.npy", self.w)
        else:
            self.round += 1
            self.t_round = time.time()
        (self.out / "history.json").write_text(json.dumps(
            {"config": asdict(self.cfg), "quality": self.quality, "history": self.history}, indent=1))


# ====================================================================== HTTP layer
def _peer_cert_der(request: web.Request) -> bytes | None:
    sslobj = request.transport.get_extra_info("ssl_object") if request.transport else None
    return sslobj.getpeercert(binary_form=True) if sslobj else None


def make_app(coord: Coordinator, authn: Authenticator, audit_path: Path | None = None) -> web.Application:
    audit = open(audit_path, "a", buffering=1) if audit_path else None  # noqa: SIM115

    def _audit(request: web.Request, site: str | None, status: int, reason: str = "") -> None:
        if audit:
            audit.write(json.dumps({"ts": time.time(), "site": site, "method": request.method,
                                    "path": request.path, "status": status, "reason": reason}) + "\n")

    @web.middleware
    async def security(request: web.Request, handler):
        key = (request.method, request.path)
        if key not in ROUTES:
            _audit(request, None, 404)
            return web.json_response({"error": "not found"}, status=404)
        principal: Principal | None = None
        try:
            principal = authn.authenticate(_peer_cert_der(request), request.headers.get("Authorization"))
            if ROUTES[key]:
                authn.authorise(principal, ROUTES[key])
            request[PRINCIPAL] = principal
            resp = await handler(request)
        except AuthError as e:
            status, reason = e.status, e.reason
        except (ValueError, KeyError, TypeError) as e:  # malformed input; never echo internals
            status, reason = 400, f"bad request ({type(e).__name__})"
        else:
            _audit(request, principal.site, resp.status)
            return resp
        site = principal.site if principal else None
        _audit(request, site, status, reason)
        log.warning("%s %s from %s -> %d %s", request.method, request.path, site, status, reason)
        return web.json_response({"error": reason}, status=status)

    async def health(_: web.Request):
        return web.json_response({"ok": True})

    async def task(_: web.Request):
        async with coord.lock:
            return web.json_response(coord.task())

    async def model(_: web.Request):
        async with coord.lock:
            if coord.phase != "training" and coord.phase != "finished":
                raise AuthError(409, f"no model available in phase {coord.phase}")
            body = coord.model_bytes()
            coord.bytes_out += len(body)
        return web.Response(body=body, content_type="application/octet-stream")

    async def quality(request: web.Request):
        p = request[PRINCIPAL]
        rep = await request.json()
        if not isinstance(rep, dict) or set(rep) - QUALITY_KEYS or rep.get("site") != p.site:
            raise AuthError(422, "quality report must be an aggregate summary for the calling site")
        if len(rep.get("label_histogram", [])) != NUM_CLASSES:
            raise AuthError(422, "bad label histogram")
        async with coord.lock:
            coord.quality[p.site] = rep
            coord._advance_phase()
        return web.json_response({"accepted": True, "pass": rep.get("pass")})

    async def secagg_key(request: web.Request):
        p = request[PRINCIPAL]
        if not coord.cfg.secagg:
            raise AuthError(409, "secure aggregation disabled")
        body = await request.json()
        pk, sig = base64.b64decode(body["public_key"]), base64.b64decode(body["signature"])
        if len(pk) != 32 or not 8 <= len(sig) <= 128:
            raise AuthError(422, "bad X25519 key or signature")
        cert_pem = x509.load_der_x509_certificate(_peer_cert_der(request)).public_bytes(serialization.Encoding.PEM)
        async with coord.lock:
            if p.site in coord.roster and coord.phase != "waiting_keys":
                raise AuthError(409, "roster frozen")
            coord.roster[p.site] = {"site": p.site, "public_key": body["public_key"],
                                    "signature": body["signature"], "cert_pem": cert_pem.decode()}
            coord._advance_phase()
        return web.json_response({"accepted": True})

    async def roster(_: web.Request):
        async with coord.lock:
            complete = set(coord.cfg.participants) <= set(coord.roster)
            return web.json_response({"complete": complete, "entries": list(coord.roster.values())})

    async def update(request: web.Request):
        p = request[PRINCIPAL]
        if p.site not in coord.cfg.participants:
            raise AuthError(403, "site is not a participant in this run")
        body = await request.read()
        async with coord.lock:
            if coord.phase != "training":
                raise AuthError(409, f"not accepting updates in phase {coord.phase}")
            coord.accept_update(p.site, body)
            if coord.ready():
                rec = await asyncio.get_running_loop().run_in_executor(None, coord.aggregate)
                coord.finish_round(rec)
        return web.json_response({"accepted": True})

    async def metrics(request: web.Request):
        p = request[PRINCIPAL]
        m = await request.json()
        allowed = {"round", "accuracy", "balanced_accuracy", "loss", "n", "epsilon", "train_loss", "macro_f1"}
        if set(m) - allowed:
            raise AuthError(422, "unexpected metric fields")
        async with coord.lock:
            coord.client_metrics.setdefault(int(m["round"]), {})[p.site] = m
        return web.json_response({"accepted": True})

    async def status(_: web.Request):
        async with coord.lock:
            return web.json_response({**coord.task(), "quality": coord.quality, "history": coord.history,
                                      "bytes_in": coord.bytes_in, "bytes_out": coord.bytes_out})

    app = web.Application(middlewares=[security], client_max_size=64 * 1024 * 1024)
    app.add_routes([
        web.get(f"{API}/health", health), web.get(f"{API}/task", task), web.get(f"{API}/model", model),
        web.post(f"{API}/quality", quality), web.post(f"{API}/secagg/key", secagg_key),
        web.get(f"{API}/secagg/roster", roster), web.post(f"{API}/update", update),
        web.post(f"{API}/metrics", metrics), web.get(f"{API}/status", status),
    ])
    return app


def server_ssl_context(pki: Path) -> ssl.SSLContext:
    """TLS 1.3 only, client certificate required, CRL checked for the client leaf."""
    pki = Path(pki)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_3
    ctx.load_cert_chain(pki / "server.crt", pki / "server.key")
    ctx.load_verify_locations(cafile=pki / "ca.crt")
    ctx.verify_mode = ssl.CERT_REQUIRED
    if (pki / "crl.pem").exists():
        ctx.load_verify_locations(cafile=pki / "crl.pem")
        ctx.verify_flags |= ssl.VERIFY_CRL_CHECK_LEAF
    return ctx


def serve(config: Path, pki: Path, policy: Path | None, host: str = "0.0.0.0", port: int = 8443) -> None:  # noqa: S104
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = FederationConfig.load(config)
    coord = Coordinator(cfg)
    pol = Policy.load(policy, default_sites={s: "trainer" for s in cfg.participants})
    authn = Authenticator(Path(pki) / "token_signing.pub", pol)
    app = make_app(coord, authn, audit_path=Path(cfg.out_dir) / "audit.jsonl")
    log.info("coordinator %s: %d participants, %d rounds, %s, secagg=%s, dp=%s",
             cfg.run_id, len(cfg.participants), cfg.rounds, cfg.algorithm, cfg.secagg, cfg.dp)
    web.run_app(app, host=host, port=port, ssl_context=server_ssl_context(pki), access_log=None)
