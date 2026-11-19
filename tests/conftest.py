from __future__ import annotations

import asyncio
import socket
import threading
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from fedmed.auth import issue_token
from fedmed.data import _write_labels
from fedmed.pki import init_pki

warnings.filterwarnings("ignore")
torch.set_num_threads(1)

SITES = ["hospital-a", "hospital-b", "hospital-c"]


def make_site(root: Path, name: str, n: int, seed: int) -> Path:
    """Synthetic site: class-dependent colour blobs, so a CNN can actually learn it."""
    rng = np.random.default_rng(seed)
    site = root / name
    (site / "images").mkdir(parents=True)
    rows = []
    for i in range(n):
        y = int(rng.integers(0, 7))
        img = rng.integers(0, 60, (28, 28, 3)).astype(np.uint8)
        img[4 * y : 4 * y + 4, :, y % 3] = 230
        Image.fromarray(img).save(site / "images" / f"img_{i:04d}.png")
        rows.append((f"img_{i:04d}.png", y))
    _write_labels(site / "labels.csv", rows)
    return site


@pytest.fixture(scope="session")
def pki(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("pki")
    init_pki(d, SITES, ["localhost", "127.0.0.1"], extra_clients=("auditor",))
    (d / "tokens").mkdir()
    for s in SITES:
        (d / "tokens" / f"{s}.jwt").write_text(issue_token(d / "token_signing.key", s, "trainer", d / f"{s}.crt"))
    (d / "tokens" / "auditor.jwt").write_text(
        issue_token(d / "token_signing.key", "auditor", "observer", d / "auditor.crt"))
    return d


@pytest.fixture(scope="session")
def sites(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("sites")
    for i, s in enumerate(SITES):
        make_site(root, s, 120, seed=i)
    return root


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ServerThread:
    """Runs the real aiohttp coordinator with its real mTLS context in a background thread."""

    def __init__(self, app, ssl_ctx):
        self.port = free_port()
        self.loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self.app, self.ssl_ctx = app, ssl_ctx
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        from aiohttp import web

        asyncio.set_event_loop(self.loop)
        runner = web.AppRunner(self.app)
        self.loop.run_until_complete(runner.setup())
        self.loop.run_until_complete(web.TCPSite(runner, "127.0.0.1", self.port, ssl_context=self.ssl_ctx).start())
        self._ready.set()
        self.loop.run_forever()

    def __enter__(self):
        self.thread.start()
        self._ready.wait(10)
        return self

    def __exit__(self, *exc):
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(5)

    @property
    def url(self) -> str:
        return f"https://localhost:{self.port}"
