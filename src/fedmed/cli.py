"""Command-line entry point: ``fedmed <command>``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="fedmed", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare-data", help="download DermaMNIST and split it into hospital folders")
    p.add_argument("--out", default="data/sites")
    p.add_argument("--sites", type=int, default=5)
    p.add_argument("--alpha", type=float, default=0.5, help="Dirichlet label skew (small = more non-IID)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-faults", action="store_true", help="do not plant data-quality defects")

    p = sub.add_parser("quality", help="validate a site's data; prints the shareable summary")
    p.add_argument("site_dir")
    p.add_argument("--k-min", type=int, default=5)
    p.add_argument("--dp-epsilon", type=float, default=None)

    p = sub.add_parser("pki", help="certificate authority operations")
    psub = p.add_subparsers(dest="pki_cmd", required=True)
    q = psub.add_parser("init")
    q.add_argument("--out", default="certs")
    q.add_argument("--sites", nargs="+", required=True)
    q.add_argument("--extra-clients", nargs="*", default=["auditor"])
    q.add_argument("--server-san", nargs="+", default=["localhost", "127.0.0.1", "coordinator"])
    q = psub.add_parser("revoke")
    q.add_argument("--pki", default="certs")
    q.add_argument("cert")

    p = sub.add_parser("token", help="issue a certificate-bound access token")
    p.add_argument("--pki", default="certs")
    p.add_argument("--site", required=True)
    p.add_argument("--role", default="trainer")
    p.add_argument("--ttl-hours", type=float, default=72)

    p = sub.add_parser("server", help="run the federation coordinator (mTLS)")
    p.add_argument("--config", required=True)
    p.add_argument("--pki", default="certs")
    p.add_argument("--policy", default=None)
    p.add_argument("--host", default="0.0.0.0")  # noqa: S104 - container default
    p.add_argument("--port", type=int, default=8443)

    p = sub.add_parser("node", help="run a hospital node")
    p.add_argument("--server", required=True, help="https://host:8443")
    p.add_argument("--site-dir", required=True)
    p.add_argument("--cert", required=True)
    p.add_argument("--key", required=True)
    p.add_argument("--ca", required=True)
    p.add_argument("--token-file", required=True)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--work-dir", default=None, help="site-private output (quality report)")

    p = sub.add_parser("simulate", help="in-process federated run (benchmarks)")
    p.add_argument("--algorithm", default="fedavg", choices=["fedavg", "fedprox", "scaffold"])
    p.add_argument("--rounds", type=int, default=30)
    p.add_argument("--sites", type=int, default=5)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--noise-multiplier", type=float, default=None)
    p.add_argument("--compression", default="none")
    p.add_argument("--secagg", action="store_true")
    p.add_argument("--out", default=None)

    sub.add_parser("he-demo", help="CKKS homomorphic aggregation benchmark (needs tenseal)")

    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if a.cmd == "prepare-data":
        from .data import DEFAULT_RAW, download_dermamnist, export_sites

        raw = download_dermamnist(DEFAULT_RAW)
        s = export_sites(raw, Path(a.out), a.sites, a.alpha, a.seed, inject_faults=not a.no_faults)
        print(json.dumps(s, indent=2))
    elif a.cmd == "quality":
        from .quality import validate_site

        r = validate_site(Path(a.site_dir), k_min=a.k_min, dp_epsilon=a.dp_epsilon)
        print(json.dumps(r.shared_report, indent=2))
        print(f"# detailed report (stays on site): {Path(a.site_dir) / 'local_quality_report.json'}",
              file=sys.stderr)
    elif a.cmd == "pki":
        from .pki import init_pki, revoke

        if a.pki_cmd == "init":
            init_pki(Path(a.out), a.sites, a.server_san, tuple(a.extra_clients))
            print(f"PKI written to {a.out}/ (keep ca.key and token_signing.key offline)")
        else:
            print(f"revoked serial {revoke(Path(a.pki), Path(a.cert)):x}; restart the coordinator to reload the CRL")
    elif a.cmd == "token":
        from .auth import issue_token

        print(issue_token(Path(a.pki) / "token_signing.key", a.site, a.role, Path(a.pki) / f"{a.site}.crt",
                          a.ttl_hours))
    elif a.cmd == "server":
        from .server.app import serve

        serve(Path(a.config), Path(a.pki), Path(a.policy) if a.policy else None, a.host, a.port)
    elif a.cmd == "node":
        from .node.agent import Node, NodeConfig

        Node(NodeConfig(a.server, Path(a.site_dir), Path(a.cert), Path(a.key), Path(a.ca), Path(a.token_file),
                        torch_threads=a.threads, work_dir=Path(a.work_dir) if a.work_dir else None)).run()
    elif a.cmd == "simulate":
        from .simulation import SimConfig, run

        log = run(SimConfig(name=f"cli-{a.algorithm}", algorithm=a.algorithm, rounds=a.rounds, n_sites=a.sites,
                            alpha=a.alpha, noise_multiplier=a.noise_multiplier, compression=a.compression,
                            secagg=a.secagg))
        print(json.dumps({k: v for k, v in log.summary.items() if "label" not in k}, indent=2))
        if a.out:
            log.save(Path(a.out))
    elif a.cmd == "he-demo":
        from .he import benchmark

        print(json.dumps(benchmark(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
