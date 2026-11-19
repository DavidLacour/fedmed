# Deployment guide

Three ways to run the same code, from least to most realistic.

| Mode | Command | Status in this repo |
|---|---|---|
| Local processes over mTLS | `make demo` | Run and verified |
| Hardened containers, isolated networks, edge-limited node | `make docker-federation` (plain `docker`) or `docker compose -f deploy/docker/docker-compose.yml up` | `run_federation.sh` run and verified. The compose file mirrors it but has not been run here. |
| Kubernetes (Helm) | `helm install fl deploy/helm/fedmed` | `helm lint` and `helm template` pass. Not yet installed on a cluster. |
| **Separate machines** | see below | **To do: the next milestone** |

> Honesty note: everything above runs on one physical host. Separate containers and
> networks emulate separate institutions, but this is not a real multi-site
> deployment. The steps below describe how to run one, and the README should only
> claim it once it has actually been done.

## Multi-machine run (cloud VMs and/or Raspberry Pi)

Topology: 1 coordinator VM with a public IP or DNS name, and N hospital machines that only
need **outbound** HTTPS to the coordinator. Hospitals never accept inbound connections,
which is usually what a hospital firewall team wants to hear.

### 1. Coordinator (e.g. a small cloud VM, 2 vCPU / 4 GB)
```bash
git clone <repo> && cd fedmed && make install
fedmed prepare-data --out data/sites --sites 5 --alpha 0.5        # or copy only _benchmark/ here
fedmed pki init --out certs --sites hospital-a hospital-b hospital-c \
                --server-san fl.example.org 203.0.113.10
make tokens
fedmed server --config deploy/config/federation.yaml --policy deploy/config/policy.yaml --pki certs
```
Open TCP 8443 in the security group **only from the hospitals' egress IPs**.

### 2. Distribute credentials (out of band, e.g. `scp`)
Each hospital receives exactly four files: `ca.crt`, `<site>.crt`, `<site>.key` and
`tokens/<site>.jwt`. It also receives *its own* data folder: in a real federation the
data would already be there, and `prepare-data` exists only to simulate it.

### 3. Each hospital
```bash
make install          # or: docker pull <registry>/fedmed:0.1.0
fedmed node --server https://fl.example.org:8443 --site-dir data/sites/hospital-a \
  --cert hospital-a.crt --key hospital-a.key --ca ca.crt --token-file hospital-a.jwt --work-dir runs/
```

### 4. Raspberry Pi as an edge hospital (optional)
* Pi 4/5 with 64-bit Raspberry Pi OS: `pip install torch` provides aarch64 CPU wheels;
  then `pip install -e .`.
* Or build a multi-arch image: `docker buildx build --platform linux/amd64,linux/arm64 -f deploy/docker/Dockerfile -t <registry>/fedmed:0.1.0 --push .`
* Use `--threads 1..4` and expect roughly 5–10× slower rounds. The coordinator waits for the slowest
  site, so record per-site round time in the write-up (it is printed in every node log line).

### What to measure and report from the real run
* wall time per round per site (node logs), bytes in and out (coordinator `/status`);
* the round-trip latency between sites (`ping`, `curl -w '%{time_connect}'`);
* that the accuracy curve matches the simulation with the same seed and config. The
  simulation and the deployment run the same `local_train` and `strategies` code;
* screenshots of `audit.jsonl` showing a denied request (for example, the auditor trying `/model`).

## Kubernetes notes
* `make k8s-secrets` creates `fedmed-coordinator-pki` and `<site>-pki` Secrets from `certs/`.
* Each hospital needs a PVC named in `values.yaml` holding `images/` and `labels.csv`.
* For a cross-cluster setup, install with `coordinator.enabled=false`, a single
  `hospitals:` entry and `coordinator.externalUrl=https://fl.example.org:8443` in each
  hospital's cluster.
* NetworkPolicies need a CNI that enforces them (Calico or Cilium; kind's default kindnet does not).
