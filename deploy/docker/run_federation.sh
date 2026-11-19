#!/usr/bin/env bash
# Same topology as docker-compose.yml using plain `docker` (no compose plugin needed).
# One coordinator, N hospital containers, each hospital on its own network shared only
# with the coordinator. Usage: deploy/docker/run_federation.sh [config.yaml]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CONFIG="${1:-$ROOT/deploy/config/federation.docker.yaml}"
IMG="${FEDMED_IMAGE:-fedmed:latest}"
SITES=(hospital-a hospital-b hospital-c hospital-d hospital-e)
U="$(id -u):$(id -g)"
HARDEN=(--user "$U" --read-only --tmpfs /work:mode=1777 --tmpfs /tmp:mode=1777
        --security-opt no-new-privileges:true --cap-drop ALL)

cleanup() {
  docker rm -f coordinator "${SITES[@]}" >/dev/null 2>&1 || true
  for s in "${SITES[@]}"; do docker network rm "net-$s" >/dev/null 2>&1 || true; done
}
trap cleanup EXIT
cleanup
mkdir -p "$ROOT/runs/coordinator"

for s in "${SITES[@]}"; do docker network create "net-$s" >/dev/null; done

docker run -d --name coordinator --network net-hospital-a -p 8443:8443 "${HARDEN[@]}" \
  -v "$CONFIG:/config/federation.yaml:ro" \
  -v "$ROOT/deploy/config/policy.yaml:/config/policy.yaml:ro" \
  -v "$ROOT/certs/ca.crt:/pki/ca.crt:ro" -v "$ROOT/certs/crl.pem:/pki/crl.pem:ro" \
  -v "$ROOT/certs/server.crt:/pki/server.crt:ro" -v "$ROOT/certs/server.key:/pki/server.key:ro" \
  -v "$ROOT/certs/token_signing.pub:/pki/token_signing.pub:ro" \
  -v "$ROOT/data/sites/_benchmark:/benchmark:ro" \
  -v "$ROOT/runs/coordinator:/work/runs/coordinator" \
  "$IMG" server --config /config/federation.yaml --policy /config/policy.yaml --pki /pki --port 8443 >/dev/null
for s in "${SITES[@]:1}"; do docker network connect "net-$s" coordinator; done

for s in "${SITES[@]}"; do
  limits=(); threads=2
  if [[ "$s" == "hospital-e" ]]; then limits=(--cpus 0.5 --memory 768m); threads=1; fi   # edge node
  docker run -d --name "$s" --network "net-$s" "${HARDEN[@]}" "${limits[@]}" \
    -v "$ROOT/data/sites/$s:/data:ro" -v "$ROOT/certs/ca.crt:/pki/ca.crt:ro" \
    -v "$ROOT/certs/$s.crt:/pki/client.crt:ro" -v "$ROOT/certs/$s.key:/pki/client.key:ro" \
    -v "$ROOT/certs/tokens/$s.jwt:/pki/token.jwt:ro" \
    "$IMG" node --server https://coordinator:8443 --site-dir /data --cert /pki/client.crt \
      --key /pki/client.key --ca /pki/ca.crt --token-file /pki/token.jwt --work-dir /work --threads "$threads" >/dev/null
done

echo "federation running; following coordinator log (Ctrl-C stops everything)"
docker logs -f coordinator &
for s in "${SITES[@]}"; do docker wait "$s" >/dev/null; done
sleep 1
echo "--- isolation check: can hospital-a reach hospital-b? (should fail)"
docker run --rm -i --network net-hospital-a --entrypoint python "$IMG" - <<'PY' 2>&1 | tail -1 || true
import socket
try:
    socket.create_connection(("hospital-b", 8443), timeout=3); print("REACHABLE (bad)")
except OSError as e:
    print("unreachable:", type(e).__name__)
PY
for s in "${SITES[@]}"; do echo "--- $s"; docker logs "$s" 2>&1 | grep -v httpx | tail -2; done
