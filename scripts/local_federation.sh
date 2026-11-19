#!/usr/bin/env bash
# Coordinator + five hospital nodes as separate local processes, talking HTTPS/mTLS.
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG="${1:-deploy/config/federation.yaml}"
mkdir -p runs/logs
fedmed server --config "$CONFIG" --pki certs --policy deploy/config/policy.yaml --port 8443 \
  > runs/logs/coordinator.log 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
sleep 3
PIDS=()
for s in hospital-a hospital-b hospital-c hospital-d hospital-e; do
  fedmed node --server https://localhost:8443 --site-dir "data/sites/$s" --cert "certs/$s.crt" \
    --key "certs/$s.key" --ca certs/ca.crt --token-file "certs/tokens/$s.jwt" --threads 2 \
    --work-dir "runs/$s" > "runs/logs/$s.log" 2>&1 &
  PIDS+=($!)
done
tail -f runs/logs/coordinator.log &
TAIL=$!
for p in "${PIDS[@]}"; do wait "$p"; done
kill $TAIL 2>/dev/null || true
echo "done. history: runs/coordinator/history.json, audit log: runs/coordinator/audit.jsonl"
