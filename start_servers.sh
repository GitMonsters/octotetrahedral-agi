#!/usr/bin/env bash
# Start OctoTetrahedral braided inference stack
# Usage: ./start_servers.sh [--workers N] [--port-py P] [--port-rs R]
set -e

WORKERS=4
PORT_PY=8765
PORT_RS=8766

while [[ $# -gt 0 ]]; do
  case $1 in
    --workers) WORKERS=$2; shift 2;;
    --port-py) PORT_PY=$2; shift 2;;
    --port-rs) PORT_RS=$2; shift 2;;
    *) echo "Unknown: $1"; exit 1;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Kill existing instances by PID (one at a time — macOS kill requires numeric PID)
while IFS= read -r pid; do
  [ -n "$pid" ] && kill "$pid" 2>/dev/null || true
done < <(ps aux | grep -E "octo_limb_server|octo-parallel" | grep -v grep | awk '{print $2}')
sleep 1

# Python limb server (multi-worker; </dev/null required so forked workers don't
# inherit a closed stdin and crash with "Bad file descriptor")
echo "Starting Python limb server (${WORKERS} workers) on port ${PORT_PY}..."
source venv/bin/activate 2>/dev/null || true
nohup python3 octo_limb_server.py \
  --host 127.0.0.1 --port "$PORT_PY" --workers "$WORKERS" \
  </dev/null >/tmp/octo_python.log 2>&1 &
PY_PID=$!

for i in $(seq 1 20); do
  sleep 1
  if curl -sf "http://127.0.0.1:${PORT_PY}/healthz" >/dev/null 2>&1; then
    echo "  Python ready (PID=$PY_PID)"
    break
  fi
done

# Rust orchestrator
echo "Starting Rust orchestrator on port ${PORT_RS}..."
nohup octo-parallel-rs/target/debug/octo-parallel \
  --python-url "http://127.0.0.1:${PORT_PY}" --port "$PORT_RS" \
  </dev/null >/tmp/octo_rust.log 2>&1 &
RS_PID=$!
sleep 3

if curl -sf "http://127.0.0.1:${PORT_RS}/healthz" >/dev/null 2>&1; then
  echo "  Rust ready (PID=$RS_PID)"
fi

echo ""
echo "Stack running:"
echo "  Python: http://127.0.0.1:${PORT_PY}  (logs: /tmp/octo_python.log)"
echo "  Rust:   http://127.0.0.1:${PORT_RS}  (logs: /tmp/octo_rust.log)"
echo ""
echo "Quick test:"
echo "  curl -s -X POST http://127.0.0.1:${PORT_RS}/infer \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"input_ids\":[42,17,93,7]}' | python3 -m json.tool"
