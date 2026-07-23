#!/usr/bin/env bash
# scripts/start_monitoring.sh
# Launch the OctoTetrahedral AGI monitoring dashboard and metrics collection.
#
# Usage: ./scripts/start_monitoring.sh [--port <port>] [--api-url <url>]
#
# The dashboard is served at http://localhost:<port> (default 8001).

set -euo pipefail

DASHBOARD_PORT="${DASHBOARD_PORT:-8001}"
API_URL="${API_URL:-http://localhost:8000}"
PYTHON="${PYTHON:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Parse optional args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --port) DASHBOARD_PORT="$2"; shift 2 ;;
        --api-url) API_URL="$2"; shift 2 ;;
        *) shift ;;
    esac
done

echo "╔══════════════════════════════════════════════════╗"
echo "║  OctoTetrahedral AGI — Start Monitoring          ║"
echo "╚══════════════════════════════════════════════════╝"
echo

# ── Check API is reachable ────────────────────────────────────────────────────
echo "▶ Checking API at $API_URL/health…"
if curl --silent --max-time 3 "$API_URL/health" > /dev/null 2>&1; then
    echo "  ✓ API is reachable"
else
    echo "  ⚠  API not reachable at $API_URL — dashboard will poll until available"
fi

# ── Start the dashboard HTTP server ──────────────────────────────────────────
echo "▶ Starting monitoring dashboard on port $DASHBOARD_PORT…"
DASHBOARD_HTML="$REPO_ROOT/monitoring_dashboard.html"

if [[ ! -f "$DASHBOARD_HTML" ]]; then
    echo "  ✗ monitoring_dashboard.html not found at $DASHBOARD_HTML"
    exit 1
fi

# Serve the dashboard in the background
cd "$REPO_ROOT"
"$PYTHON" -m http.server "$DASHBOARD_PORT" --directory "$REPO_ROOT" &
DASHBOARD_PID=$!
echo "  ✓ Dashboard PID: $DASHBOARD_PID"

# ── Start metrics collection loop ─────────────────────────────────────────────
echo "▶ Starting metrics collection (logs to stdout)…"
echo
echo "┌──────────────────────────────────────────────────┐"
echo "│  Monitoring dashboard: http://localhost:$DASHBOARD_PORT/monitoring_dashboard.html"
echo "│  Metrics endpoint:     $API_URL/metrics"
echo "│  Performance summary:  $API_URL/performance"
echo "│  Real-time stats:      $API_URL/stats"
echo "└──────────────────────────────────────────────────┘"
echo
echo "Press Ctrl+C to stop."
echo

# Poll /stats every 5 seconds as lightweight collection
cleanup() {
    echo
    echo "Stopping monitoring (PID $DASHBOARD_PID)…"
    kill "$DASHBOARD_PID" 2>/dev/null || true
    exit 0
}
trap cleanup INT TERM

while true; do
    TS="$(date '+%H:%M:%S')"
    STATS=$(curl --silent --max-time 2 "$API_URL/stats" 2>/dev/null || echo '{}')
    echo "[$TS] $STATS"
    sleep 5
done
