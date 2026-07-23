#!/usr/bin/env bash
# scripts/setup_monitoring.sh
# Install monitoring dependencies and configure the monitoring service.
#
# Usage: ./scripts/setup_monitoring.sh [--background]

set -euo pipefail

BACKGROUND="${1:-}"
PYTHON="${PYTHON:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DASHBOARD_PORT="${DASHBOARD_PORT:-8001}"

echo "╔══════════════════════════════════════════════════╗"
echo "║  OctoTetrahedral AGI — Setup Monitoring          ║"
echo "╚══════════════════════════════════════════════════╝"
echo

# ── Install dependencies ──────────────────────────────────────────────────────
echo "▶ Installing monitoring dependencies…"
"$PYTHON" -m pip install --quiet psutil 2>/dev/null && echo "  ✓ psutil installed" \
    || echo "  ⚠  psutil not installed (memory metrics will be skipped)"
echo

# ── Verify monitoring module ──────────────────────────────────────────────────
echo "▶ Verifying monitoring package…"
cd "$REPO_ROOT"
"$PYTHON" -c "
import sys
sys.path.insert(0, '.')
from monitoring import InferenceMonitor
print('  ✓ monitoring package OK')
"

# ── Verify API endpoints exist ────────────────────────────────────────────────
echo "▶ Verifying api.py endpoints…"
"$PYTHON" -c "
import sys
with open('$REPO_ROOT/api.py') as f:
    src = f.read()
for ep in ['/metrics', '/performance', '/stats']:
    if ep in src:
        print(f'  ✓ {ep} endpoint present')
    else:
        print(f'  ✗ {ep} endpoint missing')
        sys.exit(1)
"

# ── Configure logging directory ───────────────────────────────────────────────
echo "▶ Setting up log directory…"
LOG_DIR="$HOME/Library/Logs/octotetrahedral"
if ! mkdir -p "$LOG_DIR" 2>/dev/null; then
    LOG_DIR="/tmp/octotetrahedral-logs"
    mkdir -p "$LOG_DIR"
fi
echo "  ✓ Log directory: $LOG_DIR"

# ── Optionally start monitoring in background ─────────────────────────────────
if [[ "$BACKGROUND" == "--background" ]]; then
    echo "▶ Starting monitoring in background…"
    DASHBOARD_HTML="$REPO_ROOT/monitoring_dashboard.html"
    if [[ -f "$DASHBOARD_HTML" ]]; then
        nohup "$PYTHON" -m http.server "$DASHBOARD_PORT" --directory "$REPO_ROOT" \
            > "$LOG_DIR/monitoring.log" 2>&1 &
        echo "  ✓ Dashboard started (PID $!, port $DASHBOARD_PORT)"
        echo "  ✓ Logs: $LOG_DIR/monitoring.log"
    else
        echo "  ✗ monitoring_dashboard.html not found — skipping background start"
    fi
fi

echo
echo "✅ Monitoring setup complete!"
echo
echo "   To start monitoring:  ./scripts/start_monitoring.sh"
echo "   Dashboard:            http://localhost:$DASHBOARD_PORT/monitoring_dashboard.html"
echo "   Metrics:              http://localhost:8000/metrics"
echo "   Performance:          http://localhost:8000/performance"
echo "   Stats:                http://localhost:8000/stats"
