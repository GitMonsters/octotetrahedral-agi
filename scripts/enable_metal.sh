#!/usr/bin/env bash
# scripts/enable_metal.sh
# Enable Apple Silicon Metal GPU acceleration for OctoTetrahedral AGI.
#
# Usage: ./scripts/enable_metal.sh [--restart]
#
# Exit codes:
#   0  Metal is available and enabled (or already configured)
#   1  Not running on Apple Silicon / Metal unavailable

set -euo pipefail

RESTART="${1:-}"

echo "╔══════════════════════════════════════════════════╗"
echo "║  OctoTetrahedral AGI — Enable Metal GPU          ║"
echo "╚══════════════════════════════════════════════════╝"
echo

# ── 1. Detect Apple Silicon ───────────────────────────────────────────────────
echo "▶ Checking hardware…"
ARCH="$(uname -m)"
if [[ "$ARCH" != "arm64" ]]; then
    echo "  ✗ Not running on Apple Silicon (arch: $ARCH)"
    echo "    Metal GPU is only available on macOS with Apple Silicon (M1/M2/M3+)."
    exit 1
fi
echo "  ✓ Apple Silicon detected (arm64)"

# ── 2. Check macOS version ────────────────────────────────────────────────────
echo "▶ Checking macOS version…"
MACOS_VER="$(sw_vers -productVersion 2>/dev/null || echo "unknown")"
echo "  macOS: $MACOS_VER"
MAJOR="$(echo "$MACOS_VER" | cut -d. -f1)"
MINOR="$(echo "$MACOS_VER" | cut -d. -f2)"
if [[ "$MAJOR" -lt 12 ]] || ( [[ "$MAJOR" -eq 12 ]] && [[ "$MINOR" -lt 3 ]] ); then
    echo "  ⚠  macOS 12.3+ required for Metal Performance Shaders."
    echo "     Current version: $MACOS_VER"
fi

# ── 3. Verify PyTorch Metal support ──────────────────────────────────────────
echo "▶ Verifying PyTorch Metal (MPS) support…"
PYTHON="${PYTHON:-python3}"
MPS_CHECK=$("$PYTHON" - <<'PYEOF'
import sys
try:
    import torch
    if torch.backends.mps.is_available():
        print("available")
        sys.exit(0)
    else:
        print("unavailable")
        sys.exit(1)
except Exception as e:
    print(f"error: {e}")
    sys.exit(2)
PYEOF
) || true

if [[ "$MPS_CHECK" == "available" ]]; then
    echo "  ✓ PyTorch Metal (MPS) is available"
elif [[ "$MPS_CHECK" == "unavailable" ]]; then
    echo "  ✗ PyTorch Metal (MPS) is not available on this system."
    echo "    Ensure you are running PyTorch 1.12+ on macOS 12.3+."
    exit 1
else
    echo "  ✗ PyTorch check failed: $MPS_CHECK"
    exit 1
fi

# ── 4. Install / verify metal-optimized dependencies ─────────────────────────
echo "▶ Verifying Python environment…"
if "$PYTHON" -c "import torch, fastapi, uvicorn" 2>/dev/null; then
    echo "  ✓ Core dependencies present"
else
    echo "  ⚠  Some dependencies missing — installing now…"
    "$PYTHON" -m pip install --quiet torch fastapi uvicorn psutil
    echo "  ✓ Dependencies installed"
fi

# ── 5. Set environment variable and optionally restart ───────────────────────
echo "▶ Setting Metal device preference…"
export OCTO_DEVICE=mps
echo "  ✓ OCTO_DEVICE=mps"
echo
echo "  Add this to your shell profile to make it permanent:"
echo "  export OCTO_DEVICE=mps"
echo

if [[ "$RESTART" == "--restart" ]]; then
    echo "▶ Restarting API service…"
    # Restart via LaunchAgent if configured
    PLIST="$HOME/Library/LaunchAgents/com.octotetrahedral.plist"
    if [[ -f "$PLIST" ]]; then
        launchctl unload "$PLIST" 2>/dev/null || true
        launchctl load "$PLIST"
        echo "  ✓ LaunchAgent restarted with Metal enabled"
    else
        echo "  ℹ  No LaunchAgent found — start the API manually:"
        echo "  OCTO_DEVICE=mps python3 -m uvicorn api:app --host 0.0.0.0 --port 8000"
    fi
fi

echo
echo "✅ Metal GPU acceleration enabled!"
echo "   Start the API: OCTO_DEVICE=mps python3 -m uvicorn api:app --host 0.0.0.0 --port 8000"
