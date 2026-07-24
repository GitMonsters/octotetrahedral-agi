#!/usr/bin/env bash
# Enable Apple Silicon Metal (MPS) GPU acceleration for OctoTetrahedral AGI.
# Detects Apple Silicon, installs Metal-optimised PyTorch, restarts the
# LaunchAgent service, and verifies Metal support is active.
#
# Usage:  ./scripts/enable_metal.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
PLIST="$HOME/Library/LaunchAgents/com.octotetrahedral.plist"

# ---------------------------------------------------------------------------
# 1. Detect Apple Silicon
# ---------------------------------------------------------------------------
echo "🍎 Detecting Apple Silicon ..."
ARCH="$(uname -m)"
if [[ "$ARCH" != "arm64" ]]; then
    echo "❌ This script requires Apple Silicon (arm64). Detected: $ARCH"
    exit 1
fi
echo "✅ Apple Silicon detected ($ARCH)"

# ---------------------------------------------------------------------------
# 2. Locate Python / venv
# ---------------------------------------------------------------------------
if [[ -f "$REPO_ROOT/venv/bin/python" ]]; then
    PYTHON="$REPO_ROOT/venv/bin/python"
    PIP="$REPO_ROOT/venv/bin/pip"
elif command -v python3 &>/dev/null; then
    PYTHON="python3"
    PIP="pip3"
else
    PYTHON="python"
    PIP="pip"
fi

# ---------------------------------------------------------------------------
# 3. Install Metal-optimised PyTorch (nightly MPS build if needed)
# ---------------------------------------------------------------------------
echo ""
echo "📦 Installing/upgrading PyTorch with Metal (MPS) support ..."
"$PIP" install --upgrade torch torchvision torchaudio 2>&1 | tail -5

# ---------------------------------------------------------------------------
# 4. Verify MPS support
# ---------------------------------------------------------------------------
echo ""
echo "🔍 Verifying Metal support ..."
MPS_OK=$("$PYTHON" -c "
import torch, sys
if torch.backends.mps.is_available():
    t = torch.tensor([1.0]).to('mps')
    print('ok')
else:
    print('unavailable')
")

if [[ "$MPS_OK" != "ok" ]]; then
    echo "⚠️  MPS backend is not available on this system."
    echo "   PyTorch is installed but Metal GPU will not be used."
else
    echo "✅ Metal MPS backend is available!"
fi

# ---------------------------------------------------------------------------
# 5. Restart LaunchAgent service (if loaded)
# ---------------------------------------------------------------------------
echo ""
if [[ -f "$PLIST" ]]; then
    echo "🔄 Restarting OctoTetrahedral service ..."
    launchctl unload "$PLIST" 2>/dev/null || true
    sleep 1
    launchctl load "$PLIST"
    sleep 2
    echo "✅ Service restarted"

    echo ""
    echo "🏥 Verifying health ..."
    curl -sf http://localhost:8000/health | "$PYTHON" -m json.tool || echo "(service may still be starting)"
else
    echo "ℹ️  LaunchAgent plist not found at $PLIST"
    echo "   Start the service manually:  cd $REPO_ROOT && uvicorn api:app --host 0.0.0.0 --port 8000"
fi

echo ""
echo "🚀 Metal GPU enablement complete!"
if [[ "$MPS_OK" == "ok" ]]; then
    echo "   Expected inference speedup: 5-10x vs CPU"
fi
