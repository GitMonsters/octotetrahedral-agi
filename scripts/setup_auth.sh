#!/usr/bin/env bash
# Set up API key authentication for OctoTetrahedral AGI.
# Installs required Python dependencies, creates the key storage directory,
# generates an initial API key, and verifies the setup.
#
# Usage:  ./scripts/setup_auth.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
STORE_DIR="$HOME/.octotetrahedral"

# ---------------------------------------------------------------------------
# 1. Locate Python / pip
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
# 2. Install dependencies
# ---------------------------------------------------------------------------
echo "📦 Installing authentication dependencies ..."
"$PIP" install --upgrade fastapi uvicorn 2>&1 | tail -5
echo "✅ Dependencies installed"

# ---------------------------------------------------------------------------
# 3. Create key storage directory
# ---------------------------------------------------------------------------
echo ""
echo "📁 Creating key storage directory: $STORE_DIR"
mkdir -p "$STORE_DIR"
chmod 700 "$STORE_DIR"
echo "✅ Directory created (mode 700)"

# ---------------------------------------------------------------------------
# 4. Generate initial API key
# ---------------------------------------------------------------------------
echo ""
echo "🔑 Generating initial API key ..."
cd "$REPO_ROOT"
KEY=$("$PYTHON" -c "
import sys
sys.path.insert(0, '.')
from auth import generate_api_key
print(generate_api_key('initial'))
")

echo ""
echo "✅ Initial API key generated!"
echo ""
echo "  Key:   $KEY"
echo "  Store: $STORE_DIR/api_keys.json"
echo ""

# ---------------------------------------------------------------------------
# 5. Verify setup
# ---------------------------------------------------------------------------
echo "🔍 Verifying setup ..."
"$PYTHON" -c "
import sys
sys.path.insert(0, '.')
from auth import validate_key, get_key_stats
stats = get_key_stats()
assert stats['total_keys'] >= 1, 'No keys found after generation'
print(f'✅ Verification passed: {stats[\"total_keys\"]} key(s) registered')
"

echo ""
echo "🚀 Authentication setup complete!"
echo ""
echo "Use your API key in requests:"
echo "  curl -H 'Authorization: ******' http://localhost:8000/predict \\"
echo "       -H 'Content-Type: application/json' \\"
echo "       -d '{\"input_ids\": [1, 2, 3]}'"
echo ""
echo "View stats:"
echo "  curl http://localhost:8000/stats"
echo "  curl http://localhost:8000/metrics"
