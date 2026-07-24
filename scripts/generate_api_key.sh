#!/usr/bin/env bash
# Generate a new OctoTetrahedral AGI API key and save it to
# ~/.octotetrahedral/api_keys.json
#
# Usage:  ./scripts/generate_api_key.sh [label]

set -euo pipefail

LABEL="${1:-default}"
STORE_DIR="$HOME/.octotetrahedral"
STORE_FILE="$STORE_DIR/api_keys.json"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Locate python in the repo venv if present, otherwise fall back to PATH
if [[ -f "$REPO_ROOT/venv/bin/python" ]]; then
    PYTHON="$REPO_ROOT/venv/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON="python3"
else
    PYTHON="python"
fi

echo "🔑 Generating API key (label: $LABEL) ..."

KEY=$("$PYTHON" - "$LABEL" <<'EOF'
import sys
sys.path.insert(0, '.')
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/..')
from auth import generate_api_key
label = sys.argv[1] if len(sys.argv) > 1 else "default"
print(generate_api_key(label))
EOF
)

mkdir -p "$STORE_DIR"

echo ""
echo "✅ API key generated successfully!"
echo ""
echo "  Key:   $KEY"
echo "  Store: $STORE_FILE"
echo ""
echo "Use it in requests:"
echo "  curl -H 'Authorization: ******' http://localhost:8000/predict \\"
echo "       -H 'Content-Type: application/json' \\"
echo "       -d '{\"input_ids\": [1, 2, 3]}'"
