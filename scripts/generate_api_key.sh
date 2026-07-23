#!/usr/bin/env bash
# scripts/generate_api_key.sh
# Generate a secure API key for the OctoTetrahedral AGI API.
#
# Usage: ./scripts/generate_api_key.sh [--label <label>]
#
# The key is stored in ~/.octotetrahedral/api_keys.json and printed once.

set -euo pipefail

LABEL="${2:-default}"
PYTHON="${PYTHON:-python3}"

echo "╔══════════════════════════════════════════════════╗"
echo "║  OctoTetrahedral AGI — Generate API Key          ║"
echo "╚══════════════════════════════════════════════════╝"
echo

# ── Generate and store the key ────────────────────────────────────────────────
OUTPUT=$("$PYTHON" - <<PYEOF
import sys, os
# Ensure repo root is on path
sys.path.insert(0, os.getcwd())
from auth import generate_api_key, store_api_key
key = generate_api_key()
key_hash = store_api_key(key, label="$LABEL")
print(f"KEY={key}")
print(f"HASH={key_hash[:16]}...")
PYEOF
)

KEY="$(echo "$OUTPUT" | grep '^KEY=' | cut -d= -f2-)"
HASH_PREVIEW="$(echo "$OUTPUT" | grep '^HASH=' | cut -d= -f2-)"

echo "  ✓ API key generated and stored"
echo
echo "┌──────────────────────────────────────────────────┐"
echo "│  Your new API key (save this — shown only once): │"
echo "│                                                   │"
printf "│  %-49s│\n" "$KEY"
echo "│                                                   │"
echo "│  Key hash prefix: $HASH_PREVIEW"
echo "└──────────────────────────────────────────────────┘"
echo
echo "Storage location: ~/.octotetrahedral/api_keys.json"
echo
echo "Usage:"
echo "  curl -H \"Authorization: ******" \\"
echo "       http://localhost:8000/health"
echo
echo "To generate a JWT token from this key:"
echo "  python3 -c \"from auth import create_token; print(create_token('$KEY'))\""
echo
echo "⚠  Keep this key secret — it grants full API access."
