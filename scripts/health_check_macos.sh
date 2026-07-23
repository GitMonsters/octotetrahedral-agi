#!/usr/bin/env bash
# =============================================================================
# OctoTetrahedral AGI – macOS Health Check
# =============================================================================
# Performs a series of checks against a running production instance and
# reports the overall health in a structured format.
#
# Usage: bash scripts/health_check_macos.sh [--port PORT] [--host HOST]
# Exit:  0 = healthy, 1 = degraded / unhealthy
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PYTHON="${REPO_ROOT}/.venv/bin/python"

# Fall back to system python if venv not yet created
PYTHON="$(command -v python3 2>/dev/null || true)"
if [[ -x "${VENV_PYTHON}" ]]; then
  PYTHON="${VENV_PYTHON}"
fi

# Colour helpers
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
pass()  { echo -e "  ${GREEN}✓${NC}  $*"; }
fail()  { echo -e "  ${RED}✗${NC}  $*"; FAILURES=$(( FAILURES + 1 )); }
warn()  { echo -e "  ${YELLOW}!${NC}  $*"; }
FAILURES=0

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
for arg in "$@"; do
  case "$arg" in
    --port=*) PORT="${arg#*=}" ;;
    --port)   shift; PORT="$1" ;;
    --host=*) HOST="${arg#*=}" ;;
    --host)   shift; HOST="$1" ;;
  esac
done

BASE_URL="http://${HOST}:${PORT}"

echo "============================================================"
echo " OctoTetrahedral AGI – Health Check"
echo " Target: ${BASE_URL}"
echo "============================================================"
echo ""

# ---------------------------------------------------------------------------
# 1. Endpoint availability
# ---------------------------------------------------------------------------
echo "1. Endpoint availability"
if curl -sf --max-time 5 "${BASE_URL}/health" -o /dev/null; then
  pass "/health endpoint reachable"
else
  fail "/health endpoint unreachable at ${BASE_URL}"
fi

# ---------------------------------------------------------------------------
# 2. Health endpoint response
# ---------------------------------------------------------------------------
echo ""
echo "2. Health endpoint response"
HEALTH_JSON="$(curl -sf --max-time 5 "${BASE_URL}/health" 2>/dev/null || true)"
if [[ -n "${HEALTH_JSON}" ]]; then
  STATUS="$(echo "${HEALTH_JSON}" | "${PYTHON}" -c "import sys,json; d=json.load(sys.stdin); print(d.get('status','unknown'))" 2>/dev/null || echo 'unknown')"
  DEVICE="$(echo "${HEALTH_JSON}" | "${PYTHON}" -c "import sys,json; d=json.load(sys.stdin); print(d.get('device','unknown'))" 2>/dev/null || echo 'unknown')"
  if [[ "${STATUS}" == "healthy" ]]; then
    pass "Service status: ${STATUS}"
  else
    fail "Service status: ${STATUS}"
  fi
  pass "Compute device: ${DEVICE}"
else
  fail "Empty or invalid response from /health"
fi

# ---------------------------------------------------------------------------
# 3. Metal / MPS device detection
# ---------------------------------------------------------------------------
echo ""
echo "3. Metal (MPS) device detection"
if [[ -x "${VENV_PYTHON}" ]]; then
  MPS_RESULT="$("${PYTHON}" -c "
import torch, sys
if torch.backends.mps.is_available():
    try:
        t = torch.ones(4, device='mps')
        assert t.sum().item() == 4.0
        print('available')
    except Exception as e:
        print(f'error:{e}')
else:
    print('unavailable')
" 2>/dev/null || echo 'check_failed')"

  case "${MPS_RESULT}" in
    available)    pass "Metal MPS backend available and functional" ;;
    unavailable)  warn "Metal MPS not available – service running on CPU fallback" ;;
    check_failed) warn "MPS check could not complete – venv may be missing torch" ;;
    error:*)      warn "MPS functional test failed: ${MPS_RESULT#error:}" ;;
  esac
else
  warn "Virtual environment not found at ${REPO_ROOT}/.venv – skipping MPS check"
fi

# ---------------------------------------------------------------------------
# 4. Model loading verification (via /health)
# ---------------------------------------------------------------------------
echo ""
echo "4. Model loading verification"
if [[ -n "${HEALTH_JSON}" ]]; then
  MODEL="$(echo "${HEALTH_JSON}" | "${PYTHON}" -c "import sys,json; d=json.load(sys.stdin); print(d.get('model','unknown'))" 2>/dev/null || echo 'unknown')"
  if [[ "${MODEL}" != "unknown" && -n "${MODEL}" ]]; then
    pass "Model reported: ${MODEL}"
  else
    warn "Model info not available in /health response"
  fi
fi

# ---------------------------------------------------------------------------
# 5. Inference accuracy test
# ---------------------------------------------------------------------------
echo ""
echo "5. Inference accuracy test"
PREDICT_RESULT="$(curl -sf --max-time 10 \
  -X POST "${BASE_URL}/predict" \
  -H "Content-Type: application/json" \
  -d '{"input_ids": [1, 2, 3, 4, 5, 6, 7, 8]}' 2>/dev/null || true)"

if [[ -n "${PREDICT_RESULT}" ]]; then
  SUCCESS="$(echo "${PREDICT_RESULT}" | "${PYTHON}" -c "import sys,json; d=json.load(sys.stdin); print(d.get('success', False))" 2>/dev/null || echo 'false')"
  if [[ "${SUCCESS}" == "True" ]]; then
    pass "Inference returned successfully"
  else
    DETAIL="$(echo "${PREDICT_RESULT}" | "${PYTHON}" -c "import sys,json; d=json.load(sys.stdin); print(d.get('detail','no detail'))" 2>/dev/null || echo 'parse error')"
    fail "Inference returned failure: ${DETAIL}"
  fi
else
  warn "Could not reach /predict endpoint (service may be loading)"
fi

# ---------------------------------------------------------------------------
# 6. Performance baseline (latency)
# ---------------------------------------------------------------------------
echo ""
echo "6. Performance baseline (latency)"
START_NS="$(date +%s%N 2>/dev/null || echo 0)"
curl -sf --max-time 10 \
  -X POST "${BASE_URL}/predict" \
  -H "Content-Type: application/json" \
  -d '{"input_ids": [1, 2, 3, 4, 5, 6, 7, 8]}' \
  -o /dev/null 2>/dev/null || true
END_NS="$(date +%s%N 2>/dev/null || echo 0)"

if [[ "${START_NS}" != "0" && "${END_NS}" != "0" ]]; then
  ELAPSED_MS=$(( (END_NS - START_NS) / 1000000 ))
  if (( ELAPSED_MS < 2000 )); then
    pass "Latency: ${ELAPSED_MS}ms (< 2000ms threshold)"
  else
    warn "Latency: ${ELAPSED_MS}ms (above 2000ms threshold)"
  fi
else
  warn "Could not measure latency (date +%s%N not supported)"
fi

# ---------------------------------------------------------------------------
# 7. Memory usage check
# ---------------------------------------------------------------------------
echo ""
echo "7. Memory usage"
if command -v vm_stat &>/dev/null; then
  FREE_PAGES="$(vm_stat | awk '/^Pages free:/ { gsub(/\./,"",$3); print $3 }')"
  if [[ -n "${FREE_PAGES}" ]]; then
    FREE_MB=$(( FREE_PAGES * 4096 / 1024 / 1024 ))
    if (( FREE_MB > 256 )); then
      pass "Free memory: ~${FREE_MB} MB"
    else
      warn "Low free memory: ~${FREE_MB} MB"
    fi
  fi
else
  warn "vm_stat not found; skipping memory check"
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
if (( FAILURES == 0 )); then
  echo -e " ${GREEN}✅  All checks passed${NC}"
  exit 0
else
  echo -e " ${RED}❌  ${FAILURES} check(s) failed${NC}"
  exit 1
fi
