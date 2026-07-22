#!/usr/bin/env bash
# scripts/stress/fault_injection.sh
#
# Fault-Injection / Malformed-Input Handling Test
#
# Verifies that the system handles invalid inputs gracefully: clear error
# messages, no hangs, and appropriate exit codes.
#
# Usage:
#   scripts/stress/fault_injection.sh [--quick] [--out-dir DIR]
#
# Options:
#   --quick     Run abbreviated version (for use within quick_suite.sh)
#   --out-dir   Override output directory
#
# Environment:
#   PYTHONHASHSEED  (default: 0)
#   STRESS_SEED     (default: 1337)
#   STRESS_OUT_DIR  artifact directory
#
# Exit codes:
#   0  all fault-injection cases handled as expected
#   1  unexpected crash or hang

set -euo pipefail

export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export STRESS_SEED="${STRESS_SEED:-1337}"
STRESS_TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"

QUICK_MODE=false
OUT_DIR_OVERRIDE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --quick)   QUICK_MODE=true; shift ;;
        --out-dir) OUT_DIR_OVERRIDE="$2"; shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

if [[ -n "$OUT_DIR_OVERRIDE" ]]; then
    STRESS_OUT_DIR="$OUT_DIR_OVERRIDE"
else
    STRESS_OUT_DIR="${STRESS_OUT_DIR:-artifacts/stress/nightly/${STRESS_TIMESTAMP}}"
fi
export STRESS_OUT_DIR

mkdir -p "${STRESS_OUT_DIR}"
LOGFILE="${STRESS_OUT_DIR}/stderr.log"
exec > >(tee -a "${LOGFILE}") 2>&1

echo "=========================================="
echo "  Fault-Injection Test"
echo "  Timestamp  : ${STRESS_TIMESTAMP}"
echo "  Quick mode : ${QUICK_MODE}"
echo "  Out dir    : ${STRESS_OUT_DIR}"
echo "=========================================="

ERROR_CATALOG="${STRESS_OUT_DIR}/error_catalog.json"

# Helper: run a command with a timeout; record pass/fail
run_case() {
    local name="$1"
    local expected_exit="$2"  # "nonzero" or "any"
    shift 2
    local cmd=("$@")

    echo ""
    echo "--- Case: ${name} ---"
    echo "  CMD: ${cmd[*]}"

    local actual_exit=0
    timeout 60s "${cmd[@]}" 2>&1 || actual_exit=$?

    local status="PASS"
    if [[ "${expected_exit}" == "nonzero" && "${actual_exit}" -eq 0 ]]; then
        status="FAIL (expected non-zero exit, got 0)"
    elif [[ "${actual_exit}" -eq 124 ]]; then
        status="FAIL (timeout)"
    fi

    echo "  Exit: ${actual_exit}  →  ${status}"
}

# ---------------------------------------------------------------------------
# Case 1: Evaluate against a missing tasks file (should fail with clear error)
# ---------------------------------------------------------------------------
run_case "missing_tasks_file" "nonzero" \
    python -m eval_harness evaluate \
        --tasks data/stress/faults/nonexistent_file.jsonl \
        --mock \
        --runs-dir "${STRESS_OUT_DIR}/fault_runs"

# ---------------------------------------------------------------------------
# Case 2: Evaluate against an empty payload file
# Empty file → 0 tasks evaluated (exit 0); verify no crash or hang
# ---------------------------------------------------------------------------
run_case "empty_payload" "any" \
    python -m eval_harness evaluate \
        --tasks data/stress/faults/empty.json \
        --mock \
        --runs-dir "${STRESS_OUT_DIR}/fault_runs"

# ---------------------------------------------------------------------------
# Case 3: Evaluate against malformed JSON
# ---------------------------------------------------------------------------
run_case "malformed_json" "nonzero" \
    python -m eval_harness evaluate \
        --tasks data/stress/faults/malformed_json.json \
        --mock \
        --runs-dir "${STRESS_OUT_DIR}/fault_runs"

# ---------------------------------------------------------------------------
# Case 4: Evaluate against a file with missing required keys
# ---------------------------------------------------------------------------
run_case "missing_keys" "nonzero" \
    python -m eval_harness evaluate \
        --tasks data/stress/faults/missing_keys.json \
        --mock \
        --runs-dir "${STRESS_OUT_DIR}/fault_runs"

# ---------------------------------------------------------------------------
# Case 5 (nightly only): workflow health-check still passes after fault inputs
# ---------------------------------------------------------------------------
if [[ "${QUICK_MODE}" != "true" ]]; then
    run_case "health_check_after_faults" "any" \
        python workflow.py --mode health-check
fi

# ---------------------------------------------------------------------------
# Emit error catalog
# ---------------------------------------------------------------------------
python - <<'PYEOF'
import json, os, sys
from pathlib import Path

out = os.environ.get("STRESS_OUT_DIR", ".")

# Parse results from the log we tee'd above
log = Path(out) / "stderr.log"
text = log.read_text(encoding="utf-8") if log.exists() else ""

catalog = []
for block in text.split("--- Case: "):
    if not block.strip():
        continue
    first_line = block.splitlines()[0].strip().rstrip(" ---")
    exit_line = next((l for l in block.splitlines() if "Exit:" in l), "")
    status = "unknown"
    if "PASS" in exit_line:
        status = "pass"
    elif "FAIL" in exit_line:
        status = "fail"
    catalog.append({"case": first_line, "status": status, "detail": exit_line.strip()})

(Path(out) / "error_catalog.json").write_text(
    json.dumps({"fault_injection": catalog}, indent=2), encoding="utf-8"
)
print(f"error_catalog.json written ({len(catalog)} cases)")
any_fail = any(c["status"] == "fail" for c in catalog)
sys.exit(1 if any_fail else 0)
PYEOF

OVERALL_EXIT=$?

if [[ "${QUICK_MODE}" != "true" ]]; then
    python scripts/stress/collect_artifacts.py \
        --out-dir "${STRESS_OUT_DIR}" \
        --suite nightly \
        --seed "${STRESS_SEED}" \
        --command "scripts/stress/fault_injection.sh" \
        --exit-code "${OVERALL_EXIT}"
fi

if [[ "${OVERALL_EXIT}" -eq 0 ]]; then
    echo ""
    echo "PASS: All fault-injection cases handled correctly"
else
    echo ""
    echo "FAIL: One or more fault-injection cases did not behave as expected"
fi

exit "${OVERALL_EXIT}"
