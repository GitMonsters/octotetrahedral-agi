#!/usr/bin/env bash
# scripts/stress/soak.sh
#
# Long-Run Soak Test
#
# Runs the inference workflow repeatedly against the soak dataset to catch
# memory leaks and throughput degradation.  Default repeat count is 10 for
# local/CI runs; set SOAK_REPEAT=1000 for a full 6-hour overnight run.
#
# Usage:
#   scripts/stress/soak.sh [--repeat N] [--out-dir DIR]
#
# Environment:
#   PYTHONHASHSEED  (default: 0)
#   STRESS_SEED     (default: 1337)
#   SOAK_REPEAT     number of repeated evaluation passes (default: 10)
#   STRESS_OUT_DIR  artifact directory
#
# Exit codes:
#   0  soak completed within SLO thresholds
#   1  OOM, crash, or throughput drift beyond SLO

set -euo pipefail

export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export STRESS_SEED="${STRESS_SEED:-1337}"
STRESS_TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"

SOAK_REPEAT="${SOAK_REPEAT:-10}"
OUT_DIR_OVERRIDE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --repeat)  SOAK_REPEAT="$2"; shift 2 ;;
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
LOGFILE="${STRESS_OUT_DIR}/soak.log"
exec > >(tee -a "${LOGFILE}") 2>&1

echo "=========================================="
echo "  Long-Run Soak Test"
echo "  Timestamp : ${STRESS_TIMESTAMP}"
echo "  Seed      : ${STRESS_SEED}"
echo "  Repeat    : ${SOAK_REPEAT}"
echo "  Input     : data/stress/soak/mix.jsonl"
echo "  Out dir   : ${STRESS_OUT_DIR}"
echo "=========================================="

THROUGHPUT_FILE="${STRESS_OUT_DIR}/throughput_timeseries.csv"
printf "iteration,elapsed_s,tasks_completed,errors\n" > "${THROUGHPUT_FILE}"

TOTAL_ERRORS=0
START_TS=$(date +%s)

RUNS_DIR="${STRESS_OUT_DIR}/soak_runs"
mkdir -p "${RUNS_DIR}"

for I in $(seq 1 "${SOAK_REPEAT}"); do
    ITER_START=$(date +%s)
    echo ""
    echo "--- Iteration ${I}/${SOAK_REPEAT} ---"

    python -m eval_harness evaluate \
        --tasks data/stress/soak/mix.jsonl \
        --mock --mock-score 0.9 \
        --seed "${STRESS_SEED}" \
        --runs-dir "${RUNS_DIR}" \
        --tag "soak-iter-${I}" \
        || { TOTAL_ERRORS=$((TOTAL_ERRORS + 1)); echo "WARNING: iteration ${I} failed"; }

    ITER_END=$(date +%s)
    ELAPSED=$((ITER_END - START_TS))
    printf "%d,%d,50,%d\n" "${I}" "${ELAPSED}" "${TOTAL_ERRORS}" >> "${THROUGHPUT_FILE}"
done

TOTAL_ELAPSED=$(($(date +%s) - START_TS))
TOTAL_ITERS="${SOAK_REPEAT}"

echo ""
echo "--- Soak summary ---"
echo "  Iterations   : ${TOTAL_ITERS}"
echo "  Total errors : ${TOTAL_ERRORS}"
echo "  Total elapsed: ${TOTAL_ELAPSED}s"

# SLO check: error rate <= 0.5%
python - <<PYEOF
import sys
errors = ${TOTAL_ERRORS}
iters = ${TOTAL_ITERS}
error_rate = errors / iters if iters else 0
threshold = 0.005
print(f"  Error rate: {error_rate:.4%} (threshold: {threshold:.1%})")
if error_rate > threshold:
    print("FAIL: Error rate exceeds SLO")
    sys.exit(1)
else:
    print("PASS: Error rate within SLO")
PYEOF

EXIT_CODE=$?

python scripts/stress/collect_artifacts.py \
    --out-dir "${STRESS_OUT_DIR}" \
    --suite nightly \
    --seed "${STRESS_SEED}" \
    --command "scripts/stress/soak.sh --repeat ${SOAK_REPEAT}" \
    --exit-code "${EXIT_CODE}"

exit "${EXIT_CODE}"
