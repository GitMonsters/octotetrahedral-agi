#!/usr/bin/env bash
# scripts/stress/repro_seeded.sh
#
# Reproducibility Seeded-Reruns Test
#
# Runs task generation 3 times with the same seed and verifies that all
# three runs produce identical outputs (hash-stable).
#
# Usage:
#   scripts/stress/repro_seeded.sh [--num-tasks N] [--out-dir DIR]
#
# Environment:
#   PYTHONHASHSEED  (default: 0)
#   STRESS_SEED     (default: 1337)
#   STRESS_OUT_DIR  artifact directory
#
# Exit codes:
#   0  all 3 runs produced identical hashes
#   1  hash mismatch detected

set -euo pipefail

export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export STRESS_SEED="${STRESS_SEED:-1337}"
STRESS_TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"

NUM_TASKS=500
OUT_DIR_OVERRIDE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --num-tasks) NUM_TASKS="$2"; shift 2 ;;
        --out-dir)   OUT_DIR_OVERRIDE="$2"; shift 2 ;;
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
LOGFILE="${STRESS_OUT_DIR}/logs.txt"
exec > >(tee -a "${LOGFILE}") 2>&1

echo "=========================================="
echo "  Reproducibility Seeded-Reruns Test"
echo "  Timestamp  : ${STRESS_TIMESTAMP}"
echo "  Seed       : ${STRESS_SEED}"
echo "  Num tasks  : ${NUM_TASKS}"
echo "  Out dir    : ${STRESS_OUT_DIR}"
echo "=========================================="

REPRO_DIR="${STRESS_OUT_DIR}/repro_seeded"
mkdir -p "${REPRO_DIR}"
HASHES_FILE="${REPRO_DIR}/run_hashes.txt"
> "${HASHES_FILE}"

ALL_PASS=true

for RUN in 1 2 3; do
    echo ""
    echo "--- Run ${RUN}/3 ---"
    OUT_FILE="${REPRO_DIR}/tasks_run${RUN}.jsonl"
    HASH=$(python -m eval_harness generate \
        --seed "${STRESS_SEED}" \
        --num-tasks "${NUM_TASKS}" \
        --output "${OUT_FILE}" \
        | grep "Task hash:" | awk '{print $NF}')
    echo "  Hash: ${HASH}"
    printf "%s  run%s\n" "${HASH}" "${RUN}" >> "${HASHES_FILE}"
done

echo ""
echo "--- Hash comparison ---"
cat "${HASHES_FILE}"

UNIQUE_HASHES=$(awk '{print $1}' "${HASHES_FILE}" | sort -u | wc -l)

DIFF_FILE="${REPRO_DIR}/repro_diff.txt"
if [[ "${UNIQUE_HASHES}" -eq 1 ]]; then
    echo "PASS: All 3 runs produced identical hashes — determinism confirmed"
    printf "OK\n" > "${DIFF_FILE}"
    EXIT_CODE=0
else
    echo "FAIL: ${UNIQUE_HASHES} unique hashes found — non-determinism detected"
    diff "${REPRO_DIR}/tasks_run1.jsonl" "${REPRO_DIR}/tasks_run2.jsonl" \
        >> "${DIFF_FILE}" 2>&1 || true
    EXIT_CODE=1
fi

python scripts/stress/collect_artifacts.py \
    --out-dir "${STRESS_OUT_DIR}" \
    --suite nightly \
    --seed "${STRESS_SEED}" \
    --command "scripts/stress/repro_seeded.sh --num-tasks ${NUM_TASKS}" \
    --exit-code "${EXIT_CODE}"

exit "${EXIT_CODE}"
