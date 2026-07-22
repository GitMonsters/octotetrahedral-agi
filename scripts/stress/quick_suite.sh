#!/usr/bin/env bash
# scripts/stress/quick_suite.sh
#
# PR Quick Stress Suite (~15 min)
#
# Runs the fast subset of stress tests suitable for PR gating:
#   1. Existing lightweight pytest suite (eval harness + workflow + unified)
#   2. Seeded reproducibility smoke (2 identical generate runs, hash check)
#   3. Distribution-shift quick sample (20 tasks, mock outputs)
#   4. Fault-injection smoke (health-check with malformed inputs)
#
# Usage:
#   scripts/stress/quick_suite.sh [--out-dir DIR]
#
# Environment (optional, sane defaults provided):
#   PYTHONHASHSEED   fixed to 0 for determinism
#   STRESS_SEED      task seed (default: 1337)
#   STRESS_OUT_DIR   artifact output directory
#
# Exit codes:
#   0  all checks passed
#   1  one or more checks failed

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export STRESS_SEED="${STRESS_SEED:-1337}"
STRESS_TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"

# Allow --out-dir override
OUT_DIR_OVERRIDE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --out-dir) OUT_DIR_OVERRIDE="$2"; shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

if [[ -n "$OUT_DIR_OVERRIDE" ]]; then
    STRESS_OUT_DIR="$OUT_DIR_OVERRIDE"
else
    STRESS_OUT_DIR="${STRESS_OUT_DIR:-artifacts/stress/quick/${STRESS_TIMESTAMP}}"
fi
export STRESS_OUT_DIR

mkdir -p "${STRESS_OUT_DIR}"

# Log both to terminal and file
LOGFILE="${STRESS_OUT_DIR}/logs.txt"
exec > >(tee -a "${LOGFILE}") 2>&1

echo "=========================================="
echo "  PR Quick Stress Suite"
echo "  Timestamp : ${STRESS_TIMESTAMP}"
echo "  Seed      : ${STRESS_SEED}"
echo "  Out dir   : ${STRESS_OUT_DIR}"
echo "=========================================="

OVERALL_EXIT=0

# ---------------------------------------------------------------------------
# Step 1: Existing lightweight pytest suite
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 1: Lightweight pytest suite ---"
python -m pytest tests/test_eval_harness.py -v --tb=short \
    --junitxml="${STRESS_OUT_DIR}/junit_eval_harness.xml" \
    || { echo "FAIL: test_eval_harness"; OVERALL_EXIT=1; }

python -m pytest tests/test_workflow.py -q --tb=short \
    --junitxml="${STRESS_OUT_DIR}/junit_workflow.xml" \
    || { echo "FAIL: test_workflow"; OVERALL_EXIT=1; }

python -m pytest tests/test_unified.py -q --tb=short \
    --junitxml="${STRESS_OUT_DIR}/junit_unified.xml" \
    || { echo "FAIL: test_unified"; OVERALL_EXIT=1; }

# Merge junit XMLs into a single file for CI upload
python - <<'PYEOF'
import glob, os, sys
from pathlib import Path

out = os.environ.get("STRESS_OUT_DIR", "artifacts/stress/quick")
xmls = sorted(glob.glob(f"{out}/junit_*.xml"))
if not xmls:
    sys.exit(0)

# Simple concatenation wrapper — CI tools accept multi-suite junit XML
lines = ['<?xml version="1.0" encoding="utf-8"?>\n<testsuites>\n']
for xml in xmls:
    content = Path(xml).read_text(encoding="utf-8")
    # strip the xml declaration if present
    for ln in content.splitlines(keepends=True):
        if not ln.strip().startswith("<?xml"):
            lines.append(ln)
    lines.append("\n")
lines.append("</testsuites>\n")
Path(f"{out}/junit.xml").write_text("".join(lines), encoding="utf-8")
print(f"Merged junit.xml written to {out}/junit.xml")
PYEOF

# ---------------------------------------------------------------------------
# Step 2: Seeded reproducibility smoke (2 runs, hash check)
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 2: Seeded reproducibility smoke ---"
REPRO_DIR="${STRESS_OUT_DIR}/repro_smoke"
mkdir -p "${REPRO_DIR}"

HASH1=$(python -m eval_harness generate \
    --seed "${STRESS_SEED}" --num-tasks 20 \
    --output "${REPRO_DIR}/tasks_run1.jsonl" \
    | grep "Task hash:" | awk '{print $NF}')

HASH2=$(python -m eval_harness generate \
    --seed "${STRESS_SEED}" --num-tasks 20 \
    --output "${REPRO_DIR}/tasks_run2.jsonl" \
    | grep "Task hash:" | awk '{print $NF}')

echo "Run 1 hash: ${HASH1}"
echo "Run 2 hash: ${HASH2}"
printf "%s  run1\n%s  run2\n" "${HASH1}" "${HASH2}" > "${REPRO_DIR}/run_hashes.txt"

if [[ "${HASH1}" == "${HASH2}" && -n "${HASH1}" ]]; then
    echo "PASS: hashes match — determinism confirmed"
else
    echo "FAIL: hash mismatch — non-determinism detected"
    diff "${REPRO_DIR}/tasks_run1.jsonl" "${REPRO_DIR}/tasks_run2.jsonl" \
        > "${REPRO_DIR}/repro_diff.txt" 2>&1 || true
    OVERALL_EXIT=1
fi

# ---------------------------------------------------------------------------
# Step 3: Distribution-shift quick sample (mock outputs)
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 3: Distribution-shift quick sample ---"
DIST_RUNS="${STRESS_OUT_DIR}/dist_shift_runs"
mkdir -p "${DIST_RUNS}"

python -m eval_harness evaluate \
    --tasks data/stress/distribution_shift/tasks.jsonl \
    --mock --mock-score 0.9 \
    --runs-dir "${DIST_RUNS}" \
    --tag stress-dist-shift \
    || { echo "FAIL: distribution_shift evaluate"; OVERALL_EXIT=1; }

# Write a summary metrics.json for this step
python - <<'PYEOF'
import json, os, sys
from pathlib import Path
from eval_harness.tracker import load_runs

out = os.environ.get("STRESS_OUT_DIR", ".")
runs_dir = Path(f"{out}/dist_shift_runs")
runs = load_runs(runs_dir)
if runs:
    r = runs[-1]
    metrics = {
        "step": "distribution_shift",
        "run_id": r.run_id,
        "overall": r.overall,
        "n_tasks": r.n_tasks,
        "n_correct": r.n_correct,
        "tag": r.tag,
    }
    (Path(out) / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Metrics written: overall={r.overall:.4f} ({r.n_correct}/{r.n_tasks})")
else:
    print("No runs found; skipping metrics.json", file=sys.stderr)
PYEOF

# ---------------------------------------------------------------------------
# Step 4: Fault-injection smoke (health-check with malformed inputs)
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 4: Fault-injection smoke ---"
FAULT_DIR="${STRESS_OUT_DIR}/fault_injection"
mkdir -p "${FAULT_DIR}"

bash scripts/stress/fault_injection.sh --quick \
    --out-dir "${FAULT_DIR}" \
    || { echo "FAIL: fault_injection"; OVERALL_EXIT=1; }

# ---------------------------------------------------------------------------
# Finalize artifacts
# ---------------------------------------------------------------------------
echo ""
echo "--- Collecting artifacts ---"
python scripts/stress/collect_artifacts.py \
    --out-dir "${STRESS_OUT_DIR}" \
    --suite quick \
    --seed "${STRESS_SEED}" \
    --command "scripts/stress/quick_suite.sh" \
    --exit-code "${OVERALL_EXIT}"

echo ""
if [[ "${OVERALL_EXIT}" -eq 0 ]]; then
    echo "=========================================="
    echo "  Quick suite PASSED"
    echo "  Artifacts: ${STRESS_OUT_DIR}"
    echo "=========================================="
else
    echo "=========================================="
    echo "  Quick suite FAILED (exit ${OVERALL_EXIT})"
    echo "  Artifacts: ${STRESS_OUT_DIR}"
    echo "=========================================="
fi

exit "${OVERALL_EXIT}"
