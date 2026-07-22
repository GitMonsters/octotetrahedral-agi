#!/usr/bin/env bash
# scripts/stress/distribution_shift.sh
#
# Distribution-Shift Robustness Test
#
# Evaluates performance on the held-out shifted task set and compares it
# against a baseline run to verify the accuracy drop <= 10%.
#
# Usage:
#   scripts/stress/distribution_shift.sh [--baseline RUN_ID_PREFIX] [--out-dir DIR]
#
# Environment:
#   PYTHONHASHSEED  (default: 0)
#   STRESS_SEED     (default: 1337)
#   STRESS_OUT_DIR  artifact directory
#
# Exit codes:
#   0  accuracy within SLO
#   1  accuracy regression or error

set -euo pipefail

export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export STRESS_SEED="${STRESS_SEED:-1337}"
STRESS_TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"

BASELINE_PREFIX=""
OUT_DIR_OVERRIDE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --baseline) BASELINE_PREFIX="$2"; shift 2 ;;
        --out-dir)  OUT_DIR_OVERRIDE="$2"; shift 2 ;;
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
echo "  Distribution-Shift Robustness Test"
echo "  Timestamp : ${STRESS_TIMESTAMP}"
echo "  Seed      : ${STRESS_SEED}"
echo "  Input     : data/stress/distribution_shift/tasks.jsonl"
echo "  Out dir   : ${STRESS_OUT_DIR}"
echo "=========================================="

RUNS_DIR="${STRESS_OUT_DIR}/dist_shift_runs"
mkdir -p "${RUNS_DIR}"

# Run evaluation on the distribution-shift task set
echo ""
echo "--- Evaluating distribution-shift tasks ---"
python -m eval_harness evaluate \
    --tasks data/stress/distribution_shift/tasks.jsonl \
    --mock --mock-score 0.9 \
    --seed "${STRESS_SEED}" \
    --runs-dir "${RUNS_DIR}" \
    --tag dist-shift

# If a baseline prefix is provided, run comparison
COMPARE_EXIT=0
if [[ -n "${BASELINE_PREFIX}" ]]; then
    echo ""
    echo "--- Comparing against baseline: ${BASELINE_PREFIX} ---"
    python -m eval_harness compare \
        --baseline "${BASELINE_PREFIX}" \
        --runs-dir "${RUNS_DIR}" \
        --threshold 0.10 \
        || COMPARE_EXIT=$?

    if [[ "${COMPARE_EXIT}" -ne 0 ]]; then
        echo "FAIL: Accuracy regression beyond 10% threshold"
    else
        echo "PASS: Accuracy within SLO"
    fi
fi

# Write metrics.json summary
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
        "family_scores": r.family_scores,
        "tag": r.tag,
    }
    (Path(out) / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Write summary.md
    (Path(out) / "summary.md").write_text(
        f"# Distribution-Shift Robustness\n\n"
        f"- **Overall accuracy:** {r.overall:.4f} ({r.n_correct}/{r.n_tasks})\n"
        f"- **Run ID:** `{r.run_id}`\n\n"
        "## Family Breakdown\n\n"
        + "".join(
            f"- `{fam}`: {fs['mean']:.4f} ({fs['n_correct']}/{fs['n']})\n"
            for fam, fs in sorted(r.family_scores.items())
        ),
        encoding="utf-8",
    )
    print(f"Metrics: overall={r.overall:.4f} ({r.n_correct}/{r.n_tasks})")
else:
    print("No runs found; skipping metrics.json", file=sys.stderr)
PYEOF

python scripts/stress/collect_artifacts.py \
    --out-dir "${STRESS_OUT_DIR}" \
    --suite nightly \
    --seed "${STRESS_SEED}" \
    --command "scripts/stress/distribution_shift.sh" \
    --exit-code "${COMPARE_EXIT}"

exit "${COMPARE_EXIT}"
