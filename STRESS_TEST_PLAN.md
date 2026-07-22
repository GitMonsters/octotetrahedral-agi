# Stress Test Plan: Reasoning Robustness, Runtime Stability, Reproducibility, and CI Health

## Scope

- Validate reasoning behavior under harder/shifted inputs.
- Validate runtime stability under latency limits and long runs.
- Validate deterministic reproducibility with fixed seeds/environment.
- Validate CI signal quality for PR gating and nightly reliability checks.

## Stress Test Matrix

| Category | Objective | Input set | Command template | Timeout | Pass criteria | Artifacts |
|---|---|---|---|---|---|---|
| Distribution shift robustness | Detect performance drop on shifted-but-valid task distribution | Held-out shifted tasks (`data/stress/distribution_shift/*.jsonl`) | `python -m eval_harness evaluate --input {INPUT} --seed {SEED} --output {OUT}` | 20 min | Accuracy drop <= 10% vs baseline; no crash | `metrics.json`, `summary.md`, raw predictions |
| Latency-bounded tiers | Verify predictable runtime per tier (smoke/standard/heavy) | Same corpus at 3 size tiers | `python workflow.py --mode evaluate --input {INPUT} --max-tasks {N}` | smoke: 2 min, std: 10 min, heavy: 30 min | p95 latency meets tier SLO; completion rate >= 99% | `latency.csv`, `resource_usage.json` |
| Adversarial perturbation invariance | Ensure minor perturbations do not cause disproportionate failures | Perturbed variants (`noise`, `order`, `token jitter`) | `python -m eval_harness compare --baseline {BASE} --candidate {CAND}` | 20 min | Robustness ratio >= 0.90; no systematic class collapse | `delta_report.md`, `family_breakdown.json` |
| Long-run soak | Catch memory leaks / degradation over prolonged operation | Repeated mixed workload for fixed duration | `python workflow.py --mode inference --input {INPUT} --repeat {R}` | 6 hr | No OOM/crash; throughput drift <= 15% from first-hour median to last-hour median; error rate <= 0.5% | `soak.log`, `throughput_timeseries.csv` |
| Reproducibility seeded reruns | Confirm deterministic outputs under controlled env | Same input run 3x with fixed seed | `python -m eval_harness generate --seed {SEED} --num-tasks {N} --output {OUT}` | 10 min | Hash-stable outputs across reruns; metric variance = 0 in deterministic path | `run_hashes.txt`, `repro_diff.txt` |
| Fault-injection / malformed input handling | Verify graceful handling for malformed/partial data | Invalid JSON, missing keys, empty payloads, oversized payloads | `python workflow.py --mode health-check --input {INPUT}` | 5 min | Clear error classes; no hangs; non-zero exit where expected | `error_catalog.json`, `stderr.log` |
| CI health checks | Keep PR signal fast and trustworthy; catch nightly regressions | PR quick suite + nightly extended suite | `pytest ...` + stress commands in CI jobs | PR: 15 min; nightly: 2 hr | PR suite green for merge; nightly failure triaged within SLA | CI logs, junit xml, artifact bundle |

## Command Conventions (Deterministic + Repo-Aligned)

### Standard env vars

```bash
export PYTHONHASHSEED=0
export STRESS_SEED=1337
export STRESS_TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
export STRESS_OUT_DIR="artifacts/stress/${STRESS_TIMESTAMP}"
```
Example timestamp value: `20260721T143022Z`.

### Placeholder script mapping

| Placeholder script | Likely repo command pattern |
|---|---|
| `scripts/stress/quick_suite.sh` | `python -m pytest tests/test_eval_harness.py -v && python -m pytest tests/test_workflow.py -q && python -m pytest -q tests/test_unified.py` |
| `scripts/stress/distribution_shift.sh` | `python -m eval_harness evaluate --input data/stress/distribution_shift/tasks.jsonl --seed ${STRESS_SEED} --output ${STRESS_OUT_DIR}/distribution_shift.json` |
| `scripts/stress/repro_seeded.sh` | `python -m eval_harness generate --seed ${STRESS_SEED} --num-tasks 500 --output ${STRESS_OUT_DIR}/seeded_tasks.jsonl` |
| `scripts/stress/soak.sh` | `python workflow.py --mode inference --input data/stress/soak/mix.jsonl --repeat 1000` |
| `scripts/stress/fault_injection.sh` | `python workflow.py --mode health-check --input data/stress/faults/*.json` |

## Baseline Thresholds / SLOs (Initial)

> Calibrate these after the first **3 full runs**; treat below as starting targets.

- [ ] **Reasoning quality:** shifted accuracy >= 90% of baseline.
- [ ] **Latency (PR quick):** p95 <= 2x rolling median of last 3 successful PR quick runs (allows shared-runner noise while keeping fast merge signal).
- [ ] **Latency (nightly):** p95 <= 1.5x rolling median of last 3 successful nightly runs (stricter because nightly runs should be more stable).
- [ ] **Latency (initial calibration period):** if fewer than 3 successful historical runs exist for that suite, use a temporary absolute cap from the first successful run and switch to rolling-median policy after run #3.
- [ ] **Stability:** crash rate = 0 for quick suite; <= 0.5% for nightly soak.
- [ ] **Reproducibility:** deterministic path hash mismatch rate = 0%.
- [ ] **CI health:** flaky test rate < 2% (unique tests that required at least one rerun / total unique tests executed over trailing 14 days); mean time to triage (MTTT) < 1 business day (team-defined business hours/timezone documented in workflow/job description).

## Artifact Layout

```text
artifacts/
  stress/
    quick/
      YYYYMMDDTHHMMSSZ/
        metrics.json
        summary.md
        junit.xml
        logs.txt
    nightly/
      YYYYMMDDTHHMMSSZ/
        metrics.json
        summary.md
        latency.csv
        throughput_timeseries.csv
        error_catalog.json
        run_config.json
        commit_sha.txt
        environment.txt
```

Required persisted files:
- `run_config.json` (seed/env/command)
- `commit_sha.txt`
- `metrics.json`
- `summary.md`
- Raw logs (`logs.txt` or split stderr/stdout)

## Lightweight CI Integration

### PR quick suite (required)

- Keep under ~15 minutes.
- Include:
  - deterministic seeded rerun smoke
  - quick distribution-shift sample
  - existing lightweight pytest checks
- Upload `artifacts/stress/quick/<timestamp>/...` on failure and success.

### Nightly scheduled suite

- Run extended distribution + adversarial + soak + fault-injection.
- Publish `artifacts/stress/nightly/<timestamp>/...`.
- Auto-open triage issue when SLO breach or reproducibility mismatch occurs.

## Failure Triage Playbook

### Classification

- **P0:** crash, data corruption, security-related failure.
- **P1:** major regression (accuracy/latency beyond SLO).
- **P2:** flaky/non-deterministic behavior without user impact.
- **P3:** observability/reporting gaps.

### Minimum repro checklist

- [ ] Commit SHA
- [ ] Exact command
- [ ] Seed and env vars
- [ ] Input artifact path
- [ ] Expected vs actual output snippet

### Rollback / guardrails

- [ ] Gate risky path behind a feature flag.
- [ ] Revert last offending change if P0/P1 unresolved.
- [ ] Tighten PR quick suite with new repro case.
- [ ] Keep nightly test disabled only with linked issue + owner + expiry date (`YYYY-MM-DD`) in workflow/job metadata.
- [ ] Add an expiry check in nightly workflow (e.g., `nightly-stress.yml`) that fails and opens/updates a follow-up issue when expiry passes.

## Day-1 Minimum: Top 5 Tests

1. Seeded reproducibility rerun (3 identical runs, hash check).
2. Distribution-shift quick sample against baseline.
3. Latency tier smoke (`N=small`) with p95 capture.
4. Fault-injection malformed JSON/empty payload handling.
5. 1-hour soak sanity run with throughput/error trend capture.
