# Running Benchmarks

This document explains how to run the **Phase 3 Side-by-Side LLM Comparison Benchmarking Suite** for the Unified Cognitive Stack.

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [API Keys](#api-keys)
4. [Running the Full Suite](#running-the-full-suite)
5. [Running Individual Benchmarks](#running-individual-benchmarks)
6. [Interpreting Results](#interpreting-results)
7. [Cost Estimates](#cost-estimates)
8. [Expected Runtime](#expected-runtime)
9. [Output Files](#output-files)

---

## Overview

The benchmarking suite compares the **Unified Cognitive Stack** (8-limb and 16-limb variants) against leading LLMs across five dimensions:

| Benchmark | File | Description |
|-----------|------|-------------|
| CCL Comparison | `ccl_model_comparison.py` | 300 CCL tasks at L1/L2/L3 depth |
| Extended Domains | `extended_domain_benchmarks.py` | Reasoning, Language, Spatial, Planning, Multi |
| Composition Stress | `composition_stress_test.py` | Rule depth 1–5, compositionality cliff |
| Performance | `performance_comparison.py` | Latency p50/p99, throughput, memory, cost |
| Domain Coverage | `domain_coverage_analysis.py` | Capability matrix (model × domain) |

---

## Prerequisites

```bash
# Install Python dependencies
pip install -r requirements.txt

# Optional: chart generation
pip install matplotlib
```

The unified-stack models run locally and require no API keys.
External LLMs require API keys (see next section).

---

## API Keys

Set environment variables before running:

```bash
# OpenAI (for GPT-4)
export OPENAI_API_KEY="sk-..."

# Anthropic (for Claude 3 Opus and Claude 3.5-Sonnet)
export ANTHROPIC_API_KEY="sk-ant-..."
```

**Without API keys**, external LLM calls fall back to deterministic mock responses.  
This is useful for development, testing, and cost-free CI runs.  
Mock responses replicate the expected performance characteristics described in the literature
(LLM collapse at L2/L3).

---

## Running the Full Suite

```bash
# Run all benchmarks on all models (requires API keys for GPT-4 / Claude)
python -m benchmarks.run_all_benchmarks

# Run only on local models (no API keys needed)
python -m benchmarks.run_all_benchmarks --models unified-stack,unified-stack-16limb

# Run on a specific subset of models
python -m benchmarks.run_all_benchmarks --models unified-stack,gpt-4

# Skip specific benchmarks
python -m benchmarks.run_all_benchmarks --skip performance,domain-coverage

# Run benchmarks in parallel (experimental)
python -m benchmarks.run_all_benchmarks --parallel

# Suppress the final report
python -m benchmarks.run_all_benchmarks --no-report

# Verbose logging
python -m benchmarks.run_all_benchmarks --log-level DEBUG
```

---

## Running Individual Benchmarks

### CCL Model Comparison (300 tasks)

```bash
python -m benchmarks.ccl_model_comparison
```

Results → `benchmarks/results/ccl_comparison_results.json`

**Resume support**: the script saves after each model.  
Re-running automatically skips completed models.

---

### Extended Domain Benchmarks

```bash
python -m benchmarks.extended_domain_benchmarks
```

Results → `benchmarks/results/extended_domain_results.json`

---

### Composition Stress Test

```bash
python -m benchmarks.composition_stress_test
```

Results → `benchmarks/results/composition_stress_results.json`

---

### Performance Comparison

```bash
python -m benchmarks.performance_comparison
```

Results → `benchmarks/results/performance_comparison_results.json`

---

### Domain Coverage Analysis

```bash
python -m benchmarks.domain_coverage_analysis
```

Results → `benchmarks/results/domain_coverage_results.json`

---

### Generate Report Only

```bash
python -m benchmarks.benchmark_reporter
```

Reads all existing result JSON files and produces:
- `benchmarks/results/BENCHMARK_COMPARISON.md`
- `benchmarks/results/comparison_results.json`
- `benchmarks/results/charts/*.png` (requires `matplotlib`)

---

## Interpreting Results

### CES (Compounding Efficiency Score)

```
CES = L3_accuracy / L1_accuracy
```

| Score | Interpretation |
|-------|----------------|
| 0.95–1.00 | Excellent compositional generalisation |
| 0.50–0.94 | Moderate degradation at higher depths |
| 0.01–0.49 | Significant compositional collapse |
| < 0.01 | Near-complete collapse (typical of LLMs at L3) |

**Expected results:**

| Model | L1 | L2 | L3 | CES |
|-------|----|----|----|-----|
| unified-stack | ~1.000 | ~0.99 | ~0.95 | ~1.000 |
| unified-stack-16limb | ~1.000 | ~0.99 | ~0.96 | ~1.000 |
| gpt-4 | ~0.80 | ~0.05 | ~0.001 | ~0.001 |
| claude-3-opus | ~0.80 | ~0.05 | ~0.002 | ~0.002 |
| claude-3.5-sonnet | ~0.80 | ~0.05 | ~0.001 | ~0.001 |

---

### Domain Coverage Matrix

```
✅  native   — accuracy ≥ 75%
⚠️  partial  — accuracy 40–74%
❌  fails    — accuracy < 40%
```

Expected:

| Model | Reasoning | Language | Spatial | Planning |
|-------|-----------|----------|---------|----------|
| unified-stack | ✅ | ✅ | ✅ | ✅ |
| gpt-4 | ⚠️ | ✅ | ⚠️ | ⚠️ |
| claude-3.5-sonnet | ⚠️ | ✅ | ⚠️ | ⚠️ |

---

### Compositionality Cliff

The cliff chart (`compositionality_cliff.png`) shows success rate vs rule depth:

- Unified stack → flat line near 1.0
- GPT-4 / Claude → steep drop from depth 2 onward

---

## Cost Estimates

Running the full suite with real API calls:

| Scope | GPT-4 | Claude 3 Opus | Claude 3.5-Sonnet |
|-------|-------|---------------|-------------------|
| CCL (300 tasks) | ~$1.50 | ~$4.50 | ~$0.25 |
| Extended (50 tasks) | ~$0.25 | ~$0.75 | ~$0.05 |
| Stress (100 tasks) | ~$0.50 | ~$1.50 | ~$0.10 |
| Performance (50 calls) | ~$0.25 | ~$0.75 | ~$0.05 |
| **Full suite** | **~$2.50** | **~$7.50** | **~$0.45** |

Use `--models unified-stack` to run at zero cost.

---

## Expected Runtime

| Configuration | Runtime |
|---------------|---------|
| Local models only | 1–5 minutes |
| Full suite (with API keys, sequential) | 2–6 hours |
| Full suite (with API keys, parallel) | 45–90 minutes |

---

## Output Files

```
benchmarks/results/
├── ccl_comparison_results.json          # CCL per-task data + summaries
├── extended_domain_results.json         # Domain accuracy per model
├── composition_stress_results.json      # Success rate at each depth
├── performance_comparison_results.json  # Latency, throughput, memory
├── domain_coverage_results.json         # Coverage matrix
├── llm_cache.json                       # API response cache
├── BENCHMARK_COMPARISON.md             # Full markdown report
├── comparison_results.json             # Machine-readable aggregation
└── charts/
    ├── ces_comparison.png              # CES bar chart
    ├── latency_comparison.png          # p50/p99 latency
    ├── domain_coverage_heatmap.png     # Model × domain heatmap
    ├── compositionality_cliff.png      # Success rate vs depth
    └── cost_efficiency.png             # Cost vs efficiency scatter
```

---

## Running Tests

```bash
python -m pytest -q tests/test_model_comparison.py
```

All 10+ tests run without API keys using deterministic mock responses.
