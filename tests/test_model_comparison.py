"""Integration tests for the Phase 3 model comparison benchmarking suite.

Tests use deterministic mock models so no API keys are required.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_response(model: str, correct: bool, latency: float = 15.0) -> dict[str, Any]:
    return {
        "answer": "correct" if correct else "incorrect",
        "correct": correct,
        "coherence": 0.98 if "unified" in model else None,
        "latency_ms": latency,
        "model": model,
    }


# ---------------------------------------------------------------------------
# 1. LLM config tests
# ---------------------------------------------------------------------------

class TestResponseCache:
    def test_cache_stores_and_retrieves(self, tmp_path: Path) -> None:
        from benchmarks.llm_config import ResponseCache

        cache = ResponseCache(cache_path=tmp_path / "cache.json")
        cache.set("gpt-4", "hello", {"answer": "world"})
        assert cache.get("gpt-4", "hello") == {"answer": "world"}

    def test_cache_miss_returns_none(self, tmp_path: Path) -> None:
        from benchmarks.llm_config import ResponseCache

        cache = ResponseCache(cache_path=tmp_path / "cache.json")
        assert cache.get("gpt-4", "nonexistent") is None

    def test_cache_persists_across_instances(self, tmp_path: Path) -> None:
        from benchmarks.llm_config import ResponseCache

        path = tmp_path / "cache.json"
        cache1 = ResponseCache(cache_path=path)
        cache1.set("model-a", "prompt", {"v": 42})

        cache2 = ResponseCache(cache_path=path)
        assert cache2.get("model-a", "prompt") == {"v": 42}


class TestCostTracker:
    def test_records_cost_for_llm(self) -> None:
        from benchmarks.llm_config import CostTracker

        tracker = CostTracker()
        tracker.record("gpt-4")
        summary = tracker.summary()
        assert summary["cost_usd"]["gpt-4"] > 0.0
        assert summary["api_calls"]["gpt-4"] == 1

    def test_zero_cost_for_unified_stack(self) -> None:
        from benchmarks.llm_config import CostTracker

        tracker = CostTracker()
        tracker.record("unified-stack")
        assert tracker.summary()["cost_usd"]["unified-stack"] == 0.0

    def test_total_cost_accumulates(self) -> None:
        from benchmarks.llm_config import CostTracker

        tracker = CostTracker()
        tracker.record("gpt-4")
        tracker.record("gpt-4")
        assert tracker.summary()["api_calls"]["gpt-4"] == 2

    def test_estimate_cost_returns_zero_for_local(self) -> None:
        from benchmarks.llm_config import estimate_cost

        costs = estimate_cost(1000, ["unified-stack", "unified-stack-16limb"])
        assert costs["unified-stack"] == 0.0
        assert costs["unified-stack-16limb"] == 0.0

    def test_estimate_cost_positive_for_external(self) -> None:
        from benchmarks.llm_config import estimate_cost

        costs = estimate_cost(1_000_000, ["gpt-4"])
        assert costs["gpt-4"] > 0.0


# ---------------------------------------------------------------------------
# 2. CCL comparison tests
# ---------------------------------------------------------------------------

class TestCCLComparison:
    def test_generates_300_tasks(self) -> None:
        from benchmarks.ccl_model_comparison import generate_ccl_tasks

        tasks = generate_ccl_tasks()
        assert len(tasks) == 300
        levels = [t["level"] for t in tasks]
        assert levels.count(1) == 100
        assert levels.count(2) == 100
        assert levels.count(3) == 100

    def test_task_structure(self) -> None:
        from benchmarks.ccl_model_comparison import generate_ccl_tasks

        task = generate_ccl_tasks()[0]
        assert "task_id" in task
        assert "level" in task
        assert "rules" in task
        assert "prompt" in task

    def test_ccl_aggregate_computes_ces(self, tmp_path: Path) -> None:
        from benchmarks.ccl_model_comparison import _aggregate

        raw = (
            [{"task_id": f"t{i}", "level": 1, "rules": ["r"], "correct": True, "coherence": 0.9, "latency_ms": 10} for i in range(10)]
            + [{"task_id": f"t{i}", "level": 2, "rules": ["r", "r"], "correct": True, "coherence": 0.9, "latency_ms": 10} for i in range(10)]
            + [{"task_id": f"t{i}", "level": 3, "rules": ["r", "r", "r"], "correct": False, "coherence": None, "latency_ms": 10} for i in range(10)]
        )
        summary = _aggregate(raw)
        assert summary["L1"]["accuracy"] == 1.0
        assert summary["L2"]["accuracy"] == 1.0
        assert summary["L3"]["accuracy"] == 0.0
        assert summary["CES"] == 0.0

    def test_ccl_run_saves_json(self, tmp_path: Path) -> None:
        from benchmarks.ccl_model_comparison import run_ccl_comparison

        output = tmp_path / "ccl_out.json"
        with patch("benchmarks.llm_config.ModelClient.call") as mock_call:
            mock_call.return_value = _make_mock_response("unified-stack", True)
            result = run_ccl_comparison(
                models=["unified-stack"],
                output_path=output,
                seed=0,
            )
        assert output.exists()
        assert "models" in result
        assert "unified-stack" in result["models"]


# ---------------------------------------------------------------------------
# 3. Extended domain benchmark tests
# ---------------------------------------------------------------------------

class TestExtendedDomains:
    def test_domain_tasks_generated(self) -> None:
        from benchmarks.extended_domain_benchmarks import _generate_domain_tasks

        tasks = _generate_domain_tasks()
        domains = {t["domain"] for t in tasks}
        assert "reasoning" in domains
        assert "language" in domains
        assert "spatial" in domains
        assert "planning" in domains
        assert "multi" in domains

    def test_extended_run_saves_json(self, tmp_path: Path) -> None:
        from benchmarks.extended_domain_benchmarks import run_extended_benchmarks

        output = tmp_path / "ext_out.json"
        with patch("benchmarks.llm_config.ModelClient.call") as mock_call:
            mock_call.return_value = _make_mock_response("unified-stack", True)
            result = run_extended_benchmarks(models=["unified-stack"], output_path=output)
        assert output.exists()
        assert "unified-stack" in result["models"]


# ---------------------------------------------------------------------------
# 4. Composition stress test
# ---------------------------------------------------------------------------

class TestCompositionStress:
    def test_generates_correct_task_count(self) -> None:
        from benchmarks.composition_stress_test import _generate_stress_tasks, MAX_DEPTH, TASKS_PER_DEPTH

        tasks = _generate_stress_tasks()
        assert len(tasks) == MAX_DEPTH * TASKS_PER_DEPTH

    def test_stress_run_saves_json(self, tmp_path: Path) -> None:
        from benchmarks.composition_stress_test import run_composition_stress_test

        output = tmp_path / "stress_out.json"
        with patch("benchmarks.llm_config.ModelClient.call") as mock_call:
            mock_call.return_value = _make_mock_response("unified-stack", True)
            result = run_composition_stress_test(models=["unified-stack"], output_path=output)
        assert output.exists()
        assert "unified-stack" in result["models"]
        by_depth = result["models"]["unified-stack"]["by_depth"]
        assert "1" in by_depth
        assert "5" in by_depth


# ---------------------------------------------------------------------------
# 5. Performance comparison tests
# ---------------------------------------------------------------------------

class TestPerformanceComparison:
    def test_profile_model_returns_metrics(self, tmp_path: Path) -> None:
        from benchmarks.performance_comparison import profile_model
        from benchmarks.llm_config import ModelClient, ResponseCache, CostTracker

        cache = ResponseCache(cache_path=tmp_path / "c.json")
        tracker = CostTracker()
        client = ModelClient("unified-stack", cache=cache, cost_tracker=tracker)
        metrics = profile_model(client, n_samples=5, batch_size=3)
        assert metrics["latency"]["p50_ms"] >= 0
        assert metrics["latency"]["p99_ms"] >= metrics["latency"]["p50_ms"]
        assert metrics["throughput"]["single_tps"] > 0

    def test_performance_run_saves_json(self, tmp_path: Path) -> None:
        from benchmarks.performance_comparison import run_performance_comparison

        output = tmp_path / "perf_out.json"
        result = run_performance_comparison(models=["unified-stack"], output_path=output, n_samples=5)
        assert output.exists()
        assert "unified-stack" in result["models"]


# ---------------------------------------------------------------------------
# 6. Domain coverage analysis tests
# ---------------------------------------------------------------------------

class TestDomainCoverage:
    def test_coverage_matrix_keys(self) -> None:
        from benchmarks.domain_coverage_analysis import build_coverage_matrix, DOMAINS
        from benchmarks.llm_config import ALL_MODELS

        fake_extended = {
            "models": {
                m: {
                    "domain_summary": {d: {"accuracy": 0.9 if "unified" in m else 0.3} for d in DOMAINS}
                }
                for m in ALL_MODELS
            }
        }
        matrix = build_coverage_matrix(fake_extended)
        for model in ALL_MODELS:
            assert model in matrix
            for domain in DOMAINS:
                assert domain in matrix[model]
                assert matrix[model][domain] in ("native", "partial", "fails")

    def test_coverage_run_saves_json(self, tmp_path: Path) -> None:
        from benchmarks.domain_coverage_analysis import run_domain_coverage_analysis, DOMAINS
        from benchmarks.llm_config import ALL_MODELS

        fake_extended = {
            "models": {
                m: {"domain_summary": {d: {"accuracy": 0.8} for d in DOMAINS}}
                for m in ALL_MODELS
            }
        }
        output = tmp_path / "cov_out.json"
        result = run_domain_coverage_analysis(extended_results=fake_extended, output_path=output)
        assert output.exists()
        assert "matrix" in result
        assert "text_table" in result


# ---------------------------------------------------------------------------
# 7. Report generation tests
# ---------------------------------------------------------------------------

class TestBenchmarkReporter:
    def test_generate_report_creates_files(self, tmp_path: Path) -> None:
        from benchmarks.benchmark_reporter import generate_report

        aggregated: dict[str, Any] = {}
        report_md = tmp_path / "report.md"
        report_json = tmp_path / "report.json"
        charts_dir = tmp_path / "charts"

        result = generate_report(
            aggregated=aggregated,
            report_md=report_md,
            report_json=report_json,
            charts_dir=charts_dir,
        )
        assert report_md.exists()
        assert report_json.exists()
        assert "report_md" in result

    def test_markdown_contains_sections(self, tmp_path: Path) -> None:
        from benchmarks.benchmark_reporter import generate_report

        report_md = tmp_path / "report.md"
        generate_report(aggregated={}, report_md=report_md,
                        report_json=tmp_path / "r.json", charts_dir=tmp_path / "c")
        content = report_md.read_text()
        assert "Benchmark Comparison Report" in content
        assert "Executive Summary" in content


# ---------------------------------------------------------------------------
# 8. Orchestrator tests
# ---------------------------------------------------------------------------

class TestOrchestrator:
    def test_main_returns_zero_on_success(self, tmp_path: Path) -> None:
        from benchmarks.run_all_benchmarks import main

        # Run with a single fast model; patch the actual benchmark functions
        # so no heavy computation or API calls occur.
        with patch("benchmarks.run_all_benchmarks._run_ccl", return_value=None), \
             patch("benchmarks.run_all_benchmarks._run_extended", return_value=None), \
             patch("benchmarks.run_all_benchmarks._run_stress", return_value=None), \
             patch("benchmarks.run_all_benchmarks._run_performance", return_value=None), \
             patch("benchmarks.run_all_benchmarks._run_coverage", return_value=None):
            ret = main([
                "--models", "unified-stack",
                "--no-report",
            ])
        assert ret == 0

    def test_main_returns_one_when_all_skipped(self) -> None:
        from benchmarks.run_all_benchmarks import main, BENCHMARKS

        skip_all = ",".join(BENCHMARKS.keys())
        ret = main(["--models", "unified-stack", "--skip", skip_all, "--no-report"])
        assert ret == 1
