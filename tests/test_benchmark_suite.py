"""Unit tests for the production benchmark suite.

Tests use deterministic mock clients so no API keys, local models,
or running services are required.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _mock_response(model: str, correct: bool = True, latency: float = 50.0) -> dict[str, Any]:
    return {
        "answer": "correct" if correct else "incorrect",
        "correct": correct,
        "coherence": None,
        "latency_ms": latency,
        "model": model,
    }


def _make_mock_client(model: str = "octotetrahedral", latency: float = 50.0) -> MagicMock:
    client = MagicMock()
    client.model = model
    client.call.return_value = _mock_response(model, latency=latency)
    return client


# ---------------------------------------------------------------------------
# benchmark_models
# ---------------------------------------------------------------------------

class TestBenchmarkModels:
    def test_benchmark_models_list(self) -> None:
        from benchmark_models import BENCHMARK_MODELS

        assert "octotetrahedral" in BENCHMARK_MODELS
        assert "gpt-4" in BENCHMARK_MODELS
        assert "gemini-2.0" in BENCHMARK_MODELS
        assert "llama-2" in BENCHMARK_MODELS
        assert "mistral" in BENCHMARK_MODELS
        assert "phi-3" in BENCHMARK_MODELS
        assert len(BENCHMARK_MODELS) == 8

    def test_client_rejects_unknown_model(self) -> None:
        from benchmark_models import BenchmarkModelClient

        with pytest.raises(ValueError, match="Unknown model"):
            BenchmarkModelClient("nonexistent-model-xyz")

    def test_mock_response_is_deterministic(self) -> None:
        from benchmark_models import BenchmarkModelClient, ResponseCache, CostTracker
        from benchmarks.llm_config import ResponseCache as RC, CostTracker as CT

        # Use a temp cache so we hit the mock path, not disk
        import tempfile, pathlib
        with tempfile.TemporaryDirectory() as td:
            cache = RC(cache_path=str(pathlib.Path(td) / "c.json"))
            tracker = CT()
            client = BenchmarkModelClient(
                "phi-3", cache=cache, cost_tracker=tracker, max_retries=1, retry_delay=0
            )
            r1 = client._mock_response("hello", "phi-3")
            r2 = client._mock_response("hello", "phi-3")
            assert r1["latency_ms"] == r2["latency_ms"]
            assert r1["correct"] == r2["correct"]

    def test_estimate_cost_per_1m(self) -> None:
        from benchmark_models import estimate_cost_per_1m

        assert estimate_cost_per_1m("octotetrahedral") == 0.0
        assert estimate_cost_per_1m("llama-2") == 0.0
        assert estimate_cost_per_1m("gpt-4") > 0.0

    def test_estimate_energy_wh(self) -> None:
        from benchmark_models import estimate_energy_wh

        e_octo = estimate_energy_wh("octotetrahedral", 1000)
        e_gpt4 = estimate_energy_wh("gpt-4", 1000)
        assert e_octo < e_gpt4  # Apple Silicon should use less energy than large cloud model

    def test_build_benchmark_clients_subset(self, tmp_path: Path) -> None:
        from benchmark_models import build_benchmark_clients
        from benchmarks.llm_config import ResponseCache as RC, CostTracker as CT

        cache = RC(cache_path=str(tmp_path / "c.json"))
        clients = build_benchmark_clients(
            models=["octotetrahedral", "phi-3"],
            cache=cache,
        )
        assert set(clients.keys()) == {"octotetrahedral", "phi-3"}

    def test_client_falls_back_to_mock_on_error(self, tmp_path: Path) -> None:
        from benchmark_models import BenchmarkModelClient
        from benchmarks.llm_config import ResponseCache as RC, CostTracker as CT

        cache = RC(cache_path=str(tmp_path / "c.json"))
        client = BenchmarkModelClient(
            "gpt-4", cache=cache, max_retries=1, retry_delay=0
        )
        # _call_openai will raise ImportError or RuntimeError; client must return mock
        result = client.call("test prompt")
        assert "latency_ms" in result
        assert result["model"] == "gpt-4"


# ---------------------------------------------------------------------------
# benchmark_tasks
# ---------------------------------------------------------------------------

class TestBenchmarkTasks:
    def test_single_latency_returns_stats(self) -> None:
        from benchmark_tasks import run_single_latency

        client = _make_mock_client(latency=80.0)
        result = run_single_latency(client, n_warmup=1, n_samples=5)
        assert result["scenario"] == "single_latency"
        assert result["n_samples"] == 5
        assert len(result["latencies_ms"]) == 5
        assert result["mean_ms"] > 0
        assert result["p99_ms"] >= result["p95_ms"]

    def test_batch_processing_keys(self) -> None:
        from benchmark_tasks import run_batch_processing

        client = _make_mock_client(latency=50.0)
        result = run_batch_processing(client, batch_sizes=(10, 100))
        assert result["scenario"] == "batch_processing"
        assert "10" in result["batches"]
        assert "100" in result["batches"]
        assert result["batches"]["10"]["requests_per_sec"] > 0

    def test_concurrent_requests_keys(self) -> None:
        from benchmark_tasks import run_concurrent_requests

        client = _make_mock_client(latency=50.0)
        result = run_concurrent_requests(
            client, concurrency_levels=(5,), requests_per_level=5
        )
        assert result["scenario"] == "concurrent_requests"
        assert "5" in result["levels"]
        assert result["levels"]["5"]["requests"] == 5

    def test_long_context_keys(self) -> None:
        from benchmark_tasks import run_long_context

        client = _make_mock_client(latency=200.0)
        result = run_long_context(client, context_sizes_k=(2,), n_samples=2)
        assert result["scenario"] == "long_context"
        assert "2k" in result["contexts"]
        assert result["contexts"]["2k"]["mean_latency_ms"] > 0

    def test_few_shot_accuracy_range(self) -> None:
        from benchmark_tasks import run_few_shot

        # Mock that always returns "13" in the answer
        client = MagicMock()
        client.model = "test"
        client.call.return_value = {"answer": "13", "latency_ms": 40.0}
        result = run_few_shot(client, n_samples=4)
        assert result["scenario"] == "few_shot"
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_reasoning_structure(self) -> None:
        from benchmark_tasks import run_reasoning, _MMLU_SAMPLES, _ARC_SAMPLES

        client = _make_mock_client()
        result = run_reasoning(client)
        assert result["scenario"] == "reasoning"
        expected = len(_MMLU_SAMPLES) + len(_ARC_SAMPLES)
        assert result["total_tasks"] == expected
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_run_all_scenarios_subset(self) -> None:
        from benchmark_tasks import run_all_scenarios

        client = _make_mock_client()
        results = run_all_scenarios(client, scenarios=["single_latency", "reasoning"])
        assert "single_latency" in results
        assert "reasoning" in results
        assert "batch_processing" not in results

    def test_run_all_scenarios_all(self) -> None:
        from benchmark_tasks import run_all_scenarios, _ALL_SCENARIO_NAMES

        client = _make_mock_client()
        results = run_all_scenarios(client)
        for name in _ALL_SCENARIO_NAMES:
            assert name in results

    def test_make_long_context_prompt_length(self) -> None:
        from benchmark_tasks import _make_long_context_prompt

        prompt = _make_long_context_prompt(8192)
        # Rough check: should be at least 8 K chars (4 chars/token approximation)
        assert len(prompt) >= 8000


# ---------------------------------------------------------------------------
# benchmark_metrics
# ---------------------------------------------------------------------------

class TestBenchmarkMetrics:
    def _sample_scenarios(self) -> dict[str, Any]:
        return {
            "single_latency": {
                "scenario": "single_latency",
                "latencies_ms": [50.0, 55.0, 60.0],
                "mean_ms": 55.0,
                "p99_ms": 60.0,
            },
            "reasoning": {
                "scenario": "reasoning",
                "accuracy": 0.8,
                "mean_latency_ms": 55.0,
            },
        }

    def test_compute_model_metrics_keys(self) -> None:
        from benchmark_metrics import compute_model_metrics

        m = compute_model_metrics(
            model="test",
            scenario_results=self._sample_scenarios(),
            cost_per_1m=0.0,
            energy_wh_per_1k_tokens=0.001,
            peak_mem_mb=100.0,
            n_output_tokens=50,
        )
        for key in [
            "model", "latency_ms", "latency_p95_ms", "latency_p99_ms",
            "throughput_rps", "tokens_per_sec", "memory_mb",
            "cost_per_1m_tokens_usd", "accuracy", "energy_wh_per_1k_tokens",
            "efficiency_score", "scenarios",
        ]:
            assert key in m, f"Missing key: {key}"

    def test_tokens_per_sec_positive(self) -> None:
        from benchmark_metrics import estimate_tokens_per_second

        assert estimate_tokens_per_second(100.0, 50) == pytest.approx(500.0)

    def test_tokens_per_sec_zero_latency(self) -> None:
        from benchmark_metrics import estimate_tokens_per_second

        assert estimate_tokens_per_second(0.0, 100) == 0.0

    def test_to_csv_contains_headers(self) -> None:
        from benchmark_metrics import to_csv, _CSV_COLUMNS

        data = {"model-a": {"model": "model-a", "latency_ms": 10.0}}
        csv_text = to_csv(data)
        for col in _CSV_COLUMNS:
            assert col in csv_text

    def test_save_csv_creates_file(self, tmp_path: Path) -> None:
        from benchmark_metrics import save_csv

        path = tmp_path / "out.csv"
        save_csv({"m1": {"model": "m1", "latency_ms": 5.0}}, path)
        assert path.exists()
        assert "model" in path.read_text()

    def test_save_json_creates_file(self, tmp_path: Path) -> None:
        from benchmark_metrics import save_json

        path = tmp_path / "out.json"
        save_json({"key": "value"}, path)
        assert path.exists()
        loaded = json.loads(path.read_text())
        assert loaded["key"] == "value"

    def test_rank_models(self) -> None:
        from benchmark_metrics import rank_models

        data = {
            "fast-model": {"latency_ms": 10.0},
            "slow-model": {"latency_ms": 500.0},
        }
        ranked = rank_models(data, key="latency_ms", ascending=True)
        assert ranked[0][0] == "fast-model"
        assert ranked[1][0] == "slow-model"

    def test_summary_table_string(self) -> None:
        from benchmark_metrics import summary_table

        data = {"model-a": {"latency_ms": 10.0, "throughput_rps": 5.0, "accuracy": 0.9}}
        table = summary_table(data)
        assert "model-a" in table


# ---------------------------------------------------------------------------
# benchmark_report
# ---------------------------------------------------------------------------

class TestBenchmarkReport:
    def _sample_metrics(self) -> dict[str, Any]:
        return {
            "octotetrahedral": {
                "model": "octotetrahedral",
                "latency_ms": 80.0,
                "latency_p95_ms": 90.0,
                "latency_p99_ms": 95.0,
                "throughput_rps": 12.5,
                "tokens_per_sec": 625.0,
                "memory_mb": 100.0,
                "cost_per_1m_tokens_usd": 0.0,
                "accuracy": 0.85,
                "energy_wh_per_1k_tokens": 0.001,
                "efficiency_score": 85.0,
                "scenarios": {},
            },
            "gpt-4": {
                "model": "gpt-4",
                "latency_ms": 2000.0,
                "latency_p95_ms": 3000.0,
                "latency_p99_ms": 4000.0,
                "throughput_rps": 0.5,
                "tokens_per_sec": 75.0,
                "memory_mb": 0.0,
                "cost_per_1m_tokens_usd": 45.0,
                "accuracy": 0.90,
                "energy_wh_per_1k_tokens": 0.003,
                "efficiency_score": 0.02,
                "scenarios": {},
            },
        }

    def test_html_report_created(self, tmp_path: Path) -> None:
        from benchmark_report import generate_html_report

        path = tmp_path / "report.html"
        result = generate_html_report(self._sample_metrics(), path)
        assert Path(result).exists()
        content = Path(result).read_text()
        assert "octotetrahedral" in content
        assert "Chart.js" in content or "chart.umd.min.js" in content

    def test_html_report_contains_models(self, tmp_path: Path) -> None:
        from benchmark_report import generate_html_report

        path = tmp_path / "report.html"
        generate_html_report(self._sample_metrics(), path)
        content = path.read_text()
        assert "gpt-4" in content

    def test_markdown_summary_created(self, tmp_path: Path) -> None:
        from benchmark_report import generate_markdown_summary

        path = tmp_path / "summary.md"
        result = generate_markdown_summary(self._sample_metrics(), path)
        assert Path(result).exists()
        content = Path(result).read_text()
        assert "OctoTetrahedral" in content
        assert "Key Findings" in content

    def test_markdown_contains_all_models(self, tmp_path: Path) -> None:
        from benchmark_report import generate_markdown_summary

        path = tmp_path / "summary.md"
        generate_markdown_summary(self._sample_metrics(), path)
        content = path.read_text()
        for model in self._sample_metrics():
            assert model in content

    def test_dashboard_app_importable(self, tmp_path: Path) -> None:
        pytest.importorskip("fastapi")
        from benchmark_report import create_dashboard_app

        app = create_dashboard_app(tmp_path / "results.json")
        assert app is not None


# ---------------------------------------------------------------------------
# benchmark_suite integration
# ---------------------------------------------------------------------------

class TestBenchmarkSuite:
    def test_run_benchmark_produces_outputs(self, tmp_path: Path) -> None:
        from benchmark_suite import run_benchmark

        with patch("benchmark_tasks.run_single_latency") as mock_sl, \
             patch("benchmark_tasks.run_batch_processing") as mock_bp, \
             patch("benchmark_tasks.run_concurrent_requests") as mock_cr, \
             patch("benchmark_tasks.run_long_context") as mock_lc, \
             patch("benchmark_tasks.run_few_shot") as mock_fs, \
             patch("benchmark_tasks.run_reasoning") as mock_r:

            mock_sl.return_value = {"scenario": "single_latency", "latencies_ms": [50.0], "mean_ms": 50.0}
            mock_bp.return_value = {"scenario": "batch_processing", "batches": {}}
            mock_cr.return_value = {"scenario": "concurrent_requests", "levels": {}}
            mock_lc.return_value = {"scenario": "long_context", "contexts": {}}
            mock_fs.return_value = {"scenario": "few_shot", "accuracy": 0.8, "mean_latency_ms": 50.0}
            mock_r.return_value = {"scenario": "reasoning", "accuracy": 0.7, "mean_latency_ms": 50.0, "total_tasks": 7, "correct": 5}

            result = run_benchmark(
                models=["octotetrahedral"],
                scenarios=["single_latency"],
                n_samples=2,
                output_dir=tmp_path,
            )

        assert "metadata" in result
        assert "metrics" in result
        assert "octotetrahedral" in result["metrics"]
        assert (tmp_path / "results.json").exists()
        assert (tmp_path / "results.csv").exists()
        assert (tmp_path / "report.html").exists()
        assert (tmp_path / "summary.md").exists()

    def test_main_cli_no_args_returns_zero(self, tmp_path: Path) -> None:
        from benchmark_suite import main

        with patch("benchmark_suite.run_benchmark") as mock_rb:
            mock_rb.return_value = {"metadata": {}, "metrics": {}}
            ret = main(["--models", "octotetrahedral", "--output-dir", str(tmp_path), "--scenarios", "reasoning"])
        assert ret == 0

    def test_main_cli_bad_model_returns_nonzero(self, tmp_path: Path) -> None:
        from benchmark_suite import main
        from benchmark_models import BenchmarkModelClient

        # build_benchmark_clients will raise ValueError for unknown model
        ret = main(["--models", "nonexistent-xyz", "--output-dir", str(tmp_path)])
        assert ret == 1
