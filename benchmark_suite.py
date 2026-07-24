"""Production-grade LLM performance benchmark suite.

Compares OctoTetrahedral AGI against Claude 3.5 Sonnet, GPT-4, Claude 3 Opus,
Gemini 2.0, Llama 2, Mistral, and Phi-3 across latency, throughput, accuracy,
memory, cost, and energy metrics.

Usage
-----
Run all models and all scenarios:
    python benchmark_suite.py

Select specific models:
    python benchmark_suite.py --models octotetrahedral,gpt-4,gemini-2.0

Select specific scenarios:
    python benchmark_suite.py --scenarios single_latency,reasoning

Adjust sample count:
    python benchmark_suite.py --n-samples 5

Start the live dashboard after the run:
    python benchmark_suite.py --dashboard

Only start the dashboard (skip benchmarking):
    python benchmark_suite.py --dashboard-only

Environment variables for API keys:
    OCTO_API_KEY       — OctoTetrahedral AGI API key
    OPENAI_API_KEY     — OpenAI GPT-4
    ANTHROPIC_API_KEY  — Anthropic Claude models
    GEMINI_API_KEY     — Google Gemini 2.0
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

from benchmark_models import (
    BENCHMARK_MODELS,
    _ENERGY_WH_PER_1K_TOKENS,
    _RESPONSE_TOKENS,
    build_benchmark_clients,
    estimate_cost_per_1m,
)
from benchmark_metrics import (
    compute_model_metrics,
    peak_memory_mb,
    save_csv,
    save_json,
    summary_table,
)
from benchmark_report import (
    RESULTS_DIR,
    generate_html_report,
    generate_markdown_summary,
)
from benchmark_tasks import run_all_scenarios
from benchmarks.llm_config import CostTracker, ResponseCache

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(
    models: list[str] | None = None,
    scenarios: list[str] | None = None,
    n_samples: int = 10,
    output_dir: Path | str = RESULTS_DIR,
    octo_api_url: str = "http://localhost:8000",
    octo_api_key: str = "",
) -> dict[str, Any]:
    """Run the full benchmark suite and return aggregated metrics.

    Parameters
    ----------
    models:       models to benchmark (default: all 8)
    scenarios:    scenarios to run (default: all 6)
    n_samples:    latency sample count per model
    output_dir:   directory for results files
    octo_api_url: base URL for the OctoTetrahedral AGI API
    octo_api_key: API key for OctoTetrahedral AGI
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models = models or BENCHMARK_MODELS
    logger.info(
        "Starting benchmark: %d model(s), scenarios=%s, n_samples=%d",
        len(models),
        scenarios or "all",
        n_samples,
    )

    cache = ResponseCache(cache_path=str(output_dir / "benchmark_cache.json"))
    tracker = CostTracker()

    clients = build_benchmark_clients(
        models=models,
        cache=cache,
        cost_tracker=tracker,
        octo_api_url=octo_api_url,
        octo_api_key=octo_api_key,
    )

    metrics_by_model: dict[str, dict[str, Any]] = {}
    overall_start = time.perf_counter()

    for model_name, client in clients.items():
        logger.info("\u25b6 Benchmarking: %s", model_name)
        mem_before = peak_memory_mb()
        t0 = time.perf_counter()

        scenario_results = run_all_scenarios(client, scenarios=scenarios)

        elapsed = time.perf_counter() - t0
        peak_mem = max(0.0, peak_memory_mb() - mem_before)

        model_metrics = compute_model_metrics(
            model=model_name,
            scenario_results=scenario_results,
            cost_per_1m=estimate_cost_per_1m(model_name),
            energy_wh_per_1k_tokens=_ENERGY_WH_PER_1K_TOKENS.get(model_name, 0.002),
            peak_mem_mb=peak_mem,
            n_output_tokens=_RESPONSE_TOKENS.get(model_name, 100),
        )
        model_metrics["benchmark_elapsed_s"] = elapsed
        metrics_by_model[model_name] = model_metrics

        logger.info(
            "  \u2705 %s done in %.1fs | latency=%.1f ms | accuracy=%.3f | cost/1M=$%.4f",
            model_name,
            elapsed,
            model_metrics.get("latency_ms", 0.0),
            model_metrics.get("accuracy", 0.0),
            model_metrics.get("cost_per_1m_tokens_usd", 0.0),
        )

    total_elapsed = time.perf_counter() - overall_start

    final_results: dict[str, Any] = {
        "metadata": {
            "models": models,
            "scenarios": scenarios or "all",
            "n_samples": n_samples,
            "total_elapsed_s": total_elapsed,
            "api_cost_summary": tracker.summary(),
        },
        "metrics": metrics_by_model,
    }

    # Persist all output formats
    results_json = output_dir / "results.json"
    results_csv = output_dir / "results.csv"
    results_html = output_dir / "report.html"
    results_md = output_dir / "summary.md"

    save_json(final_results, results_json)
    save_csv(metrics_by_model, results_csv)
    generate_html_report(metrics_by_model, results_html)
    generate_markdown_summary(metrics_by_model, results_md)

    logger.info(
        "\U0001f389 Benchmark complete in %.1f s\n"
        "  JSON \u2192 %s\n  CSV  \u2192 %s\n  HTML \u2192 %s\n  MD   \u2192 %s",
        total_elapsed,
        results_json,
        results_csv,
        results_html,
        results_md,
    )
    logger.info("\n%s", summary_table(metrics_by_model))

    return final_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OctoTetrahedral AGI vs LLM Performance Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python benchmark_suite.py\n"
            "  python benchmark_suite.py --models octotetrahedral,gpt-4\n"
            "  python benchmark_suite.py --scenarios single_latency,reasoning --n-samples 5\n"
            "  python benchmark_suite.py --dashboard\n"
        ),
    )
    parser.add_argument(
        "--models",
        default=None,
        help=(
            "Comma-separated model names to benchmark. "
            f"Available: {', '.join(BENCHMARK_MODELS)}. "
            "Default: all 8 models."
        ),
    )
    parser.add_argument(
        "--scenarios",
        default=None,
        help=(
            "Comma-separated scenario names. "
            "Available: single_latency, batch_processing, concurrent_requests, "
            "long_context, few_shot, reasoning. "
            "Default: all 6 scenarios."
        ),
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10,
        help="Number of samples for latency measurement (default: 10).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(RESULTS_DIR),
        help="Output directory for results files (default: benchmark_results/).",
    )
    parser.add_argument(
        "--octo-api-url",
        default="http://localhost:8000",
        help="OctoTetrahedral AGI API base URL (default: http://localhost:8000).",
    )
    parser.add_argument(
        "--octo-api-key",
        default="",
        help="OctoTetrahedral AGI API key (overrides OCTO_API_KEY env var).",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Start the live dashboard server after benchmarking.",
    )
    parser.add_argument(
        "--dashboard-only",
        action="store_true",
        help="Skip benchmarking and only start the live dashboard server.",
    )
    parser.add_argument(
        "--dashboard-port",
        type=int,
        default=8001,
        help="Dashboard server port (default: 8001).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    return parser.parse_args(argv)


def _start_dashboard(results_json: Path, port: int = 8001) -> None:
    try:
        import uvicorn  # type: ignore

        from benchmark_report import create_dashboard_app

        app = create_dashboard_app(results_json)
        logger.info("Dashboard \u2192 http://localhost:%d/dashboard", port)
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
    except ImportError:
        logger.error(
            "uvicorn + fastapi required for the dashboard. "
            "Install with: pip install uvicorn fastapi"
        )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    output_dir = Path(args.output_dir)

    if not args.dashboard_only:
        models = (
            [m.strip() for m in args.models.split(",") if m.strip()]
            if args.models
            else None
        )
        scenarios = (
            [s.strip() for s in args.scenarios.split(",") if s.strip()]
            if args.scenarios
            else None
        )
        try:
            run_benchmark(
                models=models,
                scenarios=scenarios,
                n_samples=args.n_samples,
                output_dir=output_dir,
                octo_api_url=args.octo_api_url,
                octo_api_key=args.octo_api_key,
            )
        except Exception as exc:
            logger.error("Benchmark failed: %s", exc, exc_info=True)
            return 1

    if args.dashboard or args.dashboard_only:
        _start_dashboard(output_dir / "results.json", port=args.dashboard_port)

    return 0


if __name__ == "__main__":
    sys.exit(main())
