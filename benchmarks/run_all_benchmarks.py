"""Benchmark orchestrator: run all benchmarks and generate the final report.

Usage:
    python -m benchmarks.run_all_benchmarks
    python -m benchmarks.run_all_benchmarks --models unified-stack,gpt-4
    python -m benchmarks.run_all_benchmarks --skip ccl-comparison,performance
    python -m benchmarks.run_all_benchmarks --no-report
"""

from __future__ import annotations

import argparse
import concurrent.futures
import logging
import sys
import time
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Benchmark registry
# ---------------------------------------------------------------------------

def _run_ccl(models: list[str]) -> None:
    from benchmarks.ccl_model_comparison import run_ccl_comparison
    run_ccl_comparison(models=models)


def _run_extended(models: list[str]) -> None:
    from benchmarks.extended_domain_benchmarks import run_extended_benchmarks
    run_extended_benchmarks(models=models)


def _run_stress(models: list[str]) -> None:
    from benchmarks.composition_stress_test import run_composition_stress_test
    run_composition_stress_test(models=models)


def _run_performance(models: list[str]) -> None:
    from benchmarks.performance_comparison import run_performance_comparison
    run_performance_comparison(models=models)


def _run_coverage(models: list[str]) -> None:
    from benchmarks.domain_coverage_analysis import run_domain_coverage_analysis
    run_domain_coverage_analysis()


BENCHMARKS: dict[str, Callable[[list[str]], None]] = {
    "ccl-comparison": _run_ccl,
    "extended-domains": _run_extended,
    "composition-stress": _run_stress,
    "performance": _run_performance,
    "domain-coverage": _run_coverage,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full Phase 3 LLM comparison benchmarking suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--models",
        default=None,
        help=(
            "Comma-separated list of models to benchmark. "
            "Choices: unified-stack, unified-stack-16limb, gpt-4, claude-3-opus, claude-3.5-sonnet. "
            "Default: all models."
        ),
    )
    parser.add_argument(
        "--skip",
        default="",
        help=(
            f"Comma-separated list of benchmarks to skip. "
            f"Choices: {', '.join(BENCHMARKS.keys())}."
        ),
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Do not generate the final markdown/JSON report.",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run benchmarks for different models in parallel (experimental).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from benchmarks.llm_config import ALL_MODELS
    models: list[str] = (
        [m.strip() for m in args.models.split(",") if m.strip()]
        if args.models
        else ALL_MODELS
    )

    skip: set[str] = {s.strip() for s in args.skip.split(",") if s.strip()}
    to_run = {k: v for k, v in BENCHMARKS.items() if k not in skip}

    if not to_run:
        logger.error("All benchmarks skipped. Nothing to do.")
        return 1

    logger.info("Models: %s", models)
    logger.info("Benchmarks: %s", list(to_run.keys()))

    overall_start = time.perf_counter()
    failed: list[str] = []

    if args.parallel:
        # Run all benchmarks concurrently (each benchmark is independent)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(fn, models): name for name, fn in to_run.items()}
            for future in concurrent.futures.as_completed(futures):
                name = futures[future]
                try:
                    future.result()
                    logger.info("✅ %s complete", name)
                except Exception as exc:
                    logger.error("❌ %s failed: %s", name, exc)
                    failed.append(name)
    else:
        for name, fn in to_run.items():
            logger.info("▶ Running %s …", name)
            t0 = time.perf_counter()
            try:
                fn(models)
                elapsed = time.perf_counter() - t0
                logger.info("✅ %s complete (%.1fs)", name, elapsed)
            except Exception as exc:
                logger.error("❌ %s failed: %s", name, exc)
                failed.append(name)

    # Cost estimate
    try:
        from benchmarks.llm_config import estimate_cost
        total_tasks = 300 + 50 + 100 + 50  # CCL + extended + stress + perf
        cost_estimate = estimate_cost(total_tasks, models)
        logger.info("Estimated API cost: %s", {m: f"${c:.4f}" for m, c in cost_estimate.items()})
    except Exception:
        pass

    # Generate report
    if not args.no_report:
        logger.info("▶ Generating final report …")
        try:
            from benchmarks.benchmark_reporter import generate_report
            artefacts = generate_report()
            logger.info("📊 Report: %s", artefacts.get("report_md"))
        except Exception as exc:
            logger.error("Report generation failed: %s", exc)
            failed.append("report")

    total_elapsed = time.perf_counter() - overall_start
    logger.info("Total runtime: %.1fs", total_elapsed)

    if failed:
        logger.error("Failed benchmarks: %s", failed)
        return 1

    logger.info("🎉 All benchmarks complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
