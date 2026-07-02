"""Domain coverage analysis: capability matrix (model × domain).

Assigns a coverage status to each (model, domain) pair:
  - "native"   — model handles this domain well natively (✅)
  - "partial"  — works but slower or less accurate (⚠️)
  - "fails"    — consistently fails on this domain (❌)

Uses results from the extended domain benchmarks (if available) or
falls back to a threshold-based classification.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from benchmarks.extended_domain_benchmarks import (
    RESULTS_PATH as EXTENDED_RESULTS_PATH,
    run_extended_benchmarks,
)
from benchmarks.llm_config import ALL_MODELS

logger = logging.getLogger(__name__)

RESULTS_PATH = Path("benchmarks/results/domain_coverage_results.json")

DOMAINS = ["reasoning", "language", "spatial", "planning", "multi"]

# Thresholds for classifying domain coverage
NATIVE_THRESHOLD = 0.75    # accuracy ≥ 75% → native
PARTIAL_THRESHOLD = 0.40   # accuracy ≥ 40% → partial
# accuracy < 40% → fails


def _classify(accuracy: float) -> str:
    if accuracy >= NATIVE_THRESHOLD:
        return "native"
    if accuracy >= PARTIAL_THRESHOLD:
        return "partial"
    return "fails"


def _symbol(status: str) -> str:
    return {"native": "✅", "partial": "⚠️", "fails": "❌"}.get(status, "?")


def build_coverage_matrix(
    extended_results: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute the capability matrix from extended benchmark results."""
    if extended_results is None:
        ext_path = EXTENDED_RESULTS_PATH
        if ext_path.exists():
            with ext_path.open() as fh:
                extended_results = json.load(fh)
        else:
            logger.warning("Extended results not found; running benchmarks now …")
            extended_results = run_extended_benchmarks()

    model_data = extended_results.get("models", {})
    matrix: dict[str, dict[str, str]] = {}

    for model_name in ALL_MODELS:
        matrix[model_name] = {}
        domain_summary = model_data.get(model_name, {}).get("domain_summary", {})
        for domain in DOMAINS:
            domain_info = domain_summary.get(domain, {})
            accuracy = domain_info.get("accuracy", 0.0)
            matrix[model_name][domain] = _classify(accuracy)

    return matrix


def _render_text_matrix(matrix: dict[str, dict[str, str]]) -> str:
    """Render the capability matrix as a plain-text table."""
    col_w = 12
    header = f"{'Model':<25}" + "".join(f"{d:<{col_w}}" for d in DOMAINS)
    lines = [header, "-" * len(header)]
    for model, domains in matrix.items():
        row = f"{model:<25}" + "".join(
            f"{_symbol(domains.get(d, '?')):<{col_w}}" for d in DOMAINS
        )
        lines.append(row)
    return "\n".join(lines)


def run_domain_coverage_analysis(
    extended_results: dict[str, Any] | None = None,
    output_path: Path | str = RESULTS_PATH,
) -> dict[str, Any]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    matrix = build_coverage_matrix(extended_results)
    text_table = _render_text_matrix(matrix)
    logger.info("Domain coverage matrix:\n%s", text_table)

    result = {
        "matrix": matrix,
        "text_table": text_table,
        "legend": {"native": "✅ native support", "partial": "⚠️ works but degraded", "fails": "❌ consistently fails"},
        "thresholds": {"native": NATIVE_THRESHOLD, "partial": PARTIAL_THRESHOLD},
    }

    with output_path.open("w") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)

    logger.info("Domain coverage analysis complete → %s", output_path)
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_domain_coverage_analysis()
