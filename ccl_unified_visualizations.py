"""Visualizations for CCL unified benchmark results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_results(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _get_level_metric(results: dict, level: str, metric: str) -> float:
    return results["level_summary"].get(level, {}).get(metric, 0.0)


def _plot_with_matplotlib(results: dict, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    levels = ["L1", "L2", "L3"]
    coherence = [_get_level_metric(results, level, "avg_coherence") for level in levels]
    coupling = [_get_level_metric(results, level, "avg_coupling_strength") for level in levels]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(levels, coherence, marker="o", label="Coherence")
    ax.plot(levels, coupling, marker="s", label="Coupling")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("CCL Level Metrics")
    ax.set_ylabel("Score")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "ccl_level_metrics.png", dpi=150)
    plt.close(fig)

    limb_counts = [results["limb_utilization"].get(str(index), 0) for index in range(8)]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([str(index) for index in range(8)], limb_counts)
    ax.set_title("Limb Utilization")
    ax.set_xlabel("Limb index")
    ax.set_ylabel("Activation count")
    fig.tight_layout()
    fig.savefig(output_dir / "ccl_limb_utilization.png", dpi=150)
    plt.close(fig)

    rules = sorted(results["rule_routing"])
    heatmap = []
    for rule in rules:
        row = [results["rule_routing"][rule].get(str(index), 0) for index in range(8)]
        heatmap.append(row)

    fig, ax = plt.subplots(figsize=(10, max(4, len(rules) * 0.4)))
    im = ax.imshow(heatmap, cmap="viridis")
    ax.set_xticks(range(8), [str(index) for index in range(8)])
    ax.set_yticks(range(len(rules)), rules)
    ax.set_title("Rule Routing Heatmap")
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(output_dir / "ccl_rule_routing_heatmap.png", dpi=150)
    plt.close(fig)


def _write_text_fallback(results: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Visualization fallback summary",
        "",
        "matplotlib not available; generated data tables instead.",
        "",
        "## Level metrics",
    ]
    for level in ["L1", "L2", "L3"]:
        data = results["level_summary"].get(level, {})
        lines.append(
            f"- {level}: coherence={data.get('avg_coherence', 0.0):.3f}, "
            f"coupling={data.get('avg_coupling_strength', 0.0):.3f}, accuracy={data.get('avg_accuracy', 0.0):.3f}"
        )

    lines.append("\n## Limb utilization")
    for index in range(8):
        lines.append(f"- Limb {index}: {results['limb_utilization'].get(str(index), 0)}")

    lines.append("\n## Rule routing")
    for rule, counts in sorted(results["rule_routing"].items()):
        lines.append(
            f"- {rule}: "
            + ", ".join(
                f"limb{limb}={count}" for limb, count in sorted(counts.items(), key=lambda kv: int(kv[0]))
            )
        )

    (output_dir / "ccl_visualization_summary.md").write_text("\n".join(lines), encoding="utf-8")


def generate_visualizations(results_path: Path, output_dir: Path) -> None:
    results = _load_results(results_path)
    try:
        _plot_with_matplotlib(results, output_dir)
    except (ModuleNotFoundError, ImportError):
        _write_text_fallback(results, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate CCL unified benchmark visualizations")
    parser.add_argument("--results", default="ccl_unified_results.json")
    parser.add_argument("--output-dir", default="ccl_visualizations")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_visualizations(Path(args.results), Path(args.output_dir))
    print(f"Visualizations generated in {args.output_dir}")


if __name__ == "__main__":
    main()
