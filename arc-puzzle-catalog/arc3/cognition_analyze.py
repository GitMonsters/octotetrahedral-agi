#!/usr/bin/env python3
"""
Cognition Analyze — Run multilayer cognition graph experiments on ARC/RE-ARC solvers.

Implements the full pipeline from the design doc:
  1. Load puzzle families from catalog.json
  2. Run solver families on task samples (with execution tracing)
  3. Build static + dynamic multilayer graphs
  4. Compute cognition metrics
  5. Generate Figures 1-6

Solver families:
  specialized  — per-puzzle hand-crafted solvers in solves/<id>/solver.py
  pipeline     — REARCPuzzleSolver (perception → rule inference → rule application)
  baseline     — always returns the test input unchanged

Usage:
    cd arc-puzzle-catalog
    python3 arc3/cognition_analyze.py
    python3 arc3/cognition_analyze.py --max-tasks 50 --output results/cognition
    python3 arc3/cognition_analyze.py --families tiling symmetry color_map
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np

# Add repo root to path
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from arc3.cognition_tracer import CognitionTracer, TraceLog
from arc3.cognition_graph import (
    ALL_MODULES,
    MODULE_COLORS,
    MultilayerCognitionGraph,
    StaticGraphBuilder,
    DynamicGraphBuilder,
    CognitionGraphCollection,
)
from arc3.cognition_metrics import (
    MetricsResult,
    NodeMetrics,
    compute_metrics,
    compute_all,
)

logger = logging.getLogger("arc3.cognition")


# ---------------------------------------------------------------------------
# Puzzle family classification
# ---------------------------------------------------------------------------

FAMILY_KEYWORDS: list[tuple[str, list[str]]] = [
    ("tiling",       ["tile", "tiling", "repeat", "extension", "continuation",
                      "periodic", "pattern extension", "fill"]),
    ("symmetry",     ["rotation", "reflect", "mirror", "symmetric", "flip",
                      "rotational", "bilateral"]),
    ("color_map",    ["color map", "remap", "recolor", "color remapping",
                      "projection", "marker beam", "beam", "color-encoded"]),
    ("counting",     ["count", "frequency", "number", "size", "tally",
                      "stacking", "stack"]),
    ("connectivity", ["connect", "path", "flood", "reach", "adjacen", "bfs",
                      "chain", "snake", "link"]),
    ("geometry",     ["shape", "bbox", "staircase", "diagonal", "border",
                      "polygon", "triangle", "rectangular", "concentric",
                      "interlocking", "portal"]),
    ("pattern",      ["pattern", "match", "detect", "occlu", "reconstruct",
                      "template", "identify", "recognition"]),
]


def classify_task_family(name: str, rule_summary: str = "") -> str:
    """Assign a puzzle family label from the task name + rule summary."""
    text = (name + " " + rule_summary).lower()
    for fam, kws in FAMILY_KEYWORDS:
        if any(kw in text for kw in kws):
            return fam
    return "other"


def load_catalog(catalog_path: str) -> list[dict]:
    with open(catalog_path) as f:
        return json.load(f)


def load_task(task_id: str, dataset_dir: str) -> Optional[dict]:
    path = os.path.join(dataset_dir, "tasks", f"{task_id}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Solver family runners
# ---------------------------------------------------------------------------

def load_specialized_solver(task_id: str, solves_dir: str):
    """Load and return the solve() function from solves/<task_id>/solver.py."""
    path = os.path.join(solves_dir, task_id, "solver.py")
    if not os.path.exists(path):
        return None, None
    spec = importlib.util.spec_from_file_location(f"solver_{task_id}", path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception:
        return None, None
    return getattr(mod, "solve", None), path


# ---------------------------------------------------------------------------
# RE-ARC helpers
# ---------------------------------------------------------------------------

def load_re_arc_data(re_arc_dir: str) -> dict[str, list[list]]:
    """
    Load RE-ARC test pairs from <re_arc_dir>/submission.json.

    Returns:
        {task_id: [[input_grid, output_grid], ...]}
    """
    path = os.path.join(re_arc_dir, "submission.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _extract_docstring_name(solver_path: str) -> str:
    """
    Extract a descriptive text snippet from a solver file for family classification.

    Strategy (in priority order):
    1. Module-level docstring — skips the "ARC Puzzle <id> Solver" header line.
    2. First function docstring in the file (e.g. the transform() docstring).
    3. First non-trivial comment lines (lines starting with #).
    4. Empty string if none of the above yield content.
    """
    try:
        with open(solver_path) as fh:
            src = fh.read()
        import ast as _ast
        tree = _ast.parse(src)

        # 1. Module-level docstring
        module_doc = _ast.get_docstring(tree)
        if module_doc:
            for line in module_doc.splitlines():
                line = line.strip()
                if line and not line.lower().startswith("arc puzzle") \
                        and not line.lower().startswith("solver for arc"):
                    return line

        # 2. First function docstring
        for node in _ast.walk(tree):
            if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                fn_doc = _ast.get_docstring(node)
                if fn_doc:
                    first = fn_doc.strip().splitlines()[0].strip()
                    if first and len(first) > 10:
                        return first

        # 3. Comment lines
        comment_parts: list[str] = []
        for line in src.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                text = stripped.lstrip("#").strip()
                if text and len(text) > 5:
                    comment_parts.append(text)
                    if len(comment_parts) >= 3:
                        break
        if comment_parts:
            return " ".join(comment_parts)

    except Exception:
        pass
    return ""


def _grid_dims(grid: list) -> tuple[float, float]:
    """Return (width, height) of a grid that may be 1D or 2D."""
    if not grid:
        return 0.0, 0.0
    if isinstance(grid[0], list):
        return float(len(grid[0])), float(len(grid))
    # 1D grid (single row of ints)
    return float(len(grid)), 1.0


def run_re_arc_specialized(
    task_id: str,
    pairs: list[list],
    family: str,
    re_arc_solves_dir: str,
    tracer: CognitionTracer,
    log: TraceLog,
    static_builder: StaticGraphBuilder,
    mlg: MultilayerCognitionGraph,
    split: str = "dev",
    checkpoint_id: str = "re_arc_v1",
) -> bool:
    """
    Run the RE-ARC per-task transform() solver and record traces.

    RE-ARC solvers live in <re_arc_solves_dir>/<task_id>/solver.py and expose
    ``transform(grid) -> grid`` (not ``solve``).  Test pairs come from
    submission.json instead of a dataset/ task JSON.
    """
    solver_path = os.path.join(re_arc_solves_dir, task_id, "solver.py")
    if not os.path.exists(solver_path):
        return False

    spec = importlib.util.spec_from_file_location(f"re_arc_solver_{task_id}", solver_path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception:
        return False

    transform_fn = getattr(mod, "transform", None)
    if transform_fn is None:
        return False

    if not pairs:
        return False

    # Avg grid dimensions from all pairs
    dims = [_grid_dims(p[0]) for p in pairs if p and p[0] is not None]
    avg_w = sum(d[0] for d in dims) / len(dims) if dims else 0.0
    avg_h = sum(d[1] for d in dims) / len(dims) if dims else 0.0

    for idx, pair in enumerate(pairs):
        inp, expected = pair[0], pair[1]
        with tracer.trace(task_id, task_id, family, "re_arc_specialized",
                          checkpoint_id=checkpoint_id, split=split):
            try:
                result = transform_fn(inp)
                success = (result == expected)
            except Exception:
                result = None
                success = False

        tracer.set_success(success)
        tracer.set_intermediate_stats(avg_grid_width=avg_w, avg_grid_height=avg_h)
        tracer.set_re_arc(
            task_id=task_id,
            example_index=idx,
            difficulty={},
        )
        log.append(tracer.last_record)
        mlg.add_records([tracer.last_record])

    # Build static layer from solver source (once per MLG)
    if mlg.static_layer is None:
        try:
            with open(solver_path) as fh:
                src = fh.read()
            g_static = static_builder.build(src, task_id)
            mlg.set_static_layer(g_static)
        except Exception:
            pass

    return True


def run_specialized(
    task_id: str,
    task: dict,
    family: str,
    solves_dir: str,
    tracer: CognitionTracer,
    log: TraceLog,
    static_builder: StaticGraphBuilder,
    mlg: MultilayerCognitionGraph,
    split: str = "dev",
    checkpoint_id: str = "v1",
) -> bool:
    solve_fn, solver_path = load_specialized_solver(task_id, solves_dir)
    if solve_fn is None:
        return False

    test_pairs = task.get("test", [])
    if not test_pairs:
        return False

    # Derive avg grid dimensions for intermediate_stats
    all_inputs = task.get("train", []) + task.get("test", [])
    widths  = [len(p["input"][0]) for p in all_inputs if p.get("input")]
    heights = [len(p["input"])    for p in all_inputs if p.get("input")]
    avg_w = sum(widths)  / len(widths)  if widths  else 0.0
    avg_h = sum(heights) / len(heights) if heights else 0.0

    for pair in test_pairs:
        with tracer.trace(task_id, task_id, family, "specialized",
                          checkpoint_id=checkpoint_id, split=split):
            try:
                result = solve_fn(pair["input"])
                success = (result == pair["output"])
            except Exception:
                result = None
                success = False
        tracer.set_success(success)
        tracer.set_intermediate_stats(avg_grid_width=avg_w, avg_grid_height=avg_h)
        log.append(tracer.last_record)
        mlg.add_records([tracer.last_record])

    # Build static layer from solver source (once per task)
    if solver_path and mlg.static_layer is None:
        try:
            with open(solver_path) as fh:
                src = fh.read()
            g_static = static_builder.build(src, task_id)
            mlg.set_static_layer(g_static)
        except Exception:
            pass

    return True


def run_pipeline(
    task_id: str,
    task: dict,
    family: str,
    tracer: CognitionTracer,
    log: TraceLog,
    mlg: MultilayerCognitionGraph,
    split: str = "dev",
    checkpoint_id: str = "pipeline_v1",
) -> bool:
    """Run REARCPuzzleSolver on a task."""
    try:
        from arc3.puzzle_solver import REARCPuzzleSolver
    except ImportError:
        return False

    solver = REARCPuzzleSolver()
    test_pairs = task.get("test", [])
    if not test_pairs:
        return False

    for pair in test_pairs:
        with tracer.trace(task_id, task_id, family, "pipeline",
                          checkpoint_id=checkpoint_id, split=split):
            try:
                pred = solver.solve_task(task)
                if pred is not None:
                    expected = pair["output"]
                    success = (np.array(pred).tolist() == expected)
                else:
                    success = False
                # Capture intermediate stats from pipeline internals where possible
                n_rules = len(solver.reasoning.rules) if (
                    solver.reasoning and hasattr(solver.reasoning, "rules")
                ) else 0
            except Exception:
                success = False
                n_rules = 0
        tracer.set_success(success)
        tracer.set_intermediate_stats(num_rules_hypothesized=n_rules)
        log.append(tracer.last_record)
        mlg.add_records([tracer.last_record])

    return True


def run_baseline(
    task_id: str,
    task: dict,
    family: str,
    tracer: CognitionTracer,
    log: TraceLog,
    mlg: MultilayerCognitionGraph,
    split: str = "dev",
    checkpoint_id: str = "baseline_v1",
) -> bool:
    """Baseline: return input unchanged."""
    test_pairs = task.get("test", [])
    if not test_pairs:
        return False

    for pair in test_pairs:
        with tracer.trace(task_id, task_id, family, "baseline",
                          checkpoint_id=checkpoint_id, split=split):
            result = [row[:] for row in pair["input"]]
            success = (result == pair["output"])
        tracer.set_success(success)
        log.append(tracer.last_record)
        mlg.add_records([tracer.last_record])

    return True


# ---------------------------------------------------------------------------
# Figure generation (Figures 1–6 from design doc)
# ---------------------------------------------------------------------------

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def figure1_module_reuse_vs_success(
    all_metrics: dict[str, MetricsResult],
    output_dir: str,
) -> None:
    """Figure 1: Node participation coefficient vs contribution to success."""
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = cm.tab10.colors
    for i, (sf, mr) in enumerate(all_metrics.items()):
        xs = [nm.participation for nm in mr.node_metrics.values()]
        # Proxy for "contribution to success": betweenness × success_rate
        ys = [nm.betweenness * mr.success_rate for nm in mr.node_metrics.values()]
        labels = list(mr.node_metrics.keys())
        ax.scatter(xs, ys, label=sf, color=colors[i % len(colors)], alpha=0.7, s=80)
        for x, y, lbl in zip(xs, ys, labels):
            ax.annotate(lbl.replace("_", "\n"), (x, y), fontsize=6, alpha=0.7,
                        textcoords="offset points", xytext=(4, 2))

    ax.set_xlabel("Participation Coefficient (P)", fontsize=12)
    ax.set_ylabel("Betweenness × Success Rate", fontsize=12)
    ax.set_title("Figure 1: Module Reuse vs Success Contribution", fontsize=13)
    ax.legend(title="Solver Family", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, "fig1_module_reuse_vs_success.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved Figure 1 → {path}")


def figure2_community_structure(
    all_metrics: dict[str, MetricsResult],
    mlgs: dict[str, MultilayerCognitionGraph],
    output_dir: str,
) -> None:
    """Figure 2: Community-labelled graph for 2–4 representative solvers."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import networkx as nx

    n_solvers = min(len(mlgs), 4)
    fig, axes = plt.subplots(1, n_solvers, figsize=(5 * n_solvers, 5))
    if n_solvers == 1:
        axes = [axes]

    community_colors = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8",
                        "#f58231", "#911eb4", "#42d4f4", "#f032e6"]

    for ax, (sf, mlg) in zip(axes, list(mlgs.items())[:n_solvers]):
        g = mlg.combined_dynamic_layer()
        mr = all_metrics.get(sf)
        if mr is None:
            ax.set_title(sf)
            continue

        # Node colour by community
        communities = {m: mr.node_metrics[m].community for m in ALL_MODULES if m in mr.node_metrics}
        unique_comms = sorted(set(communities.values()))
        comm_to_color = {c: community_colors[i % len(community_colors)]
                         for i, c in enumerate(unique_comms)}
        node_colors = [comm_to_color.get(communities.get(m, "?"), "#cccccc") for m in g.nodes()]

        # Node size by activation
        max_act = max(
            (g.nodes[m].get("activation", 1) for m in g.nodes()), default=1
        ) or 1
        node_sizes = [
            200 + 800 * (g.nodes[m].get("activation", 0) / max_act)
            for m in g.nodes()
        ]

        pos = nx.circular_layout(g)
        edge_weights = [g[u][v].get("weight", 1) for u, v in g.edges()]
        max_w = max(edge_weights, default=1) or 1
        edge_widths = [0.5 + 2.5 * (w / max_w) for w in edge_weights]

        nx.draw_networkx(
            g, pos=pos, ax=ax,
            node_color=node_colors,
            node_size=node_sizes,
            edge_color="#888888",
            width=edge_widths,
            font_size=6,
            arrows=True,
            arrowsize=12,
            labels={m: m.replace("_", "\n") for m in g.nodes()},
        )
        ax.set_title(f"Solver: {sf}\nQ={mr.modularity:.2f}", fontsize=10)
        ax.axis("off")

        patches = [mpatches.Patch(color=comm_to_color[c], label=c) for c in unique_comms]
        ax.legend(handles=patches, fontsize=6, loc="lower left")

    fig.suptitle("Figure 2: Community Structure per Solver Family", fontsize=13)
    fig.tight_layout()
    path = os.path.join(output_dir, "fig2_community_structure.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved Figure 2 → {path}")


def figure3_transcendplexity_profile(
    all_metrics: dict[str, MetricsResult],
    output_dir: str,
) -> None:
    """Figure 3: Bar chart of transcendplexity + components per solver."""
    import matplotlib.pyplot as plt
    import numpy as np

    solvers = list(all_metrics.keys())
    n = len(solvers)
    if n == 0:
        return

    components = {
        "avg_participation": [all_metrics[s].avg_participation for s in solvers],
        "cross_layer_stability": [all_metrics[s].cross_layer_stability for s in solvers],
        "modularity": [min(all_metrics[s].modularity, 1.0) for s in solvers],
        "success_rate": [all_metrics[s].success_rate for s in solvers],
    }
    weights = {"avg_participation": 0.40, "cross_layer_stability": 0.30,
                "modularity": 0.20, "success_rate": 0.10}
    comp_colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]

    x = np.arange(n)
    bar_width = 0.18

    fig, ax = plt.subplots(figsize=(max(8, n * 2), 6))
    for i, (comp, vals) in enumerate(components.items()):
        weighted = [v * weights[comp] for v in vals]
        ax.bar(x + i * bar_width, weighted, bar_width,
               label=f"{comp} (×{weights[comp]:.2f})",
               color=comp_colors[i], alpha=0.85)

    # Overlay total transcendplexity as line
    tx = [all_metrics[s].transcendplexity for s in solvers]
    ax.plot(x + 1.5 * bar_width, tx, "k-o", linewidth=2, markersize=8, label="Transcendplexity T")

    ax.set_xticks(x + 1.5 * bar_width)
    ax.set_xticklabels(solvers, fontsize=10)
    ax.set_ylabel("Score (weighted)", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_title("Figure 3: Transcendplexity Profile per Solver Family", fontsize=13)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, "fig3_transcendplexity_profile.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved Figure 3 → {path}")


def compute_training_curve(
    mlg: "MultilayerCognitionGraph",
) -> list[dict]:
    """
    Simulate a training curve by progressively adding puzzle families.

    For a solver with families [A, B, C, D], checkpoint k uses families
    [A, B, ..., family_k].  Returns a list of per-checkpoint metric dicts:
        checkpoint, family_added, n_families, success_rate,
        avg_participation, cross_layer_stability, transcendplexity,
        frac_high_participation
    Requires at least 2 families to produce a meaningful curve.
    """
    from arc3.cognition_metrics import compute_metrics
    from arc3.cognition_graph import MultilayerCognitionGraph as MLG, DynamicGraphBuilder

    sorted_families = sorted(mlg.family_layers.keys())
    if len(sorted_families) < 2:
        return []

    builder = DynamicGraphBuilder()
    curve: list[dict] = []
    for i, fam in enumerate(sorted_families, 1):
        active_fams = set(sorted_families[:i])

        # Build a sub-MLG with only the first i families
        sub = MLG(solver_family=mlg.solver_family)
        sub.static_layer = mlg.static_layer
        for f in sorted_families[:i]:
            sub.family_layers[f] = mlg.family_layers[f]
        sub.records = [r for r in mlg.records if r.puzzle_family in active_fams]

        mr = compute_metrics(sub)
        n_nodes = len(mr.node_metrics)
        frac_hp = (
            sum(1 for nm in mr.node_metrics.values() if nm.participation > 0.5) / n_nodes
            if n_nodes else 0.0
        )
        curve.append({
            "checkpoint": i,
            "family_added": fam,
            "n_families": i,
            "success_rate": mr.success_rate,
            "avg_participation": mr.avg_participation,
            "cross_layer_stability": mr.cross_layer_stability,
            "transcendplexity": mr.transcendplexity,
            "frac_high_participation": frac_hp,
        })
    return curve


def figure3_transcendplexity_over_training(
    mlgs: dict[str, "MultilayerCognitionGraph"],
    output_dir: str,
) -> None:
    """
    Figure 3 (extended): Transcendplexity over simulated training.

    X-axis  : number of puzzle families seen (simulated checkpoint).
    Y1 (left)  : test accuracy / success rate on held-out family.
    Y2 (right) : fraction of nodes with participation > 0.5.
    Y3 (right) : cross-layer community stability.

    One set of curves per solver family.  Only plotted for solvers
    with ≥ 2 puzzle families.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    solver_curves: dict[str, list[dict]] = {}
    for sf, mlg in mlgs.items():
        curve = compute_training_curve(mlg)
        if curve:
            solver_curves[sf] = curve

    if not solver_curves:
        logger.warning("Figure 3b: no solver has ≥ 2 families — skipping over-training plot")
        return

    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]
    linestyles = ["-", "--", "-.", ":"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Left panel: success rate + transcendplexity ---
    for i, (sf, curve) in enumerate(solver_curves.items()):
        xs = [pt["n_families"] for pt in curve]
        success = [pt["success_rate"] for pt in curve]
        tx = [pt["transcendplexity"] for pt in curve]
        col = colors[i % len(colors)]
        ls = linestyles[i % len(linestyles)]
        ax1.plot(xs, success, ls, color=col, linewidth=2,
                 label=f"{sf} (success)", marker="o", markersize=5)
        ax1.plot(xs, tx, ls, color=col, linewidth=1.2, alpha=0.5,
                 label=f"{sf} (T)", marker="s", markersize=4)

    ax1.set_xlabel("Number of Puzzle Families (Checkpoint)", fontsize=11)
    ax1.set_ylabel("Score", fontsize=11)
    ax1.set_ylim(-0.05, 1.1)
    ax1.set_title("Y1: Success Rate   Y2: Transcendplexity", fontsize=11)
    ax1.legend(fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)

    # --- Right panel: participation + stability ---
    for i, (sf, curve) in enumerate(solver_curves.items()):
        xs = [pt["n_families"] for pt in curve]
        fhp = [pt["frac_high_participation"] for pt in curve]
        stab = [pt["cross_layer_stability"] for pt in curve]
        col = colors[i % len(colors)]
        ls = linestyles[i % len(linestyles)]
        ax2.plot(xs, fhp, ls, color=col, linewidth=2,
                 label=f"{sf} (frac P>0.5)", marker="o", markersize=5)
        ax2.plot(xs, stab, ls, color=col, linewidth=1.2, alpha=0.5,
                 label=f"{sf} (stability)", marker="^", markersize=4)

    ax2.set_xlabel("Number of Puzzle Families (Checkpoint)", fontsize=11)
    ax2.set_ylabel("Score", fontsize=11)
    ax2.set_ylim(-0.05, 1.1)
    ax2.set_title("Y2: Fraction High-P Nodes   Y3: Cross-Layer Stability", fontsize=11)
    ax2.legend(fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Figure 3: Transcendplexity Over Simulated Training (Progressive Families)",
                 fontsize=13)
    fig.tight_layout()
    path = os.path.join(output_dir, "fig3b_transcendplexity_over_training.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved Figure 3b → {path}")


def figure4_graph_features_vs_performance(
    all_metrics: dict[str, MetricsResult],
    output_dir: str,
) -> None:
    """Figure 4: Graph features vs held-out family success rate scatter."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    features = ["avg_participation", "cross_layer_stability", "modularity"]
    feature_labels = ["Avg Participation", "Cross-Layer Stability", "Modularity Q"]
    colors = ["#e41a1c", "#377eb8", "#4daf4a"]

    for ax, feat, flbl, color in zip(axes, features, feature_labels, colors):
        xs = [getattr(mr, feat) for mr in all_metrics.values()]
        ys = [mr.success_rate for mr in all_metrics.values()]
        labels = list(all_metrics.keys())
        ax.scatter(xs, ys, color=color, s=120, zorder=3)
        for x, y, lbl in zip(xs, ys, labels):
            ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(5, 3), fontsize=9)
        ax.set_xlabel(flbl, fontsize=11)
        ax.set_ylabel("Success Rate", fontsize=11)
        ax.set_title(f"{flbl} vs Success", fontsize=11)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Figure 4: Graph Features vs Cross-Family Performance", fontsize=13)
    fig.tight_layout()
    path = os.path.join(output_dir, "fig4_features_vs_performance.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved Figure 4 → {path}")


def figure5_robustness_vs_modularity(
    all_metrics: dict[str, MetricsResult],
    output_dir: str,
) -> None:
    """Figure 5: Robustness (community stability) vs modularity per module."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    import matplotlib.cm as cm
    colors = cm.tab10.colors

    for i, (sf, mr) in enumerate(all_metrics.items()):
        xs = [mr.modularity] * len(mr.node_metrics)  # same modularity for all nodes
        ys = [nm.community_stability for nm in mr.node_metrics.values()]
        labels = list(mr.node_metrics.keys())
        # Jitter x slightly per solver to avoid overlap
        jitter = (i - len(all_metrics) / 2) * 0.01
        ax.scatter([x + jitter for x in xs], ys,
                   label=sf, color=colors[i % len(colors)], alpha=0.7, s=60)

    ax.set_xlabel("Modularity Q (solver-level)", fontsize=12)
    ax.set_ylabel("Node Community Stability", fontsize=12)
    ax.set_title("Figure 5: Robustness vs Modular Structure", fontsize=13)
    ax.legend(title="Solver Family", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, "fig5_robustness_vs_modularity.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved Figure 5 → {path}")


def figure6_architectural_radar(
    all_metrics: dict[str, MetricsResult],
    output_dir: str,
) -> None:
    """Figure 6: Radar/spider chart comparing solver families."""
    import matplotlib.pyplot as plt
    import numpy as np

    dims = ["avg_participation", "cross_layer_stability", "modularity",
            "success_rate", "transcendplexity"]
    dim_labels = ["Participation", "Stability", "Modularity", "Success", "Transcendplexity"]
    N = len(dims)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close polygon

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]

    for i, (sf, mr) in enumerate(all_metrics.items()):
        vals = [getattr(mr, d) for d in dims]
        # Clamp to [0,1]
        vals = [min(max(v, 0.0), 1.0) for v in vals]
        vals += vals[:1]
        ax.plot(angles, vals, "-o", label=sf, color=colors[i % len(colors)], linewidth=2)
        ax.fill(angles, vals, color=colors[i % len(colors)], alpha=0.1)

    ax.set_thetagrids(np.degrees(angles[:-1]), dim_labels, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title("Figure 6: Architectural Profiles", fontsize=13, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
    fig.tight_layout()
    path = os.path.join(output_dir, "fig6_architectural_radar.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved Figure 6 → {path}")


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_experiment(
    catalog_path: str,
    dataset_dir: str,
    solves_dir: str,
    output_dir: str,
    max_tasks_per_family: int = 20,
    target_families: Optional[list[str]] = None,
    solver_families: Optional[list[str]] = None,
    log_path: Optional[str] = None,
    re_arc_dir: Optional[str] = None,
) -> dict[str, MetricsResult]:
    """
    Full pipeline: load data → trace runs → build MLGs → compute metrics → plot.

    Args:
        re_arc_dir: If provided, also run re_arc_specialized on RE-ARC tasks
                    (reads <re_arc_dir>/submission.json and
                    <re_arc_dir>/solves/<id>/solver.py).
    """
    _ensure_dir(output_dir)

    if log_path is None:
        log_path = os.path.join(output_dir, "traces.jsonl")

    log = TraceLog(log_path)
    tracer = CognitionTracer()
    static_builder = StaticGraphBuilder()

    if solver_families is None:
        solver_families = ["specialized", "pipeline", "baseline"]

    # ------------------------------------------------------------------ #
    # 1. Load catalog and classify tasks into families
    # ------------------------------------------------------------------ #
    catalog = load_catalog(catalog_path)
    tasks_by_family: dict[str, list[dict]] = defaultdict(list)
    for entry in catalog:
        tid = entry["id"]
        fam = classify_task_family(entry.get("name", ""), entry.get("rule_summary", ""))
        tasks_by_family[fam].append({"id": tid, "family": fam, "name": entry.get("name", "")})

    if target_families:
        tasks_by_family = {k: v for k, v in tasks_by_family.items() if k in target_families}

    total_tasks = sum(
        min(len(v), max_tasks_per_family) for v in tasks_by_family.values()
    )
    logger.info(f"Families: {sorted(tasks_by_family)} | Tasks (capped): {total_tasks}")
    logger.info(f"Solver families: {solver_families}")

    # ------------------------------------------------------------------ #
    # 2. Build MLG containers
    # ------------------------------------------------------------------ #
    all_solver_families = list(solver_families)
    if re_arc_dir and "re_arc_specialized" not in all_solver_families:
        all_solver_families.append("re_arc_specialized")

    mlgs: dict[str, MultilayerCognitionGraph] = {
        sf: MultilayerCognitionGraph(solver_family=sf)
        for sf in all_solver_families
    }

    # ------------------------------------------------------------------ #
    # 3. Run ARC solvers — last family is held-out, rest are dev
    # ------------------------------------------------------------------ #
    all_families = sorted(tasks_by_family.keys())
    heldout_family = all_families[-1] if len(all_families) >= 2 else None
    if heldout_family:
        logger.info(f"Held-out family (cross-family eval): {heldout_family}")

    n_done = 0
    for family, task_list in sorted(tasks_by_family.items()):
        split = "heldout_family" if family == heldout_family else "dev"
        for entry in task_list[:max_tasks_per_family]:
            tid = entry["id"]
            task = load_task(tid, dataset_dir)
            if task is None:
                continue

            if "specialized" in solver_families:
                run_specialized(
                    tid, task, family, solves_dir, tracer, log,
                    static_builder, mlgs["specialized"], split=split,
                )

            if "pipeline" in solver_families:
                run_pipeline(tid, task, family, tracer, log, mlgs["pipeline"], split=split)

            if "baseline" in solver_families:
                run_baseline(tid, task, family, tracer, log, mlgs["baseline"], split=split)

            n_done += 1
            if n_done % 20 == 0:
                logger.info(f"  {n_done}/{total_tasks} tasks processed…")

    # ------------------------------------------------------------------ #
    # 3b. Run RE-ARC specialized solver (optional)
    # ------------------------------------------------------------------ #
    if re_arc_dir:
        re_arc_data = load_re_arc_data(re_arc_dir)
        re_arc_solves_dir = os.path.join(re_arc_dir, "solves")
        all_re_arc_ids = sorted(re_arc_data.keys())
        # Last 20% → heldout_family, rest → dev
        n_heldout = max(1, len(all_re_arc_ids) // 5)
        heldout_ids = set(all_re_arc_ids[-n_heldout:])
        capped_ids = all_re_arc_ids[:max_tasks_per_family * 5]  # cap total RE-ARC tasks

        logger.info(
            f"RE-ARC: {len(capped_ids)} tasks "
            f"({n_heldout} held-out), solver=re_arc_specialized"
        )

        for re_tid in capped_ids:
            pairs = re_arc_data[re_tid]
            if not pairs:
                continue
            # Classify family using solver docstring
            solver_path = os.path.join(re_arc_solves_dir, re_tid, "solver.py")
            doc_name = _extract_docstring_name(solver_path)
            fam = classify_task_family(re_tid, doc_name)
            re_split = "heldout_family" if re_tid in heldout_ids else "dev"

            run_re_arc_specialized(
                re_tid, pairs, fam,
                re_arc_solves_dir, tracer, log,
                static_builder, mlgs["re_arc_specialized"],
                split=re_split,
            )

    logger.info(f"Tracing complete. Total records logged: {sum(len(m.records) for m in mlgs.values())}")

    # ------------------------------------------------------------------ #
    # 4. Compute metrics
    # ------------------------------------------------------------------ #
    all_metrics = compute_all(mlgs)
    for sf, mr in all_metrics.items():
        logger.info(
            f"  {sf}: n={mr.n_records}, success={mr.success_rate:.1%}, "
            f"T={mr.transcendplexity:.3f}, Q={mr.modularity:.3f}, "
            f"P={mr.avg_participation:.3f}, stab={mr.cross_layer_stability:.3f}"
        )

    # ------------------------------------------------------------------ #
    # 5. Save metrics JSON
    # ------------------------------------------------------------------ #
    metrics_path = os.path.join(output_dir, "cognition_metrics.json")
    with open(metrics_path, "w") as fh:
        json.dump({sf: mr.to_dict() for sf, mr in all_metrics.items()}, fh, indent=2)
    logger.info(f"Metrics saved → {metrics_path}")

    # ------------------------------------------------------------------ #
    # 6. Generate figures
    # ------------------------------------------------------------------ #
    try:
        figure1_module_reuse_vs_success(all_metrics, output_dir)
        figure2_community_structure(all_metrics, mlgs, output_dir)
        figure3_transcendplexity_profile(all_metrics, output_dir)
        figure3_transcendplexity_over_training(mlgs, output_dir)
        figure4_graph_features_vs_performance(all_metrics, output_dir)
        figure5_robustness_vs_modularity(all_metrics, output_dir)
        figure6_architectural_radar(all_metrics, output_dir)
        logger.info("All figures generated.")
    except ImportError:
        logger.warning("matplotlib not available — skipping figure generation.")

    return all_metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multilayer Cognition Graph analysis for ARC/RE-ARC solvers"
    )
    parser.add_argument(
        "--catalog", default="catalog.json",
        help="Path to catalog.json (default: catalog.json)"
    )
    parser.add_argument(
        "--dataset-dir", default="dataset",
        help="Directory containing tasks/ subfolder (default: dataset)"
    )
    parser.add_argument(
        "--solves-dir", default="solves",
        help="Directory containing per-task solver.py files (default: solves)"
    )
    parser.add_argument(
        "--output", "-o", default="results/cognition",
        help="Output directory for figures and metrics (default: results/cognition)"
    )
    parser.add_argument(
        "--max-tasks", type=int, default=20,
        help="Max tasks per puzzle family (default: 20)"
    )
    parser.add_argument(
        "--families", nargs="*", default=None,
        metavar="FAMILY",
        help="Restrict to specific puzzle families, e.g. --families tiling symmetry"
    )
    parser.add_argument(
        "--solvers", nargs="*",
        default=["specialized", "pipeline", "baseline"],
        metavar="SOLVER",
        help="Solver families to run (default: specialized pipeline baseline)"
    )
    parser.add_argument(
        "--log", default=None,
        help="Path for trace JSONL log (default: <output>/traces.jsonl)"
    )
    parser.add_argument(
        "--re-arc", default=None, metavar="DIR",
        help=(
            "Path to re-arc directory (default: none). When provided, runs the "
            "'re_arc_specialized' solver family on RE-ARC tasks from "
            "<DIR>/submission.json + <DIR>/solves/."
        )
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    run_experiment(
        catalog_path=args.catalog,
        dataset_dir=args.dataset_dir,
        solves_dir=args.solves_dir,
        output_dir=args.output,
        max_tasks_per_family=args.max_tasks,
        target_families=args.families,
        solver_families=args.solvers,
        log_path=args.log,
        re_arc_dir=args.re_arc,
    )


if __name__ == "__main__":
    main()
