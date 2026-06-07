"""
Cognition Graph — Build static and dynamic multilayer graphs from ARC solver traces.

Three layer types:
  S  — Static structural layer: function-level call graph extracted from AST.
  T  — Dynamic family layers: aggregated call graphs for each puzzle family.

Nodes represent *semantic module types* (8 categories) so that graphs are
comparable across solver families, even when each solver uses different
function names.

Graph library: networkx (already installed in this repo's environment).
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Any, Optional

import networkx as nx

from arc3.cognition_tracer import (
    RunRecord,
    TraceEvent,
    TraceLog,
    extract_static_functions,
)


# ---------------------------------------------------------------------------
# Semantic module taxonomy
# ---------------------------------------------------------------------------

#: Ordered list of (module_name, keywords).  First match wins.
SEMANTIC_MODULES: list[tuple[str, list[str]]] = [
    ("object_segmentation", [
        "connect", "component", "flood", "segment", "blob", "region",
        "objects", "detect", "find_object", "get_object", "extract",
        "island", "cluster", "group",
    ]),
    ("spatial_reasoning", [
        "bfs", "path", "neighbor", "adjacent", "border", "edge", "walk",
        "direction", "bound", "position", "loc", "coord", "move", "grid",
        "inside", "outside", "surround",
    ]),
    ("transformation", [
        "transform", "rotat", "flip", "reflect", "mirror", "scale", "crop",
        "translat", "shift", "pad", "resize", "zoom", "expand", "contract",
        "invert", "transpose", "apply",
    ]),
    ("color_mapping", [
        "color", "remap", "recolor", "palette", "hue", "value", "map_color",
        "swap_color", "replace", "paint",
    ]),
    ("counting", [
        "count", "size", "number", "frequency", "tally", "total", "len",
        "area", "weight",
    ]),
    ("pattern_matching", [
        "pattern", "match", "compare", "similar", "hash", "normalize",
        "template", "recogni", "classify", "identify",
    ]),
    ("search", [
        "search", "hypothesis", "candidate", "try_rule", "score", "best",
        "rank", "test", "prune", "infer",
    ]),
    ("utility", [
        "solve", "main", "run", "get", "set", "build", "create", "init",
        "load", "save", "helper", "util",
    ]),
]

#: Canonical set of node labels (same across all layers)
ALL_MODULES: list[str] = [m for m, _ in SEMANTIC_MODULES]

#: Visual color map for matplotlib figures
MODULE_COLORS: dict[str, str] = {
    "object_segmentation": "#e41a1c",
    "spatial_reasoning":   "#377eb8",
    "transformation":      "#4daf4a",
    "color_mapping":       "#984ea3",
    "counting":            "#ff7f00",
    "pattern_matching":    "#a65628",
    "search":              "#f781bf",
    "utility":             "#999999",
}


def classify_function(name: str, docstring: str = "") -> str:
    """Assign a semantic module label to a function by name/docstring keywords."""
    text = (name + " " + docstring).lower()
    for mod_name, keywords in SEMANTIC_MODULES:
        if any(kw in text for kw in keywords):
            return mod_name
    return "utility"


# ---------------------------------------------------------------------------
# Static layer (Layer S) builder
# ---------------------------------------------------------------------------

class StaticGraphBuilder:
    """
    Builds a directed weighted call graph (Layer S) for a solver from its
    Python source code.

    Nodes  = semantic module labels (from ALL_MODULES).
    Edges  = directed call relationship (caller_module → callee_module).
    Weight = number of distinct function-to-function call edges that map to
             this module-to-module edge.
    """

    def build(self, source: str, solver_id: str = "") -> nx.DiGraph:
        funcs = extract_static_functions(source)
        if not funcs:
            return self._empty_graph(solver_id)

        # Map each function name → semantic module
        name_to_mod: dict[str, str] = {
            f["name"]: classify_function(f["name"], f.get("docstring", ""))
            for f in funcs
        }

        g = nx.DiGraph(solver_id=solver_id, layer="static")
        for m in ALL_MODULES:
            g.add_node(m, module=m)

        for func in funcs:
            caller_mod = name_to_mod.get(func["name"], "utility")
            for callee_name in func.get("calls", []):
                callee_mod = name_to_mod.get(callee_name, "utility")
                if caller_mod == callee_mod:
                    continue
                if g.has_edge(caller_mod, callee_mod):
                    g[caller_mod][callee_mod]["weight"] += 1
                else:
                    g.add_edge(caller_mod, callee_mod, weight=1)

        return g

    def build_from_file(self, solver_py_path: str) -> nx.DiGraph:
        solver_id = os.path.basename(os.path.dirname(solver_py_path))
        try:
            with open(solver_py_path) as fh:
                source = fh.read()
        except OSError:
            return self._empty_graph(solver_id)
        return self.build(source, solver_id)

    @staticmethod
    def _empty_graph(solver_id: str) -> nx.DiGraph:
        g = nx.DiGraph(solver_id=solver_id, layer="static")
        for m in ALL_MODULES:
            g.add_node(m, module=m)
        return g


# ---------------------------------------------------------------------------
# Dynamic layer (Layer T) builder
# ---------------------------------------------------------------------------

class DynamicGraphBuilder:
    """
    Aggregates RunRecords from a puzzle family into a weighted call graph.

    Nodes  = semantic module labels.
    Edges  = co-activation or sequential-call relationship in execution traces.
    Weight = summed call counts over all runs in the family.

    Node attribute ``activation`` = total calls to that module across all runs.
    Node attribute ``run_count``  = number of runs that activated that module.
    """

    def build(
        self,
        records: list[RunRecord],
        family: str = "unknown",
        solver_family: str = "unknown",
    ) -> nx.DiGraph:
        g = nx.DiGraph(family=family, solver_family=solver_family, layer="dynamic")
        for m in ALL_MODULES:
            g.add_node(m, module=m, activation=0, run_count=0)

        for rec in records:
            # Map each trace event to a semantic module
            active: dict[str, int] = defaultdict(int)  # module → total calls
            for ev in rec.events:
                mod = classify_function(ev.func_name)
                active[mod] += ev.call_count

            for mod, cnt in active.items():
                g.nodes[mod]["activation"] += cnt
                g.nodes[mod]["run_count"] += 1

            # Build directed edges from ordered event trace
            # (caller_module[i] → callee_module[i+1] in call stack)
            if len(rec.events) >= 2:
                for i in range(len(rec.events) - 1):
                    src_mod = classify_function(rec.events[i].func_name)
                    dst_mod = classify_function(rec.events[i + 1].func_name)
                    if src_mod == dst_mod:
                        continue
                    if g.has_edge(src_mod, dst_mod):
                        g[src_mod][dst_mod]["weight"] += 1
                    else:
                        g.add_edge(src_mod, dst_mod, weight=1)

        return g


# ---------------------------------------------------------------------------
# Multilayer cognition graph
# ---------------------------------------------------------------------------

class MultilayerCognitionGraph:
    """
    Container for the full multilayer graph for one solver family.

    Structure:
        .static_layer   : nx.DiGraph  — Layer S (structural)
        .family_layers  : dict[str, nx.DiGraph]  — Layer T per puzzle family
        .solver_family  : str
        .records        : list[RunRecord]  — all raw trace records

    All graphs share the same node set (ALL_MODULES).  Cross-layer coupling
    is implicit: the same node label in different layers refers to the same
    semantic module.
    """

    def __init__(self, solver_family: str):
        self.solver_family = solver_family
        self.static_layer: Optional[nx.DiGraph] = None
        self.family_layers: dict[str, nx.DiGraph] = {}
        self.records: list[RunRecord] = []
        self._dynamic_builder = DynamicGraphBuilder()

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    def set_static_layer(self, g: nx.DiGraph) -> None:
        self.static_layer = g

    def add_records(self, records: list[RunRecord]) -> None:
        """Add run records and rebuild all dynamic family layers."""
        self.records.extend(records)
        self._rebuild_dynamic_layers()

    def _rebuild_dynamic_layers(self) -> None:
        by_family: dict[str, list[RunRecord]] = defaultdict(list)
        for rec in self.records:
            by_family[rec.puzzle_family].append(rec)
        for fam, recs in by_family.items():
            self.family_layers[fam] = self._dynamic_builder.build(
                recs, family=fam, solver_family=self.solver_family
            )

    # ------------------------------------------------------------------
    # Convenience queries
    # ------------------------------------------------------------------

    def families(self) -> list[str]:
        return sorted(self.family_layers.keys())

    def node_activation_across_families(self, node: str) -> dict[str, float]:
        """Return {family: activation_count} for a semantic module node."""
        return {
            fam: self.family_layers[fam].nodes[node].get("activation", 0)
            for fam in self.family_layers
        }

    def combined_dynamic_layer(self) -> nx.DiGraph:
        """Merge all family layers into a single aggregated dynamic graph."""
        g = nx.DiGraph(family="all", solver_family=self.solver_family, layer="dynamic")
        for m in ALL_MODULES:
            g.add_node(m, module=m, activation=0, run_count=0)
        for fam_g in self.family_layers.values():
            for m in ALL_MODULES:
                g.nodes[m]["activation"] += fam_g.nodes[m].get("activation", 0)
                g.nodes[m]["run_count"] += fam_g.nodes[m].get("run_count", 0)
            for u, v, data in fam_g.edges(data=True):
                if g.has_edge(u, v):
                    g[u][v]["weight"] += data.get("weight", 1)
                else:
                    g.add_edge(u, v, weight=data.get("weight", 1))
        return g

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        return {
            "solver_family": self.solver_family,
            "n_records": len(self.records),
            "families": self.families(),
            "n_layers": len(self.family_layers) + (1 if self.static_layer else 0),
            "nodes": ALL_MODULES,
        }


# ---------------------------------------------------------------------------
# Cross-solver graph collection
# ---------------------------------------------------------------------------

class CognitionGraphCollection:
    """
    Holds MultilayerCognitionGraphs for multiple solver families,
    enabling cross-family comparisons.
    """

    def __init__(self):
        self.graphs: dict[str, MultilayerCognitionGraph] = {}

    def add(self, mlg: MultilayerCognitionGraph) -> None:
        self.graphs[mlg.solver_family] = mlg

    def solver_families(self) -> list[str]:
        return sorted(self.graphs.keys())

    def get(self, solver_family: str) -> Optional[MultilayerCognitionGraph]:
        return self.graphs.get(solver_family)

    @staticmethod
    def from_trace_log(log_path: str) -> "CognitionGraphCollection":
        """Build a collection from a JSONL trace log file."""
        log = TraceLog(log_path)
        by_solver: dict[str, list[RunRecord]] = defaultdict(list)
        for rec in log:
            by_solver[rec.solver_family].append(rec)

        coll = CognitionGraphCollection()
        for sf, recs in by_solver.items():
            mlg = MultilayerCognitionGraph(solver_family=sf)
            mlg.add_records(recs)
            coll.add(mlg)
        return coll
