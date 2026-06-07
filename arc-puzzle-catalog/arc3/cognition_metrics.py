"""
Cognition Metrics — Compute multilayer graph metrics for ARC/RE-ARC solvers.

Key metrics:
  participation_coefficient   — How evenly a module is used across puzzle families
  modularity                  — Community structure quality
  cross_layer_stability       — How similar community assignments are across families
  centrality                  — Betweenness and PageRank per module
  transcendplexity_score      — Composite index of reuse + stability + centrality

Reference notation follows the design doc (Sec. 5).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import networkx as nx
import networkx.algorithms.community as nx_comm

from arc3.cognition_graph import ALL_MODULES, MultilayerCognitionGraph


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class NodeMetrics:
    """Per-module metrics for one solver family."""
    module: str
    participation: float = 0.0         # 0 (single family) → 1 (uniform across all)
    activation_total: float = 0.0      # total calls across all families
    activation_entropy: float = 0.0    # Shannon entropy of activation distribution
    betweenness: float = 0.0           # betweenness centrality on combined dynamic layer
    pagerank: float = 0.0              # PageRank on combined dynamic layer
    community: str = ""                # community label from combined layer
    community_stability: float = 0.0   # 1 − (fraction of families with different community)


@dataclass
class MetricsResult:
    """All metrics for one solver family."""
    solver_family: str
    node_metrics: dict[str, NodeMetrics] = field(default_factory=dict)

    # Graph-level metrics
    modularity: float = 0.0
    avg_participation: float = 0.0
    cross_layer_stability: float = 0.0
    transcendplexity: float = 0.0

    # Success stats
    n_records: int = 0
    success_rate: float = 0.0
    n_families: int = 0

    def to_dict(self) -> dict[str, Any]:
        d = {
            "solver_family": self.solver_family,
            "modularity": self.modularity,
            "avg_participation": self.avg_participation,
            "cross_layer_stability": self.cross_layer_stability,
            "transcendplexity": self.transcendplexity,
            "n_records": self.n_records,
            "success_rate": self.success_rate,
            "n_families": self.n_families,
            "nodes": {
                m: {
                    "participation": nm.participation,
                    "activation_total": nm.activation_total,
                    "activation_entropy": nm.activation_entropy,
                    "betweenness": nm.betweenness,
                    "pagerank": nm.pagerank,
                    "community": nm.community,
                    "community_stability": nm.community_stability,
                }
                for m, nm in self.node_metrics.items()
            },
        }
        return d


# ---------------------------------------------------------------------------
# Core metric computations
# ---------------------------------------------------------------------------

def participation_coefficient(activation_per_family: dict[str, float]) -> float:
    """
    Compute the participation coefficient for a node across task families.

    P_i = 1 - sum_s (k_{is} / k_i)^2

    where k_{is} is the node's activation in family s and k_i is its total
    activation.  P=0 means all activity in one family; P→1 means perfectly
    uniform across families.

    Returns 0.0 if the node has zero total activation.
    """
    values = list(activation_per_family.values())
    total = sum(values)
    if total == 0:
        return 0.0
    n = len(values)
    if n <= 1:
        return 0.0
    sq_sum = sum((v / total) ** 2 for v in values)
    # Normalise so max value is 1 (multiply by n/(n-1))
    return (1.0 - sq_sum) * (n / (n - 1))


def activation_entropy(activation_per_family: dict[str, float]) -> float:
    """Shannon entropy of activation distribution (nats, normalised by log n)."""
    values = [v for v in activation_per_family.values() if v > 0]
    total = sum(values)
    if total == 0 or len(values) <= 1:
        return 0.0
    probs = [v / total for v in values]
    h = -sum(p * math.log(p) for p in probs)
    h_max = math.log(len(values))
    return h / h_max if h_max > 0 else 0.0


def compute_modularity_and_communities(
    g: nx.DiGraph,
) -> tuple[float, dict[str, str]]:
    """
    Detect communities and return (modularity_score, {node: community_label}).

    Uses greedy modularity maximisation on the undirected version of the graph.
    Falls back to each node in its own community if the graph has no edges.
    """
    ug = g.to_undirected()
    # Remove isolated nodes for community detection
    ug_connected = ug.copy()
    ug_connected.remove_nodes_from(list(nx.isolates(ug)))

    if ug_connected.number_of_edges() == 0:
        # No edges — each node is its own community
        communities = [{n} for n in g.nodes()]
        node_to_comm = {n: str(i) for i, c in enumerate(communities) for n in c}
        return 0.0, node_to_comm

    try:
        communities = list(nx_comm.greedy_modularity_communities(ug_connected))
        q = nx_comm.modularity(ug_connected, communities)
    except Exception:
        communities = [{n} for n in ug_connected.nodes()]
        q = 0.0

    node_to_comm: dict[str, str] = {}
    for i, comm in enumerate(communities):
        label = f"C{i}"
        for n in comm:
            node_to_comm[n] = label
    # Isolated nodes get their own community
    for n in nx.isolates(ug):
        node_to_comm[n] = f"C_iso_{n}"

    return float(q), node_to_comm


def cross_layer_stability(
    family_communities: dict[str, dict[str, str]],
) -> tuple[float, dict[str, float]]:
    """
    Measure how stable each node's community assignment is across families.

    Returns:
        (global_stability, {node: node_stability})

    node_stability = fraction of family pairs where the node is in the
    same relative community (same community label).  1.0 = always same.
    """
    families = list(family_communities.keys())
    if len(families) <= 1:
        node_stab = {n: 1.0 for n in ALL_MODULES}
        return 1.0, node_stab

    node_stab: dict[str, float] = {}
    for node in ALL_MODULES:
        assignments = [
            family_communities[fam].get(node, "?")
            for fam in families
        ]
        # Stability = fraction of pairs that agree
        pairs = 0
        agreements = 0
        for i in range(len(assignments)):
            for j in range(i + 1, len(assignments)):
                pairs += 1
                if assignments[i] == assignments[j]:
                    agreements += 1
        node_stab[node] = agreements / pairs if pairs > 0 else 1.0

    global_stab = sum(node_stab.values()) / len(node_stab) if node_stab else 0.0
    return global_stab, node_stab


def transcendplexity_score(
    avg_participation: float,
    cross_layer_stab: float,
    modularity_q: float,
    success_rate: float,
) -> float:
    """
    Composite transcendplexity index.

    T = 0.40 * participation + 0.30 * cross_layer_stability
          + 0.20 * modularity + 0.10 * success_rate

    All inputs should be in [0, 1].  Higher = more "transcendent" reuse.
    """
    return (
        0.40 * avg_participation
        + 0.30 * cross_layer_stab
        + 0.20 * min(max(modularity_q, 0.0), 1.0)
        + 0.10 * success_rate
    )


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------

def compute_metrics(mlg: MultilayerCognitionGraph) -> MetricsResult:
    """
    Compute all metrics for a single MultilayerCognitionGraph.

    Returns a fully populated MetricsResult.
    """
    result = MetricsResult(solver_family=mlg.solver_family)
    result.n_records = len(mlg.records)
    result.n_families = len(mlg.families())
    if result.n_records > 0:
        result.success_rate = sum(1 for r in mlg.records if r.success) / result.n_records

    # ---- Combined dynamic layer ----------------------------------------
    combined = mlg.combined_dynamic_layer()

    # Centrality on combined graph
    try:
        betweenness = nx.betweenness_centrality(combined, weight="weight", normalized=True)
    except Exception:
        betweenness = {m: 0.0 for m in ALL_MODULES}

    try:
        pagerank = nx.pagerank(combined, weight="weight")
    except Exception:
        pagerank = {m: 1.0 / len(ALL_MODULES) for m in ALL_MODULES}

    # Community detection on combined layer
    result.modularity, combined_communities = compute_modularity_and_communities(combined)

    # Per-family community assignments (for cross-layer stability)
    family_communities: dict[str, dict[str, str]] = {}
    for fam, fam_g in mlg.family_layers.items():
        _, fam_comms = compute_modularity_and_communities(fam_g)
        family_communities[fam] = fam_comms

    result.cross_layer_stability, node_stability = cross_layer_stability(family_communities)

    # ---- Per-node metrics -----------------------------------------------
    participations: list[float] = []
    for mod in ALL_MODULES:
        activation_by_fam = mlg.node_activation_across_families(mod)
        p = participation_coefficient(activation_by_fam)
        h = activation_entropy(activation_by_fam)
        total_act = sum(activation_by_fam.values())

        nm = NodeMetrics(
            module=mod,
            participation=p,
            activation_total=total_act,
            activation_entropy=h,
            betweenness=betweenness.get(mod, 0.0),
            pagerank=pagerank.get(mod, 0.0),
            community=combined_communities.get(mod, "?"),
            community_stability=node_stability.get(mod, 1.0),
        )
        result.node_metrics[mod] = nm
        if total_act > 0:
            participations.append(p)

    result.avg_participation = (
        sum(participations) / len(participations) if participations else 0.0
    )
    result.transcendplexity = transcendplexity_score(
        result.avg_participation,
        result.cross_layer_stability,
        result.modularity,
        result.success_rate,
    )

    return result


def compute_all(mlgs: dict[str, MultilayerCognitionGraph]) -> dict[str, MetricsResult]:
    """Compute metrics for a dict of {solver_family: MLG}."""
    return {sf: compute_metrics(mlg) for sf, mlg in mlgs.items()}
