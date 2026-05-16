"""
Tetrahedral Calculus Ops
Discrete exterior calculus primitives on a graph (directed edges, triangular faces).

All functions operate on pre-built edge/face data structures — no torch_geometric needed.
Functions are standalone and independently testable.
"""

import torch
from torch import Tensor


def gradient(node_feat: Tensor, edge_index: Tensor, edge_weights: Tensor) -> Tensor:
    """Directed discrete gradient: edge flow = weighted feature difference.

    Args:
        node_feat:    [N, D]  node feature vectors
        edge_index:   [2, E]  (src, dst) directed edges
        edge_weights: [E]     per-edge weights (typically 1/distance)

    Returns:
        edge_feat: [E, D]  edge_feat[e] = (f[dst_e] - f[src_e]) * w_e
    """
    src = edge_index[0]  # [E]
    dst = edge_index[1]  # [E]
    return (node_feat[dst] - node_feat[src]) * edge_weights.unsqueeze(-1)


def divergence(edge_feat: Tensor, edge_index: Tensor, n_nodes: int) -> Tensor:
    """Discrete divergence: net inflow minus outflow at each node.

    Args:
        edge_feat:  [E, D]  edge feature vectors (e.g. from gradient())
        edge_index: [2, E]  (src, dst) directed edges
        n_nodes:    int     number of nodes N

    Returns:
        div: [N, D]  div[i] = Σ_{e→i} edge_feat[e] − Σ_{i→e} edge_feat[e]
    """
    src = edge_index[0]  # [E]
    dst = edge_index[1]  # [E]
    D = edge_feat.shape[-1]
    E = edge_feat.shape[0]

    out = torch.zeros(n_nodes, D, device=edge_feat.device, dtype=edge_feat.dtype)
    # Incoming edges contribute positively
    out.scatter_add_(0, dst.unsqueeze(-1).expand(E, D), edge_feat)
    # Outgoing edges contribute negatively
    out.scatter_add_(0, src.unsqueeze(-1).expand(E, D), -edge_feat)
    return out


def laplacian(
    node_feat: Tensor,
    edge_index: Tensor,
    edge_weights: Tensor,
    n_nodes: int,
) -> Tensor:
    """Symmetric graph Laplacian: weighted sum of neighbor differences.

    Treats the directed edge set as an undirected graph — each stored edge
    i→j contributes w*(f[j]-f[i]) at i AND w*(f[i]-f[j]) at j.

    Args:
        node_feat:    [N, D]
        edge_index:   [2, E]  stored as directed (src, dst)
        edge_weights: [E]
        n_nodes:      int

    Returns:
        lap: [N, D]  Lf[i] = Σ_j w_ij * (f[j] - f[i])
    """
    src = edge_index[0]  # [E]
    dst = edge_index[1]  # [E]
    D = node_feat.shape[-1]
    E = edge_index.shape[1]

    diff = (node_feat[dst] - node_feat[src]) * edge_weights.unsqueeze(-1)  # [E, D]

    lap = torch.zeros(n_nodes, D, device=node_feat.device, dtype=node_feat.dtype)
    src_exp = src.unsqueeze(-1).expand(E, D)
    dst_exp = dst.unsqueeze(-1).expand(E, D)
    # At src node: w*(f[dst]-f[src]) = +diff
    lap.scatter_add_(0, src_exp, diff)
    # At dst node: w*(f[src]-f[dst]) = -diff
    lap.scatter_add_(0, dst_exp, -diff)
    return lap


def curl(
    edge_feat: Tensor,
    face_edge_idx: Tensor,
    face_edge_sign: Tensor,
) -> Tensor:
    """Discrete curl: circulation of edge flow around each triangular face.

    Args:
        edge_feat:      [E, D]  directed edge features
        face_edge_idx:  [3, F]  precomputed edge indices for each face's 3 edges
        face_edge_sign: [3, F]  ±1 sign (−1 when edge is stored in reverse direction)

    Returns:
        face_feat: [F, D]  curl[f] = Σ_{k=0..2} sign_k * edge_feat[idx_k]
    """
    D = edge_feat.shape[-1]
    F = face_edge_idx.shape[1]
    result = torch.zeros(F, D, device=edge_feat.device, dtype=edge_feat.dtype)

    # Unrolled for TorchScript-friendliness (fixed 3 half-edges per face)
    idx0 = face_edge_idx[0]  # [F]
    idx1 = face_edge_idx[1]
    idx2 = face_edge_idx[2]
    s0 = face_edge_sign[0].unsqueeze(-1)  # [F, 1]
    s1 = face_edge_sign[1].unsqueeze(-1)
    s2 = face_edge_sign[2].unsqueeze(-1)

    result = result + edge_feat[idx0] * s0
    result = result + edge_feat[idx1] * s1
    result = result + edge_feat[idx2] * s2
    return result


def face_to_node(face_feat: Tensor, face_index: Tensor, n_nodes: int) -> Tensor:
    """Project face features back to nodes by averaging over incident faces.

    Args:
        face_feat:  [F, D]
        face_index: [3, F]  node indices for each face corner
        n_nodes:    int

    Returns:
        node_feat: [N, D]  mean of all incident face features at each node
    """
    D = face_feat.shape[-1]
    F = face_feat.shape[0]

    out = torch.zeros(n_nodes, D, device=face_feat.device, dtype=face_feat.dtype)
    cnt = torch.zeros(n_nodes, 1, device=face_feat.device, dtype=face_feat.dtype)
    ones_f = torch.ones(F, 1, device=face_feat.device, dtype=face_feat.dtype)

    for k in range(3):
        nodes = face_index[k]  # [F]
        out.scatter_add_(0, nodes.unsqueeze(-1).expand(F, D), face_feat)
        cnt.scatter_add_(0, nodes.unsqueeze(-1), ones_f)

    return out / cnt.clamp(min=1.0)
