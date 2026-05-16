"""
Tetrahedral Grid Graph Vision Calculus Reasoning System

Overlays a tetrahedral simplicial complex on vision feature maps and applies
discrete exterior calculus (gradient, divergence, Laplacian, curl) on the
resulting graph.  This transforms flat CNN/ViT features into geometrically-
aware spatial reasoning features that capture edges, attention focal points,
rotational patterns, and diffused context through the lens of tetrahedral
geometry.
"""

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from core.tetrahedral_geometry import TetrahedralGeometry
from core.tetrahedral_calculus_ops import (
    curl,
    divergence,
    face_to_node,
    gradient,
    laplacian,
)


# ---------------------------------------------------------------------------
# TetrahedralGridGraph
# ---------------------------------------------------------------------------

class TetrahedralGridGraph(nn.Module):
    """Pre-computed tetrahedral simplicial complex over an H×W feature grid.

    Nodes  : H×W positions, indexed as i*W + j.
    Edges  : 4 directed types per cell — right, down, diag-DR, diag-DL.
    Faces  : 2 oriented triangles per interior cell.
    Tets   : Paired-face groups; tet_volumes stores per-face triangle areas
             (volume-weighted integration placeholder for 2-D embeddings).

    All tensors are registered as buffers — device-portable, not trainable.
    """

    edge_index:      Tensor  # [2, E]
    edge_weights:    Tensor  # [E]
    face_index:      Tensor  # [3, F]
    face_edge_idx:   Tensor  # [3, F]  precomputed edge index per face half-edge
    face_edge_sign:  Tensor  # [3, F]  ±1 sign corrections for direction
    positions:       Tensor  # [N, 2]  normalised (x, y) grid positions
    tet_volumes:     Tensor  # [F]     triangle areas (≡ "tet volumes" in 2-D)

    def __init__(self, grid_h: int = 8, grid_w: int = 8) -> None:
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        N = grid_h * grid_w

        # ------------------------------------------------------------------
        # Node positions  [N, 2]  (normalised so grid spacing ≈ 1)
        # ------------------------------------------------------------------
        pos_list: list[list[float]] = []
        for i in range(grid_h):
            for j in range(grid_w):
                x = j / max(grid_w - 1, 1) * (grid_w - 1)
                y = i / max(grid_h - 1, 1) * (grid_h - 1)
                pos_list.append([x, y])
        positions = torch.tensor(pos_list, dtype=torch.float32)  # [N, 2]

        # ------------------------------------------------------------------
        # Directed edges  (stored once per undirected pair, in canonical dir)
        # ------------------------------------------------------------------
        src_list: list[int] = []
        dst_list: list[int] = []

        for i in range(grid_h):
            for j in range(grid_w):
                n = i * grid_w + j
                # Right
                if j + 1 < grid_w:
                    src_list.append(n); dst_list.append(i * grid_w + j + 1)
                # Down
                if i + 1 < grid_h:
                    src_list.append(n); dst_list.append((i + 1) * grid_w + j)
                # Diagonal down-right
                if i + 1 < grid_h and j + 1 < grid_w:
                    src_list.append(n); dst_list.append((i + 1) * grid_w + j + 1)
                # Diagonal down-left
                if i + 1 < grid_h and j - 1 >= 0:
                    src_list.append(n); dst_list.append((i + 1) * grid_w + j - 1)

        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)  # [2, E]
        E = edge_index.shape[1]

        # Edge weights = 1 / Euclidean distance
        src_pos = positions[edge_index[0]]  # [E, 2]
        dst_pos = positions[edge_index[1]]  # [E, 2]
        dists = (dst_pos - src_pos).norm(dim=-1).clamp(min=1e-6)  # [E]
        edge_weights = 1.0 / dists  # [E]

        # ------------------------------------------------------------------
        # Triangular faces  (2 per interior cell)
        # face_index: [3, F] — oriented as (a, b, c) counter-clockwise
        # ------------------------------------------------------------------
        fa_list: list[int] = []
        fb_list: list[int] = []
        fc_list: list[int] = []

        for i in range(grid_h - 1):
            for j in range(grid_w - 1):
                n00 = i * grid_w + j
                n10 = (i + 1) * grid_w + j
                n01 = i * grid_w + j + 1
                n11 = (i + 1) * grid_w + j + 1
                # Lower-left triangle: (n00, n10, n01)
                fa_list.append(n00); fb_list.append(n10); fc_list.append(n01)
                # Upper-right triangle: (n11, n10, n01)
                fa_list.append(n11); fb_list.append(n10); fc_list.append(n01)

        face_index = torch.tensor([fa_list, fb_list, fc_list], dtype=torch.long)  # [3, F]
        F_count = face_index.shape[1]

        # ------------------------------------------------------------------
        # Triangle areas  (tet_volumes proxy in 2-D)
        # ------------------------------------------------------------------
        pa = positions[face_index[0]]  # [F, 2]
        pb = positions[face_index[1]]
        pc = positions[face_index[2]]
        ab = pb - pa  # [F, 2]
        ac = pc - pa  # [F, 2]
        # |cross product z-component| / 2
        tet_volumes = (ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]).abs() * 0.5  # [F]

        # ------------------------------------------------------------------
        # Precompute face → edge lookup  (face_edge_idx, face_edge_sign)
        # ------------------------------------------------------------------
        # Build a flat edge map: edge_map[src * N + dst] = edge_idx
        edge_map = torch.full((N * N,), -1, dtype=torch.long)
        flat_idx = edge_index[0] * N + edge_index[1]
        edge_map[flat_idx] = torch.arange(E, dtype=torch.long)

        fei_rows: list[Tensor] = []
        fes_rows: list[Tensor] = []

        # For each face corner k→(k+1 mod 3), resolve the canonical edge
        corners = [0, 1, 2]
        next_corners = [1, 2, 0]
        for k, nk in zip(corners, next_corners):
            a_nodes = face_index[k]           # [F]
            b_nodes = face_index[nk]          # [F]

            ab_flat = a_nodes * N + b_nodes   # [F]
            ba_flat = b_nodes * N + a_nodes   # [F]

            e_ab = edge_map[ab_flat]          # [F]  ≥0 if a→b stored
            e_ba = edge_map[ba_flat]          # [F]  ≥0 if b→a stored

            has_ab = e_ab >= 0
            has_ba = e_ba >= 0

            # Index: prefer a→b; fall back to b→a; clamp to 0 if neither
            e_idx = torch.where(has_ab, e_ab,
                    torch.where(has_ba, e_ba,
                                torch.zeros(F_count, dtype=torch.long)))

            # Sign: +1 if forward (a→b), −1 if reversed, 0 if missing
            e_sign = torch.where(has_ab, torch.ones(F_count),
                     torch.where(has_ba, -torch.ones(F_count),
                                 torch.zeros(F_count)))

            fei_rows.append(e_idx)
            fes_rows.append(e_sign)

        face_edge_idx  = torch.stack(fei_rows, dim=0)  # [3, F]
        face_edge_sign = torch.stack(fes_rows, dim=0)  # [3, F]

        # ------------------------------------------------------------------
        # Register everything as buffers
        # ------------------------------------------------------------------
        self.register_buffer("positions",       positions)
        self.register_buffer("edge_index",      edge_index)
        self.register_buffer("edge_weights",    edge_weights)
        self.register_buffer("face_index",      face_index)
        self.register_buffer("face_edge_idx",   face_edge_idx)
        self.register_buffer("face_edge_sign",  face_edge_sign)
        self.register_buffer("tet_volumes",     tet_volumes)

    def forward(self) -> None:  # type: ignore[override]
        raise RuntimeError("TetrahedralGridGraph is a data container; call methods directly.")


# ---------------------------------------------------------------------------
# TetrahedralVisionCalculus
# ---------------------------------------------------------------------------

class TetrahedralVisionCalculus(nn.Module):
    """Apply discrete exterior calculus to vision tokens via a tetrahedral grid.

    Takes [B, T, D] vision tokens from VisionEncoder, reshapes them onto an
    H×W grid, computes gradient / divergence / Laplacian / curl, mixes all
    five feature streams (f, |grad|, div, lap, curl→node), then blends back
    into the token stream via a learnable residual gate.

    Args:
        hidden_dim: token feature dimensionality D (default 256)
        grid_h:     grid height  (default 8)
        grid_w:     grid width   (default 8)
        dropout:    dropout rate on output projection (default 0.1)
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        grid_h: int = 8,
        grid_w: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.grid_h = grid_h
        self.grid_w = grid_w
        N = grid_h * grid_w

        # Triangulated grid (all geometry as frozen buffers)
        self.graph = TetrahedralGridGraph(grid_h, grid_w)

        # ------------------------------------------------------------------
        # Tetrahedral geometry prior (64-point Sloane packing)
        # Frozen — used only as a geometric embedding prior via geo_bridge.
        # ------------------------------------------------------------------
        self.tet_geometry = TetrahedralGeometry()
        for p in self.tet_geometry.parameters():
            p.requires_grad_(False)
        # Projects 3-D tet point coordinates → hidden_dim feature space
        self.geo_bridge = nn.Linear(3, hidden_dim, bias=False)

        # ------------------------------------------------------------------
        # Feature projections
        # ------------------------------------------------------------------
        self.project_in = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Blend [f, |grad|, div, lap, curl_n] → D
        self.calc_mixer = nn.Linear(5 * hidden_dim, hidden_dim)

        self.project_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

        # Learnable residual gate α;  out = x + tanh(α) * calc_out
        self.alpha = nn.Parameter(torch.zeros(1))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _geo_prior(self, N: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        """Map tetrahedral point coordinates to a [N, D] geometric prior."""
        geo_pts = self.tet_geometry.points  # [64, 3]
        n_geo = geo_pts.shape[0]
        if N <= n_geo:
            pts_used = geo_pts[:N]
        else:
            reps = math.ceil(N / n_geo)
            pts_used = geo_pts.repeat(reps, 1)[:N]
        # geo_bridge is trained; no_grad only on tet_geometry params
        geo_feat = self.geo_bridge(pts_used.to(dtype=dtype))  # [N, D]
        return geo_feat

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: Tensor,
        seq_len: Optional[int] = None,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """Apply tetrahedral discrete exterior calculus to vision tokens.

        Args:
            x:       [B, T, D]  vision-enriched token sequence
            seq_len: ignored (kept for API symmetry)

        Returns:
            out:      [B, T, D]  calculus-enriched tokens
            info:     dict of diagnostic scalars and graph statistics
        """
        B, T, D = x.shape
        H, W = self.grid_h, self.grid_w
        N = H * W

        # ------------------------------------------------------------------
        # Pad / truncate T → N
        # ------------------------------------------------------------------
        if T >= N:
            x_grid = x[:, :N, :]
            pad_size = 0
        else:
            pad_size = N - T
            x_grid = F.pad(x, (0, 0, 0, pad_size))  # [B, N, D]

        # ------------------------------------------------------------------
        # Project to node features + add geometric prior
        # ------------------------------------------------------------------
        node_feat = self.project_in(x_grid)  # [B, N, D]
        geo_feat = self._geo_prior(N, x.device, x.dtype)  # [N, D]
        node_feat = node_feat + geo_feat.unsqueeze(0)

        # ------------------------------------------------------------------
        # Retrieve graph topology
        # ------------------------------------------------------------------
        edge_index    = self.graph.edge_index     # [2, E]
        edge_weights  = self.graph.edge_weights   # [E]
        face_index    = self.graph.face_index     # [3, F]
        face_edge_idx = self.graph.face_edge_idx  # [3, F]
        face_edge_sign= self.graph.face_edge_sign # [3, F]

        E      = edge_index.shape[1]
        F_count = face_index.shape[1]

        src = edge_index[0]  # [E]
        dst = edge_index[1]  # [E]

        # Expand indices for batched scatter (dim=1 on [B, N, D])
        src_exp = src.unsqueeze(0).unsqueeze(-1).expand(B, E, D)  # [B, E, D]
        dst_exp = dst.unsqueeze(0).unsqueeze(-1).expand(B, E, D)

        # ------------------------------------------------------------------
        # Gradient  [B, E, D]
        # ------------------------------------------------------------------
        # grad_e[b, e, :] = (f[dst] - f[src]) * w_e
        grad_e = (node_feat[:, dst, :] - node_feat[:, src, :]) * \
                  edge_weights.view(1, E, 1)  # [B, E, D]

        # ------------------------------------------------------------------
        # Gradient magnitude aggregated at nodes  [B, N, D]
        # ------------------------------------------------------------------
        grad_mag_e = grad_e.abs()  # [B, E, D]

        grad_mag = torch.zeros(B, N, D, device=x.device, dtype=x.dtype)
        grad_cnt = torch.zeros(B, N, 1, device=x.device, dtype=x.dtype)
        ones_e   = torch.ones(B, E, 1, device=x.device, dtype=x.dtype)

        grad_mag.scatter_add_(1, src_exp, grad_mag_e)
        grad_cnt.scatter_add_(1, src.unsqueeze(0).unsqueeze(-1).expand(B, E, 1), ones_e)
        grad_mag = grad_mag / grad_cnt.clamp(min=1.0)

        # ------------------------------------------------------------------
        # Divergence  [B, N, D]
        # ------------------------------------------------------------------
        div = torch.zeros(B, N, D, device=x.device, dtype=x.dtype)
        div.scatter_add_(1, dst_exp,  grad_e)   # incoming
        div.scatter_add_(1, src_exp, -grad_e)   # outgoing

        # ------------------------------------------------------------------
        # Laplacian  [B, N, D]   (symmetric — treat directed set as undirected)
        # ------------------------------------------------------------------
        lap = torch.zeros(B, N, D, device=x.device, dtype=x.dtype)
        lap.scatter_add_(1, src_exp,  grad_e)   # w*(f[j]-f[i]) at i
        lap.scatter_add_(1, dst_exp, -grad_e)   # w*(f[i]-f[j]) at j

        # ------------------------------------------------------------------
        # Curl on faces  [B, F, D]
        # ------------------------------------------------------------------
        # face_edge_sign is stored as float so multiply directly
        fe_sign = face_edge_sign  # [3, F]

        idx0 = face_edge_idx[0]   # [F]
        idx1 = face_edge_idx[1]
        idx2 = face_edge_idx[2]
        s0 = fe_sign[0].view(1, F_count, 1)   # [1, F, 1]
        s1 = fe_sign[1].view(1, F_count, 1)
        s2 = fe_sign[2].view(1, F_count, 1)

        curl_f = (grad_e[:, idx0, :] * s0 +
                  grad_e[:, idx1, :] * s1 +
                  grad_e[:, idx2, :] * s2)  # [B, F, D]

        # ------------------------------------------------------------------
        # Face → node projection  [B, N, D]
        # ------------------------------------------------------------------
        curl_n = torch.zeros(B, N, D, device=x.device, dtype=x.dtype)
        curl_cnt = torch.zeros(B, N, 1, device=x.device, dtype=x.dtype)
        ones_f = torch.ones(B, F_count, 1, device=x.device, dtype=x.dtype)

        for k in range(3):
            fn = face_index[k]  # [F]
            fn_exp = fn.unsqueeze(0).unsqueeze(-1).expand(B, F_count, D)
            curl_n.scatter_add_(1, fn_exp, curl_f)
            curl_cnt.scatter_add_(
                1,
                fn.unsqueeze(0).unsqueeze(-1).expand(B, F_count, 1),
                ones_f,
            )
        curl_n = curl_n / curl_cnt.clamp(min=1.0)

        # ------------------------------------------------------------------
        # Mix all five feature streams
        # ------------------------------------------------------------------
        cat_feat = torch.cat([node_feat, grad_mag, div, lap, curl_n], dim=-1)  # [B, N, 5D]
        mixed = self.calc_mixer(cat_feat)  # [B, N, D]

        # ------------------------------------------------------------------
        # Reshape back to [B, T, D]
        # ------------------------------------------------------------------
        if T >= N:
            # Tokens beyond the grid are not calculus-enriched; reuse original
            mixed_out = torch.cat([mixed, x[:, N:, :]], dim=1)
        else:
            mixed_out = mixed[:, :T, :]  # drop padding

        # ------------------------------------------------------------------
        # Output projection and gated residual blend
        # ------------------------------------------------------------------
        calc_out = self.project_out(mixed_out)                        # [B, T, D]
        out = x + torch.tanh(self.alpha) * calc_out                   # [B, T, D]

        # ------------------------------------------------------------------
        # Diagnostic info
        # ------------------------------------------------------------------
        info: Dict[str, Any] = {
            "grad_magnitude":  float(grad_e.abs().mean().item()),
            "divergence_norm": float(div.abs().mean().item()),
            "laplacian_norm":  float(lap.abs().mean().item()),
            "curl_norm":       float(curl_f.abs().mean().item()),
            "grid_shape":      (H, W),
            "n_edges":         E,
            "n_faces":         F_count,
        }
        return out, info
