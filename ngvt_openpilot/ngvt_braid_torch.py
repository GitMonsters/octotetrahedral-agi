"""
NGVT Braid Engine — PyTorch (Differentiable, Training-Time)
============================================================
Differentiable implementation of the NGVT torus projection and Compounding
Braid attention weighting.  Designed for:

  - Training NGVT-aware models end-to-end (gradients flow through both the
    torus mapping and the Braid boost).
  - Offline log analysis from Python without building the Rust crate.
  - Export to TorchScript for downstream use.

The hard `if dist < 1.5` threshold from the Rust version is replaced with a
differentiable Gaussian RBF kernel so gradients are defined everywhere:

    boost_weight(d) = exp(-d² / (2σ²))   ∈ (0, 1]
    adjusted_score  = raw_prob * (1 + (boost_factor - 1) * max_weight)
                      clamped to [0, 1]

This means at d=0 (exact match) the full boost_factor is applied, and the
boost decays smoothly with distance.  The σ (sigma) parameter controls the
zone radius — default matches the Rust engine's hard 1.5-unit threshold at
≈50% weight (sigma=1.28 → exp(-1.5²/(2*1.28²)) ≈ 0.5).

Usage:
    import torch
    from ngvt_braid_torch import NgvtBraidEngineTorch

    engine = NgvtBraidEngineTorch()

    # Single node
    coords, score = engine.process_node(x=200.0, y=150.0, raw_prob=0.5)

    # Batched (B leads at once)
    xy = torch.tensor([[200.0, 150.0], [400.0, 300.0]])   # (B, 2)
    probs = torch.tensor([0.5, 0.7])                       # (B,)
    coords, scores = engine.process_batch(xy, probs)       # (B,3), (B,)

    # As an nn.Module layer (differentiable):
    engine.register_failure_zones(zone_tensor)             # (N, 3)
    coords, scores = engine.forward(xy, probs)             # gradients flow through
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class NgvtBraidEngineTorch(nn.Module):
    """
    Differentiable NGVT torus projection + Compounding Braid attention.

    Args:
        major_radius:  R — distance from torus centre to tube centre
        minor_radius:  r — tube radius
        boost_factor:  maximum score multiplier near a failure zone
        sigma:         RBF kernel width (controls failure-zone influence radius)
        x_bounds:      (x_min, x_max) for input normalisation
        y_bounds:      (y_min, y_max) for input normalisation
    """

    def __init__(
        self,
        major_radius: float = 10.0,
        minor_radius: float = 3.0,
        boost_factor: float = 3.0,
        sigma: float = 1.28,
        x_bounds: Tuple[float, float] = (0.0, 1164.0),
        y_bounds: Tuple[float, float] = (0.0, 874.0),
    ) -> None:
        super().__init__()
        self.R = major_radius
        self.r = minor_radius
        self.boost_factor = boost_factor
        self.sigma = sigma
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds

        # Failure zone cache — not a learned parameter, updated per-frame
        self.register_buffer(
            "failure_zones",
            torch.empty(0, 3, dtype=torch.float32),
        )

    # ------------------------------------------------------------------
    # Core torus projection (differentiable)
    # ------------------------------------------------------------------

    def _project_to_torus(self, xy: torch.Tensor) -> torch.Tensor:
        """
        Map normalised image coordinates to the torus surface.

        Args:
            xy: (B, 2) tensor of (x, y) pixel coordinates
        Returns:
            (B, 3) tensor of (X, Y, Z) manifold coordinates
        """
        x, y = xy[:, 0], xy[:, 1]
        theta = ((x - self.x_bounds[0]) / (self.x_bounds[1] - self.x_bounds[0])) * 2.0 * math.pi
        phi   = ((y - self.y_bounds[0]) / (self.y_bounds[1] - self.y_bounds[0])) * 2.0 * math.pi

        m_x = (self.R + self.r * phi.cos()) * theta.cos()
        m_y = (self.R + self.r * phi.cos()) * theta.sin()
        m_z = self.r * phi.sin()
        return torch.stack([m_x, m_y, m_z], dim=1)   # (B, 3)

    # ------------------------------------------------------------------
    # Braid attention boost (differentiable Gaussian RBF)
    # ------------------------------------------------------------------

    def _braid_boost(self, coords: torch.Tensor, raw_probs: torch.Tensor) -> torch.Tensor:
        """
        Apply Compounding Braid amplification near registered failure zones.

        Args:
            coords:     (B, 3) manifold coordinates
            raw_probs:  (B,)   existence probabilities
        Returns:
            (B,) adjusted scores, clamped to [0, 1]
        """
        if self.failure_zones.shape[0] == 0:
            return raw_probs.clamp(0.0, 1.0)

        # (B, N, 3) pairwise distance to each failure zone
        zones = self.failure_zones.unsqueeze(0)        # (1, N, 3)
        c_exp = coords.unsqueeze(1)                    # (B, 1, 3)
        dists = (c_exp - zones).pow(2).sum(dim=2).sqrt()  # (B, N)

        # Gaussian RBF: weight ∈ (0,1], peak=1 at d=0
        rbf_weights = torch.exp(-dists.pow(2) / (2.0 * self.sigma ** 2))  # (B, N)
        max_weight  = rbf_weights.max(dim=1).values                        # (B,)

        # Interpolate between raw_prob (weight=0) and raw_prob*boost (weight=1)
        boost = 1.0 + (self.boost_factor - 1.0) * max_weight
        return (raw_probs * boost).clamp(0.0, 1.0)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def forward(
        self, xy: torch.Tensor, raw_probs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batched differentiable forward pass.

        Args:
            xy:         (B, 2) pixel coordinates
            raw_probs:  (B,)   existence probabilities from modelV2
        Returns:
            coords:  (B, 3) torus manifold coordinates
            scores:  (B,)   Braid-adjusted scores ∈ [0, 1]
        """
        coords = self._project_to_torus(xy)
        scores = self._braid_boost(coords, raw_probs)
        return coords, scores

    def process_node(
        self, x: float, y: float, raw_prob: float
    ) -> Tuple[list, float]:
        """Single-node convenience wrapper — matches the Rust API signature."""
        xy = torch.tensor([[x, y]], dtype=torch.float32)
        probs = torch.tensor([raw_prob], dtype=torch.float32)
        with torch.no_grad():
            coords, scores = self.forward(xy, probs)
        return coords[0].tolist(), float(scores[0])

    def process_batch(
        self, xy: torch.Tensor, raw_probs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Alias for forward(); present for API symmetry."""
        return self.forward(xy, raw_probs)

    def register_failure_zones(self, zones: torch.Tensor) -> None:
        """
        Update the failure-zone cache.  Called once per analysis frame.

        Args:
            zones: (N, 3) tensor of manifold coordinates, or empty tensor to clear.
        """
        if zones.ndim != 2 or zones.shape[1] != 3:
            raise ValueError(f"zones must be (N, 3), got {tuple(zones.shape)}")
        self.failure_zones = zones.to(dtype=torch.float32)

    def register_verification_results(self, unstable_nodes: list) -> None:
        """List-based wrapper — matches the Rust API signature."""
        if unstable_nodes:
            tensor = torch.tensor(unstable_nodes, dtype=torch.float32)
        else:
            tensor = torch.empty(0, 3, dtype=torch.float32)
        self.register_failure_zones(tensor)

    def get_active_failure_zones(self) -> list:
        """Returns current failure zones as a list — matches Rust API."""
        return self.failure_zones.tolist()

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    @torch.jit.export
    def scripted_forward(
        self, xy: torch.Tensor, raw_probs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """TorchScript-compatible forward pass (no Python-only types)."""
        return self.forward(xy, raw_probs)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    engine = NgvtBraidEngineTorch()

    # Torus projection
    xy = torch.tensor([[582.0, 437.0], [0.0, 0.0], [1164.0, 874.0]])
    probs = torch.tensor([0.8, 0.5, 0.3])
    coords, scores = engine(xy, probs)
    print("Torus coords:\n", coords)
    print("Scores (no zones):", scores)
    assert coords.shape == (3, 3)
    assert (scores >= 0).all() and (scores <= 1).all()

    # Braid boost
    engine.register_failure_zones(coords[:1])   # register first node as a zone
    _, boosted = engine(xy[:1], probs[:1])
    print(f"\nRaw prob: {probs[0].item():.3f}  →  Boosted score: {boosted[0].item():.3f}")
    assert boosted[0] > probs[0], "Braid boost did not increase score"

    # Gradient flow
    engine.register_failure_zones(coords.detach())
    xy_grad = xy.clone().requires_grad_(True)
    probs_grad = probs.clone().requires_grad_(True)
    _, scores_grad = engine(xy_grad, probs_grad)
    scores_grad.sum().backward()
    assert xy_grad.grad is not None, "No gradient flowed to xy"
    assert probs_grad.grad is not None, "No gradient flowed to probs"
    print("\n✓ Gradients flow through torus projection and Braid boost")

    # TorchScript export
    scripted = torch.jit.script(engine)
    scripted.save("/tmp/ngvt_braid.pt")
    print("✓ TorchScript export succeeded → /tmp/ngvt_braid.pt")

    print("\nAll checks passed.")
