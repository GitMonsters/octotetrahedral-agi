"""
NGVT Braid Engine — tinygrad (Inference-Time, openpilot-Native)
===============================================================
tinygrad implementation of the NGVT torus projection and Compounding Braid
attention weighting.  Designed for:

  - openpilot inference: modeld already uses tinygrad as its runtime, so
    this version integrates naturally with the existing build pipeline.
  - GPU/Metal/CUDA acceleration via tinygrad's backend selection.
  - Offline log analysis without the Rust build step.

This version is functionally identical to the PyTorch version but uses
tinygrad's Tensor API.  The same Gaussian RBF Braid boost is used so that
if weights are trained in PyTorch and exported to a .npy checkpoint, they
can be loaded here directly.

Backend selection (set METAL=1, CUDA=1, etc. before import):
    METAL=1 python ngvt_braid_tinygrad.py     # Apple GPU
    CUDA=1  python ngvt_braid_tinygrad.py     # NVIDIA GPU
    # default = CPU

Usage:
    from ngvt_braid_tinygrad import NgvtBraidEngineTinygrad
    import numpy as np

    engine = NgvtBraidEngineTinygrad()

    # Single node (matches Rust API)
    coords, score = engine.process_node(x=200.0, y=150.0, raw_prob=0.5)

    # Batched (matches PyTorch API)
    from tinygrad import Tensor
    xy    = Tensor([[200.0, 150.0], [400.0, 300.0]])
    probs = Tensor([0.5, 0.7])
    coords_t, scores_t = engine.forward(xy, probs)

    # Update failure-zone cache
    engine.register_verification_results([[c1, c2, c3], ...])

Weight compatibility with PyTorch version
-----------------------------------------
The PyTorch and tinygrad versions share no learnable weights in the base
engine (all parameters are scalars).  If you subclass either to add learned
components (e.g., a trainable σ or a learned projection head), export the
weights from PyTorch with:

    np.save("ngvt_sigma.npy", engine.sigma_param.detach().numpy())

and load in tinygrad with:

    import numpy as np
    from tinygrad import Tensor
    sigma = float(Tensor(np.load("ngvt_sigma.npy")).numpy())
"""

import math
from typing import List, Optional, Tuple

import numpy as np

try:
    from tinygrad import Tensor
    from tinygrad import dtypes
except ImportError as exc:
    raise ImportError(
        "tinygrad is required for NgvtBraidEngineTinygrad.\n"
        "Install with:  pip install tinygrad\n"
        "openpilot ships tinygrad in third_party/tinygrad."
    ) from exc


class NgvtBraidEngineTinygrad:
    """
    tinygrad NGVT torus projection + Compounding Braid attention.

    Args match the PyTorch and Rust versions for drop-in compatibility.
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
        self.R = major_radius
        self.r = minor_radius
        self.boost_factor = boost_factor
        self.sigma = sigma
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        # (N, 3) failure-zone cache; empty until first registration
        self._failure_zones: Optional[Tensor] = None

    # ------------------------------------------------------------------
    # Core torus projection
    # ------------------------------------------------------------------

    def _project_to_torus(self, xy: Tensor) -> Tensor:
        """
        Map image coordinates to the torus surface.

        Args:
            xy: (B, 2) Tensor of (x, y) pixel coordinates
        Returns:
            (B, 3) Tensor of (X, Y, Z) manifold coordinates
        """
        x = xy[:, 0]
        y = xy[:, 1]

        x_range = self.x_bounds[1] - self.x_bounds[0]
        y_range = self.y_bounds[1] - self.y_bounds[0]

        theta = ((x - self.x_bounds[0]) / x_range) * (2.0 * math.pi)
        phi   = ((y - self.y_bounds[0]) / y_range) * (2.0 * math.pi)

        m_x = (self.R + self.r * phi.cos()) * theta.cos()
        m_y = (self.R + self.r * phi.cos()) * theta.sin()
        m_z = self.r * phi.sin()

        # Stack to (B, 3)
        return m_x.unsqueeze(1).cat(m_y.unsqueeze(1), m_z.unsqueeze(1), dim=1)

    # ------------------------------------------------------------------
    # Braid attention boost (differentiable Gaussian RBF)
    # ------------------------------------------------------------------

    def _braid_boost(self, coords: Tensor, raw_probs: Tensor) -> Tensor:
        """
        Apply Compounding Braid amplification near registered failure zones.

        Args:
            coords:     (B, 3) manifold coordinates
            raw_probs:  (B,)   existence probabilities
        Returns:
            (B,) adjusted scores, clamped to [0, 1]
        """
        if self._failure_zones is None or self._failure_zones.shape[0] == 0:
            return raw_probs.clip(0.0, 1.0)

        # Pairwise distances: (B, N)
        # coords: (B, 1, 3)  zones: (1, N, 3)
        c_exp = coords.unsqueeze(1)                       # (B, 1, 3)
        z_exp = self._failure_zones.unsqueeze(0)          # (1, N, 3)
        dists = ((c_exp - z_exp) ** 2).sum(axis=2).sqrt() # (B, N)

        # Gaussian RBF weights
        rbf = (-(dists ** 2) / (2.0 * self.sigma ** 2)).exp()  # (B, N)
        max_weight = rbf.max(axis=1)                             # (B,)

        boost = 1.0 + (self.boost_factor - 1.0) * max_weight
        return (raw_probs * boost).clip(0.0, 1.0)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def forward(self, xy: Tensor, raw_probs: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Batched forward pass.

        Args:
            xy:         (B, 2) Tensor of pixel coordinates
            raw_probs:  (B,)   Tensor of existence probabilities
        Returns:
            coords: (B, 3), scores: (B,)
        """
        coords = self._project_to_torus(xy)
        scores = self._braid_boost(coords, raw_probs)
        return coords, scores

    def process_node(
        self, x: float, y: float, raw_prob: float
    ) -> Tuple[List[float], float]:
        """Single-node wrapper — matches the Rust / PyTorch API."""
        xy    = Tensor([[x, y]])
        probs = Tensor([raw_prob])
        coords, scores = self.forward(xy, probs)
        return coords[0].numpy().tolist(), float(scores[0].numpy())

    def process_batch(self, xy: Tensor, raw_probs: Tensor) -> Tuple[Tensor, Tensor]:
        """Alias for forward() — present for API symmetry."""
        return self.forward(xy, raw_probs)

    def register_failure_zones(self, zones: Tensor) -> None:
        """Update failure-zone cache from a (N, 3) Tensor."""
        if zones.shape[0] == 0:
            self._failure_zones = None
        else:
            self._failure_zones = zones.cast(dtypes.float32)

    def register_verification_results(self, unstable_nodes: List[List[float]]) -> None:
        """List-based wrapper — matches the Rust / PyTorch API."""
        if unstable_nodes:
            arr = np.array(unstable_nodes, dtype=np.float32)
            self._failure_zones = Tensor(arr)
        else:
            self._failure_zones = None

    def get_active_failure_zones(self) -> List[List[float]]:
        """Returns failure zones as a list — matches Rust / PyTorch API."""
        if self._failure_zones is None:
            return []
        return self._failure_zones.numpy().tolist()

    # ------------------------------------------------------------------
    # openpilot integration note
    # ------------------------------------------------------------------
    # openpilot's modeld uses tinygrad for inference.  To incorporate the
    # NGVT Braid projection as part of a custom model:
    #
    #   1. Subclass this and add a tinygrad Linear / Conv layer.
    #   2. Save weights to .safetensors or .npy after training in PyTorch.
    #   3. Load weights and run inference here.
    #   4. For offline log analysis, feed LogReader modelV2 output through
    #      tools/ngvt_analysis.py (which already uses the Rust or Python
    #      backend transparently).


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    engine = NgvtBraidEngineTinygrad()

    # Torus projection
    xy    = Tensor([[582.0, 437.0], [0.0, 0.0], [1164.0, 874.0]])
    probs = Tensor([0.8, 0.5, 0.3])
    coords, scores = engine.forward(xy, probs)
    coords_np = coords.numpy()
    scores_np = scores.numpy()

    print("Torus coords:\n", coords_np)
    print("Scores (no zones):", scores_np)
    assert coords_np.shape == (3, 3), f"Bad shape: {coords_np.shape}"
    assert np.isfinite(coords_np).all(), "Non-finite torus coords"
    assert (scores_np >= 0).all() and (scores_np <= 1).all(), "Scores out of [0,1]"

    # Braid boost
    zone_tensor = Tensor(coords_np[:1])
    engine.register_failure_zones(zone_tensor)
    _, boosted = engine.forward(xy[:1], probs[:1])
    boosted_val = float(boosted[0].numpy())
    raw_val = float(probs[:1].numpy()[0])
    print(f"\nRaw prob: {raw_val:.3f}  →  Boosted score: {boosted_val:.3f}")
    assert boosted_val > raw_val, "Braid boost did not increase score"

    # Cache lifecycle
    engine.register_verification_results([])
    assert engine.get_active_failure_zones() == [], "Cache not cleared"

    # process_node single-node API
    c, s = engine.process_node(200.0, 150.0, 0.6)
    assert len(c) == 3 and isinstance(s, float)
    print(f"\nprocess_node: coords={[round(v,3) for v in c]}  score={s:.3f}")

    # Verify parity with PyTorch version (if available)
    try:
        from ngvt_braid_torch import NgvtBraidEngineTorch
        import torch
        torch_engine = NgvtBraidEngineTorch()
        torch_engine.eval()
        t_coords, t_scores = torch_engine.process_node(200.0, 150.0, 0.6)
        coord_diff = max(abs(a - b) for a, b in zip(c, t_coords))
        score_diff = abs(s - t_scores)
        print(f"\nParity vs PyTorch — max coord diff: {coord_diff:.2e}  score diff: {score_diff:.2e}")
        assert coord_diff < 1e-4, f"Coord parity failure: {coord_diff}"
        assert score_diff < 1e-4, f"Score parity failure: {score_diff}"
        print("✓ tinygrad and PyTorch versions are numerically equivalent")
    except ImportError:
        print("\n(PyTorch not available; skipping parity check)")

    print("\nAll checks passed.")
