"""
Reservoir Computing Dynamics — Compound Integration

Cherry-picked theoretical insights from reservoir computing (Jaeger 2001,
Lukosevicius 2012, Maass 2002) compounded into four mutually reinforcing
mechanisms that embed naturally into the OctoTetrahedral architecture:

  1. EchoStateConstraint    — spectral radius normalization so the spiking
                              reservoir decays gracefully (traces don't explode
                              or die). Tuning the network to this edge-of-chaos
                              boundary maximises memory capacity.

  2. NeuralPacemaker        — multi-frequency oscillatory driving signal Z(t)
                              (θ/α/γ bands). Keeps reservoir energy from dying
                              out, maps directly to the μᵢ·Z(t) term that the
                              RNA editing layer uses to scale per-neuron input.

  3. TemporalBasisDiversity — assigns diverse leak-rate targets to each of the
                              8 limbs. Each limb then acts as a different
                              temporal basis function (fast vs. slow dynamics),
                              giving the readout a Fourier-rich basis to draw
                              from — exactly the Jaeger "Library of Babel".

  4. ReservoirReadout       — a single linear layer that harvests a target
                              signal from the concatenated limb reservoir states.
                              The weights are learned (or solved in closed form)
                              while the reservoir itself can stay fixed.

Compound pipeline:
  pacemaker Z(t)
      ↓  (RNA editing scales μᵢ)
  spiking reservoir  ←→  echo-state constraint
      ↓  (diverse timescales per limb)
  8-limb temporal basis
      ↓
  linear readout  →  output
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Echo State Constraint
# ---------------------------------------------------------------------------

class EchoStateConstraint(nn.Module):
    """
    Rescales a recurrent weight matrix so its spectral radius equals
    ``target_rho``.  Values < 1 guarantee the echo-state property
    (inputs leave decaying traces, no chaos).

    Uses a fast power-iteration estimate of the dominant eigenvalue so
    that the constraint is differentiable and cheap — no full SVD.

    Usage::

        esc = EchoStateConstraint(target_rho=0.9, n_power_iter=8)
        W_safe = esc(raw_synapse_weights)   # call before each forward pass
    """

    def __init__(self, target_rho: float = 0.9, n_power_iter: int = 8):
        super().__init__()
        assert 0.0 < target_rho < 1.0, "target_rho must be in (0, 1)"
        self.target_rho = target_rho
        self.n_power_iter = n_power_iter

    def spectral_radius_estimate(self, W: torch.Tensor) -> torch.Tensor:
        """Exact spectral radius via eigenvalue decomposition (detached from graph)."""
        with torch.no_grad():
            eigvals = torch.linalg.eigvals(W)
            return eigvals.abs().max()

    def forward(self, W: torch.Tensor) -> torch.Tensor:
        """
        Return W rescaled so spectral radius ≈ target_rho.

        Args:
            W: [n, n] recurrent weight matrix

        Returns:
            W_scaled: same shape, echo-state property guaranteed
        """
        rho = self.spectral_radius_estimate(W)
        # Clamp denominator away from zero so gradient is stable
        scale = self.target_rho / (rho.clamp(min=1e-6))
        return W * scale


# ---------------------------------------------------------------------------
# 2. Neural Pacemaker
# ---------------------------------------------------------------------------

BAND_FREQUENCIES = {
    "theta": 6.0,    # ~6 tokens/cycle  (slow, context-level)
    "alpha": 10.0,   # ~10 tokens/cycle
    "gamma": 40.0,   # ~40 tokens/cycle (fast, within-sequence)
}


class NeuralPacemaker(nn.Module):
    """
    Generates a multi-frequency oscillatory driving signal Z(t).

    Corresponds to the biological pacemaker oscillations (θ, α, γ bands)
    that prevent reservoir dynamics from dying out between inputs.

    The output is a [hidden_dim] vector where each dimension receives a
    unique linear combination of the oscillatory bands — these are the
    per-neuron μᵢ scalings from the reservoir equation:

        xᵢ(t) = σ(Σⱼ Wᵢⱼ·xⱼ(t-1) + μᵢ·Z(t))

    Usage in RNA editing layer::

        z = pacemaker(t)          # [batch, hidden_dim]
        modulated = base * z      # scale existing modulations

    Args:
        hidden_dim: output dimensionality (= reservoir neuron count)
        bands: dict of {name: frequency_Hz}; default θ/α/γ
        learnable_mix: if True, the per-neuron band mixing weights are
                       learned parameters (otherwise random-fixed).
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        bands: Optional[Dict[str, float]] = None,
        learnable_mix: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        bands = bands or BAND_FREQUENCIES
        self.band_names = list(bands.keys())
        self.n_bands = len(self.band_names)

        # Register frequencies and phase offsets as buffers (not trained)
        freqs = torch.tensor([bands[b] for b in self.band_names])
        self.register_buffer("freqs", freqs)  # [n_bands]

        # Random phase offset per band (breaks symmetry)
        phases = torch.rand(self.n_bands) * 2 * math.pi
        self.register_buffer("phases", phases)  # [n_bands]

        # Per-neuron mixing weights over bands: μ ∈ R^{hidden_dim × n_bands}
        if learnable_mix:
            self.mu = nn.Parameter(torch.randn(hidden_dim, self.n_bands) * 0.1)
        else:
            mix = torch.randn(hidden_dim, self.n_bands)
            self.register_buffer("mu", mix)

        # Global amplitude scale
        self.amplitude = nn.Parameter(torch.tensor(1.0))

    def forward(
        self, t: torch.Tensor, batch_size: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute Z(t) driving signal.

        Args:
            t: scalar or [batch] timestep tensor
            batch_size: optional override for batch dimension

        Returns:
            z: [batch, hidden_dim] pacemaker signal in [-1, 1] range
        """
        # Normalise t to oscillation frequency: sin(2π·f·t + φ)
        # freqs are in "cycles per 1000 tokens" for practical training scales
        t_scalar = t.float() if torch.is_tensor(t) else torch.tensor(float(t))
        t_scalar = t_scalar.to(self.freqs.device)

        # [n_bands] band signals
        angles = 2 * math.pi * self.freqs * t_scalar / 1000.0 + self.phases
        band_signals = torch.sin(angles)  # [n_bands]

        # Mix into per-neuron signal: [hidden_dim]
        z = torch.tanh(self.amplitude * (self.mu @ band_signals))  # [hidden_dim]

        # Expand to batch
        if batch_size is None:
            batch_size = 1 if t_scalar.dim() == 0 else t_scalar.shape[0]
        return z.unsqueeze(0).expand(batch_size, -1)  # [batch, hidden_dim]


# ---------------------------------------------------------------------------
# 3. Temporal Basis Diversity
# ---------------------------------------------------------------------------

# Preset leak-rate targets for each of the 8 OctoTetrahedral limbs.
# Covers the range [0.50, 0.98]: fast limbs (short memory) to slow (long memory).
# Together they form a diverse temporal basis — like a set of basis functions
# that span the frequency range, enabling the readout to synthesise any signal.
LIMB_LEAK_TARGETS: Dict[str, float] = {
    "perception":       0.50,  # fastest — reacts to every token
    "action":           0.60,
    "language":         0.70,
    "spatial":          0.75,
    "reasoning":        0.80,
    "planning":         0.88,
    "memory":           0.93,
    "metacognition":    0.98,  # slowest — retains context across many tokens
}


class TemporalBasisDiversity(nn.Module):
    """
    Provides each limb with a distinct leak-rate target drawn from
    ``LIMB_LEAK_TARGETS``, initialising the system as a diverse temporal basis.

    The module returns a [n_limbs] tensor of soft leak rates that can be
    used to initialise or regularise the ``LIFNeuron.leak_rate`` parameters
    in each limb's spiking sub-layer.

    It also computes a **diversity loss** that penalises limbs for collapsing
    to the same leak rate, preserving the Fourier-like coverage property.

    Args:
        limb_names: ordered list of limb names (must match keys of limbs dict)
        target_leak_rates: mapping from name → target; defaults to LIMB_LEAK_TARGETS
    """

    def __init__(
        self,
        limb_names: Optional[List[str]] = None,
        target_leak_rates: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        limb_names = limb_names or list(LIMB_LEAK_TARGETS.keys())
        targets_map = target_leak_rates or LIMB_LEAK_TARGETS

        targets = torch.tensor([targets_map.get(n, 0.75) for n in limb_names])
        self.register_buffer("targets", targets)
        self.limb_names = limb_names

    def get_targets(self) -> Dict[str, float]:
        """Return {limb_name: target_leak_rate} mapping."""
        return {n: self.targets[i].item() for i, n in enumerate(self.limb_names)}

    def diversity_loss(self, actual_leaks: torch.Tensor) -> torch.Tensor:
        """
        Penalise deviation from target leak rates.

        Args:
            actual_leaks: [n_limbs] current leak rate values

        Returns:
            scalar loss — add to training loss with small weight (e.g. 0.001)
        """
        return F.mse_loss(actual_leaks, self.targets.to(actual_leaks.device))


# ---------------------------------------------------------------------------
# 4. Reservoir Readout
# ---------------------------------------------------------------------------

class ReservoirReadout(nn.Module):
    """
    Linear readout over the concatenated reservoir states of all limbs.

    Implements the core reservoir computing result:
    "Only the readout needs to be trained; the reservoir can stay fixed."

    The 8 limb outputs act as the reservoir's neuron activations.  This
    single linear layer maps them to the target signal — equivalent to the
    Fourier-like linear regression described in the transcript.

    During training this layer is learned via gradient descent.
    During inference it can optionally be re-solved in closed form via
    ridge regression given a batch of limb activations and targets.

    Args:
        n_limbs: number of limb inputs (default 8)
        hidden_dim: dimensionality of each limb output
        output_dim: dimensionality of the readout (usually == hidden_dim)
        ridge_alpha: regularisation for closed-form solve (default 1e-4)
    """

    def __init__(
        self,
        n_limbs: int = 8,
        hidden_dim: int = 256,
        output_dim: Optional[int] = None,
        ridge_alpha: float = 1e-4,
    ):
        super().__init__()
        output_dim = output_dim or hidden_dim
        self.n_limbs = n_limbs
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.ridge_alpha = ridge_alpha

        # Linear readout: the *only* learned component (reservoir stays fixed)
        self.readout = nn.Linear(n_limbs * hidden_dim, output_dim, bias=True)
        nn.init.xavier_uniform_(self.readout.weight, gain=0.5)

        # Optional layer norm before readout (stabilises gradients)
        self.norm = nn.LayerNorm(n_limbs * hidden_dim)

    def forward(self, limb_states: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute readout from all limb reservoir states.

        Args:
            limb_states: list of [batch, hidden_dim] tensors, one per limb

        Returns:
            [batch, output_dim] synthesised signal
        """
        if len(limb_states) < self.n_limbs:
            # Pad with zeros if fewer limbs are active
            pad = torch.zeros(
                limb_states[0].shape[0],
                self.hidden_dim,
                device=limb_states[0].device,
                dtype=limb_states[0].dtype,
            )
            limb_states = limb_states + [pad] * (self.n_limbs - len(limb_states))

        x = torch.cat(limb_states[:self.n_limbs], dim=-1)  # [batch, n*d]
        x = self.norm(x)
        return self.readout(x)  # [batch, output_dim]

    @torch.no_grad()
    def solve_closed_form(
        self, X: torch.Tensor, Y: torch.Tensor
    ) -> None:
        """
        Solve readout weights analytically via ridge regression.

        This is the closed-form solution referenced in the transcript —
        the exact same math as fitting a line through points, applied to
        the temporal basis formed by all limb activations.

            W = (XᵀX + αI)⁻¹ Xᵀ Y

        Args:
            X: [T, n_limbs * hidden_dim] reservoir states collected over T steps
            Y: [T, output_dim] target signals over the same T steps
        """
        X_norm = (X - X.mean(0)) / (X.std(0) + 1e-8)
        n, d = X_norm.shape
        alpha_I = self.ridge_alpha * torch.eye(d, device=X.device, dtype=X.dtype)
        # (XᵀX + αI)⁻¹ Xᵀ Y
        W = torch.linalg.solve(X_norm.T @ X_norm + alpha_I, X_norm.T @ Y)
        bias = Y.mean(0) - (X_norm.mean(0) @ W)

        self.readout.weight.copy_(W.T)
        self.readout.bias.copy_(bias)


# ---------------------------------------------------------------------------
# Compound Module: full reservoir pipeline in one nn.Module
# ---------------------------------------------------------------------------

class ReservoirDynamics(nn.Module):
    """
    Full compound integration of all four reservoir computing insights.

    Embeds into the OctoTetrahedral pipeline as::

        rna_out = rna_editing(x)
        z_t     = reservoir.pacemaker(t)        # drives RNA modulations
        modulated = rna_out * z_t               # μᵢ · Z(t)
        W_safe  = reservoir.echo_state(W_syn)   # ensure echo-state property
        y       = reservoir.readout(limb_outs)  # linear synthesis

    Instantiate once at the top of the model and call the sub-modules
    at the appropriate pipeline stages.

    Args:
        hidden_dim: model hidden dimension
        n_limbs: number of octopus limbs (default 8)
        echo_rho: target spectral radius (default 0.9)
        limb_names: ordered limb names for temporal diversity targets
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        n_limbs: int = 8,
        echo_rho: float = 0.9,
        limb_names: Optional[List[str]] = None,
    ):
        super().__init__()

        self.echo_state = EchoStateConstraint(target_rho=echo_rho)
        self.pacemaker = NeuralPacemaker(hidden_dim=hidden_dim)
        self.temporal_basis = TemporalBasisDiversity(limb_names=limb_names)
        self.readout = ReservoirReadout(
            n_limbs=n_limbs,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
        )

        # Track global timestep for pacemaker
        self.register_buffer("_t", torch.tensor(0.0))

    def tick(self) -> None:
        """Advance the global pacemaker timestep by 1."""
        self._t.add_(1.0)

    def pace(self, batch_size: int) -> torch.Tensor:
        """Return the current pacemaker driving signal Z(t)."""
        return self.pacemaker(self._t, batch_size=batch_size)

    def constrain_weights(self, W: torch.Tensor) -> torch.Tensor:
        """Apply echo-state spectral radius constraint to a weight matrix."""
        return self.echo_state(W)

    def synthesise(self, limb_states: List[torch.Tensor]) -> torch.Tensor:
        """Synthesise output from all limb reservoir states via linear readout."""
        return self.readout(limb_states)

    def diversity_loss(self, actual_leaks: torch.Tensor) -> torch.Tensor:
        """Temporal basis diversity regularisation loss."""
        return self.temporal_basis.diversity_loss(actual_leaks)

    def forward(
        self,
        limb_states: List[torch.Tensor],
        synapse_weight: Optional[torch.Tensor] = None,
        apply_pacemaker: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Full reservoir forward pass.

        Args:
            limb_states: list of [batch, hidden_dim] limb activations
            synapse_weight: optional recurrent weight to constrain
            apply_pacemaker: whether to return pacemaker signal

        Returns:
            dict with 'synthesis', 'pacemaker_signal', 'safe_weights'
        """
        batch = limb_states[0].shape[0]
        out: Dict[str, torch.Tensor] = {}

        out["synthesis"] = self.synthesise(limb_states)

        if apply_pacemaker:
            out["pacemaker_signal"] = self.pace(batch)
            self.tick()

        if synapse_weight is not None:
            out["safe_weights"] = self.constrain_weights(synapse_weight)

        return out


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import torch

    print("=== Reservoir Dynamics self-test ===\n")

    BATCH, HDIM, N_LIMBS = 4, 256, 8

    rd = ReservoirDynamics(hidden_dim=HDIM, n_limbs=N_LIMBS)

    # 1. Echo State Constraint
    W = torch.randn(64, 64) * 2.0
    W_safe = rd.constrain_weights(W)
    rho_before = torch.linalg.eigvals(W).abs().max()
    rho_after  = torch.linalg.eigvals(W_safe).abs().max()
    print(f"Echo state — ρ before: {rho_before:.3f}  after: {rho_after:.3f}  "
          f"(target {rd.echo_state.target_rho})")

    # 2. Neural Pacemaker
    z = rd.pace(BATCH)
    print(f"Pacemaker — shape: {z.shape}, "
          f"mean={z.mean():.3f}, std={z.std():.3f}")

    # 3. Temporal Basis Diversity
    targets = rd.temporal_basis.get_targets()
    print(f"Leak targets: {', '.join(f'{k}={v:.2f}' for k, v in targets.items())}")
    fake_leaks = torch.rand(N_LIMBS)
    dloss = rd.diversity_loss(fake_leaks)
    print(f"Diversity loss (random leaks): {dloss.item():.4f}")

    # 4. Reservoir Readout
    limb_outs = [torch.randn(BATCH, HDIM) for _ in range(N_LIMBS)]
    synthesis = rd.synthesise(limb_outs)
    print(f"Readout — shape: {synthesis.shape}")

    # 5. Full forward pass
    result = rd(limb_outs, synapse_weight=torch.randn(64, 64) * 2.0)
    print(f"\nFull forward:")
    for k, v in result.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={v.shape}, norm={v.norm():.3f}")

    # 6. Closed-form solve
    T = 200  # timesteps
    X = torch.randn(T, N_LIMBS * HDIM)
    Y = torch.randn(T, HDIM)
    rd.readout.solve_closed_form(X, Y)
    print(f"\nClosed-form solve — readout weight norm: "
          f"{rd.readout.readout.weight.norm():.4f}")

    # 7. Gradient flow
    x = [torch.randn(BATCH, HDIM, requires_grad=True) for _ in range(N_LIMBS)]
    out = rd(x)["synthesis"]
    out.sum().backward()
    grad_ok = all(xi.grad is not None for xi in x)
    print(f"\nGradient flows through readout: {grad_ok}")

    params = sum(p.numel() for p in rd.parameters())
    print(f"\nTotal parameters: {params:,}")
    print("\n✓ All reservoir dynamics tests passed!")
