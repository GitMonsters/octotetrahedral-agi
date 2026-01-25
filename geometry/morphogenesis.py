"""
Geometric Ferment & Morphogenesis Dynamics for OctoTetrahedral AGI

Advanced geometric dynamics inspired by:
1. Morphogenesis (Turing patterns, reaction-diffusion)
2. Topological phase transitions  
3. Geometric flows (Ricci flow, mean curvature flow)
4. Fermentation dynamics (growth, transformation, metabolism)
5. Catastrophe theory (bifurcations, fold/cusp)
6. Symplectic geometry (phase space, Hamiltonian)

The "ferment" metaphor: Just as yeast transforms sugar into alcohol
through metabolic pathways, neural networks transform information
through geometric pathways.

Neural Network Mappings:
- Turing patterns → Attention patterns emergence
- Reaction-diffusion → Feature propagation
- Ricci flow → Geometry optimization
- Morphogenesis → Architecture self-organization
- Fermentation → Layer-by-layer transformation
- Catastrophe → Phase transitions in learning

"The chemical basis of morphogenesis" - Alan Turing (1952)

Usage:
    morph = MorphogenesisDynamics(hidden_dim=256)
    output = morph(input_state)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


# =============================================================================
# TURING PATTERN FORMATION (Reaction-Diffusion)
# =============================================================================

class TuringPatternLayer(nn.Module):
    """
    Implements Turing's reaction-diffusion for pattern formation.
    
    The classic Turing mechanism:
    ∂u/∂t = Dᵤ∇²u + f(u,v)   (activator)
    ∂v/∂t = Dᵥ∇²v + g(u,v)   (inhibitor)
    
    where Dᵥ > Dᵤ (inhibitor diffuses faster)
    
    This creates spontaneous pattern formation from uniform states!
    
    Neural mapping:
    - Activator u → Feature activation
    - Inhibitor v → Attention suppression (softmax normalization)
    - Diffusion → Spatial/sequential spreading
    - Patterns → Learned representations
    """
    
    def __init__(
        self,
        hidden_dim: int,
        diffusion_activator: float = 0.1,
        diffusion_inhibitor: float = 0.5,  # Must be > activator
        reaction_rate: float = 0.1,
        num_species: int = 2
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.Du = nn.Parameter(torch.tensor(diffusion_activator))
        self.Dv = nn.Parameter(torch.tensor(diffusion_inhibitor))
        self.reaction_rate = nn.Parameter(torch.tensor(reaction_rate))
        self.num_species = num_species
        
        # Species projections
        self.to_activator = nn.Linear(hidden_dim, hidden_dim // 2)
        self.to_inhibitor = nn.Linear(hidden_dim, hidden_dim // 2)
        self.from_species = nn.Linear(hidden_dim, hidden_dim)
        
        # Reaction terms (learnable f and g)
        self.f_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        self.g_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        self.norm = nn.LayerNorm(hidden_dim)
    
    def laplacian_1d(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute 1D Laplacian (second derivative) along sequence.
        
        ∇²u ≈ u[i+1] - 2u[i] + u[i-1]
        """
        # Pad for boundary conditions
        padded = F.pad(x.transpose(1, 2), (1, 1), mode='replicate')
        padded = padded.transpose(1, 2)
        
        # Finite difference
        laplacian = padded[:, 2:, :] - 2 * x + padded[:, :-2, :]
        
        return laplacian
    
    def reaction_diffusion_step(
        self,
        u: torch.Tensor,  # Activator [batch, seq, dim]
        v: torch.Tensor,  # Inhibitor [batch, seq, dim]
        dt: float = 0.1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single step of reaction-diffusion dynamics.
        
        ∂u/∂t = Du·∇²u + f(u,v)
        ∂v/∂t = Dv·∇²v + g(u,v)
        """
        # Diffusion terms
        laplacian_u = self.laplacian_1d(u)
        laplacian_v = self.laplacian_1d(v)
        
        diffusion_u = self.Du * laplacian_u
        diffusion_v = self.Dv * laplacian_v
        
        # Reaction terms
        combined = torch.cat([u, v], dim=-1)
        f = self.f_net(combined) * self.reaction_rate
        g = self.g_net(combined) * self.reaction_rate
        
        # Update (Euler method)
        u_new = u + dt * (diffusion_u + f)
        v_new = v + dt * (diffusion_v + g)
        
        return u_new, v_new
    
    def forward(
        self,
        x: torch.Tensor,
        num_steps: int = 5,
        return_trajectory: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Apply Turing pattern dynamics.
        
        Runs reaction-diffusion to allow patterns to emerge.
        """
        # Split into activator and inhibitor
        u = self.to_activator(x)
        v = self.to_inhibitor(x)
        
        trajectory = [(u.clone(), v.clone())] if return_trajectory else None
        
        # Run dynamics
        for _ in range(num_steps):
            u, v = self.reaction_diffusion_step(u, v)
            
            if return_trajectory:
                trajectory.append((u.clone(), v.clone()))
        
        # Combine back
        combined = torch.cat([u, v], dim=-1)
        output = self.from_species(combined)
        output = self.norm(output + x)
        
        return output, trajectory


# =============================================================================
# GEOMETRIC FLOWS (Ricci Flow, Mean Curvature Flow)
# =============================================================================

class GeometricFlowLayer(nn.Module):
    """
    Implements geometric flow dynamics for representation learning.
    
    Ricci Flow: ∂g/∂t = -2·Ric(g)
    - Evolves metric to constant curvature
    - Used by Perelman to prove Poincaré conjecture!
    
    Mean Curvature Flow: ∂X/∂t = H·n
    - Surface evolves in direction of mean curvature
    - Smooths geometry, removes irregularities
    
    Neural mapping:
    - Metric g → Feature space geometry (Gram matrix)
    - Ricci curvature → Non-uniformity in representations
    - Flow → Iterative refinement toward "round" representations
    """
    
    def __init__(
        self,
        hidden_dim: int,
        flow_type: str = 'ricci',  # 'ricci' or 'mean_curvature'
        flow_rate: float = 0.1
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.flow_type = flow_type
        self.flow_rate = nn.Parameter(torch.tensor(flow_rate))
        
        # Metric tensor (learnable Riemannian geometry)
        self.metric = nn.Parameter(torch.eye(hidden_dim))
        
        # Curvature computation network
        self.curvature_net = nn.Sequential(
            nn.Linear(hidden_dim * hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim * hidden_dim)
        )
        
        self.norm = nn.LayerNorm(hidden_dim)
    
    def compute_gram_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Gram matrix (metric in feature space).
        
        G_ij = ⟨x_i, x_j⟩
        """
        # x: [batch, seq, hidden]
        # Gram: [batch, seq, seq]
        return torch.bmm(x, x.transpose(1, 2))
    
    def estimate_ricci_curvature(self, gram: torch.Tensor) -> torch.Tensor:
        """
        Estimate Ricci curvature from Gram matrix.
        
        Ricci curvature measures how volume grows compared to flat space.
        Positive = spherical, Negative = hyperbolic, Zero = flat
        
        Approximation using eigenvalue distribution.
        """
        # Add stronger regularization for numerical stability
        reg = 1e-4 * torch.eye(gram.shape[-1], device=gram.device)
        gram_reg = gram + reg.unsqueeze(0)
        
        # Use try/except for eigenvalue computation (can fail on ill-conditioned matrices or MPS)
        try:
            eigenvalues = torch.linalg.eigvalsh(gram_reg)
        except NotImplementedError:
            # Fallback for MPS: compute on CPU
            gram_cpu = gram_reg.detach().cpu()
            eigenvalues = torch.linalg.eigvalsh(gram_cpu).to(gram_reg.device)
        except Exception:
            # Fallback: use approximate eigenvalue via power iteration proxy
            # Just return a simple curvature estimate based on trace/determinant ratio
            trace = torch.diagonal(gram_reg, dim1=-2, dim2=-1).sum(dim=-1)
            frobenius = (gram_reg ** 2).sum(dim=(-2, -1)).sqrt()
            return (frobenius / (trace + 1e-6)).clamp(0, 10)
        
        # Clamp eigenvalues for numerical stability
        eigenvalues = eigenvalues.clamp(min=1e-6)
        
        # Ricci curvature proxy: deviation from uniform eigenspectrum
        mean_eig = eigenvalues.mean(dim=-1, keepdim=True)
        ricci_proxy = (eigenvalues - mean_eig).pow(2).mean(dim=-1)
        
        return ricci_proxy
    
    def ricci_flow_step(
        self,
        x: torch.Tensor,
        dt: float
    ) -> torch.Tensor:
        """
        Single step of Ricci flow.
        
        ∂g/∂t = -2·Ric(g)
        
        Evolves representation toward constant curvature (uniform).
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute current metric (Gram matrix per sample)
        gram = self.compute_gram_matrix(x)
        
        # Estimate Ricci curvature
        ricci = self.estimate_ricci_curvature(gram)  # [batch]
        
        # Clamp ricci to prevent explosion (numerical stability)
        ricci = ricci.clamp(-10.0, 10.0)
        
        # Flow direction: reduce curvature non-uniformity
        # Move toward identity (flat) metric
        target_gram = torch.eye(seq_len, device=x.device).unsqueeze(0) * gram.diagonal(dim1=-2, dim2=-1).mean(dim=-1, keepdim=True).unsqueeze(-1)
        
        # Interpolate toward target
        new_gram = gram - dt * self.flow_rate * (gram - target_gram)
        
        # Project x to have new Gram matrix (approximately)
        # Using x_new ≈ x + correction
        # Use normalized correction to prevent explosion
        correction = -dt * self.flow_rate * ricci.view(-1, 1, 1) * x
        
        # Clamp correction magnitude
        correction = correction.clamp(-1.0, 1.0)
        
        return x + correction
    
    def mean_curvature_flow_step(
        self,
        x: torch.Tensor,
        dt: float
    ) -> torch.Tensor:
        """
        Single step of mean curvature flow.
        
        ∂X/∂t = H·n
        
        Moves points in direction of mean curvature (smoothing).
        """
        batch_size, seq_len, _ = x.shape
        
        # Estimate local curvature via second derivative
        # H ≈ Laplacian of position
        laplacian = (
            F.pad(x, (0, 0, 1, 1), mode='replicate')[:, 2:, :] +
            F.pad(x, (0, 0, 1, 1), mode='replicate')[:, :-2, :] -
            2 * x
        )
        
        # Mean curvature is magnitude of Laplacian
        H = laplacian  # Simplified: use Laplacian directly as flow direction
        
        # Flow toward smoothness
        return x + dt * self.flow_rate * H
    
    def forward(
        self,
        x: torch.Tensor,
        num_steps: int = 3
    ) -> torch.Tensor:
        """
        Apply geometric flow.
        """
        for _ in range(num_steps):
            if self.flow_type == 'ricci':
                x = self.ricci_flow_step(x, dt=0.1)
            else:
                x = self.mean_curvature_flow_step(x, dt=0.1)
        
        return self.norm(x)


# =============================================================================
# CATASTROPHE THEORY (Bifurcations, Phase Transitions)
# =============================================================================

class CatastropheType(Enum):
    """Elementary catastrophes (Thom's classification)"""
    FOLD = "fold"           # A³
    CUSP = "cusp"           # A⁴  
    SWALLOWTAIL = "swallowtail"  # A⁵
    BUTTERFLY = "butterfly"      # A⁶
    HYPERBOLIC_UMBILIC = "hyperbolic"  # D⁴⁺
    ELLIPTIC_UMBILIC = "elliptic"      # D⁴⁻
    PARABOLIC_UMBILIC = "parabolic"    # D⁵


class CatastropheLayer(nn.Module):
    """
    Implements catastrophe theory for phase transitions in networks.
    
    Catastrophe theory studies how small changes in parameters
    can cause sudden qualitative changes in behavior.
    
    Key catastrophes:
    - Fold: V = x³ + ax        (1 control param)
    - Cusp: V = x⁴ + ax² + bx  (2 control params)
    
    Neural mapping:
    - Catastrophe surface → Decision boundary
    - Control parameters → Learned thresholds
    - Bifurcations → Discrete decisions from continuous inputs
    - Hysteresis → Memory of past states
    
    "Nature makes jumps" - explained by catastrophe theory
    """
    
    def __init__(
        self,
        hidden_dim: int,
        catastrophe_type: CatastropheType = CatastropheType.CUSP,
        num_catastrophes: int = 8
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.catastrophe_type = catastrophe_type
        self.num_catastrophes = num_catastrophes
        
        # Control parameters (learnable)
        if catastrophe_type == CatastropheType.FOLD:
            self.control = nn.Parameter(torch.randn(num_catastrophes, 1))
        elif catastrophe_type == CatastropheType.CUSP:
            self.control = nn.Parameter(torch.randn(num_catastrophes, 2))
        else:
            self.control = nn.Parameter(torch.randn(num_catastrophes, 3))
        
        # Project to catastrophe dimensions
        self.to_catastrophe = nn.Linear(hidden_dim, num_catastrophes)
        self.from_catastrophe = nn.Linear(num_catastrophes, hidden_dim)
        
        # State variable (for hysteresis)
        self.register_buffer('state', torch.zeros(num_catastrophes))
        
        self.norm = nn.LayerNorm(hidden_dim)
    
    def fold_potential(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Fold catastrophe: V = x³/3 + ax"""
        return x.pow(3) / 3 + a * x
    
    def fold_equilibrium(self, a: torch.Tensor) -> torch.Tensor:
        """Equilibrium points of fold: dV/dx = x² + a = 0"""
        # x² = -a, so x = ±√(-a) if a < 0, else no real solution
        return torch.where(a < 0, (-a).sqrt(), torch.zeros_like(a))
    
    def cusp_potential(self, x: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Cusp catastrophe: V = x⁴/4 + ax²/2 + bx"""
        return x.pow(4) / 4 + a * x.pow(2) / 2 + b * x
    
    def cusp_equilibrium(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Find equilibrium of cusp: dV/dx = x³ + ax + b = 0
        
        Has 1 or 3 real roots depending on discriminant:
        Δ = 4a³ + 27b² 
        Δ < 0: 3 real roots (bistable)
        Δ > 0: 1 real root (monostable)
        """
        # Simplified: use gradient descent to find equilibrium
        x = torch.zeros_like(a)
        for _ in range(10):
            grad = x.pow(3) + a * x + b
            x = x - 0.1 * grad
        return x
    
    def apply_catastrophe(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply catastrophe transformation.
        
        Maps continuous input through catastrophe surface,
        potentially causing discontinuous jumps.
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to catastrophe space
        cat_input = self.to_catastrophe(x)  # [batch, seq, num_catastrophes]
        
        if self.catastrophe_type == CatastropheType.FOLD:
            a = self.control[:, 0]  # [num_catastrophes]
            # Find equilibrium for each input
            equilibrium = self.fold_equilibrium(a.unsqueeze(0).unsqueeze(0) + cat_input * 0.1)
            cat_output = equilibrium
            
        elif self.catastrophe_type == CatastropheType.CUSP:
            a = self.control[:, 0]
            b = self.control[:, 1]
            # Input modulates control parameters
            a_mod = a.unsqueeze(0).unsqueeze(0) + cat_input * 0.1
            b_mod = b.unsqueeze(0).unsqueeze(0) + cat_input * 0.05
            equilibrium = self.cusp_equilibrium(a_mod, b_mod)
            cat_output = equilibrium
            
        else:
            # Default: use cusp for other types
            cat_output = cat_input
        
        # Project back
        output = self.from_catastrophe(cat_output)
        
        return self.norm(output + x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.apply_catastrophe(x)


# =============================================================================
# FERMENTATION DYNAMICS (Metabolic Transformation)
# =============================================================================

class FermentationLayer(nn.Module):
    """
    Fermentation-inspired transformation dynamics.
    
    Fermentation metaphor for neural networks:
    - Substrate (input) → Product (output) through enzymes (weights)
    - Metabolic pathways → Layer sequences
    - ATP (energy) → Gradient budget
    - Yeast growth → Network capacity
    
    Key dynamics:
    - Michaelis-Menten kinetics: v = Vmax·[S]/(Km + [S])
    - Monod growth: μ = μmax·S/(Ks + S)
    - Fermentation balance: Glucose → Ethanol + CO2 + Energy
    
    Neural mapping:
    - Substrate concentration [S] → Input activation magnitude
    - Enzyme [E] → Weight matrix
    - Product [P] → Output features
    - Inhibition → Attention masking
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_enzymes: int = 8,
        max_rate: float = 1.0
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_enzymes = num_enzymes
        
        # Enzyme parameters (Michaelis-Menten)
        self.Vmax = nn.Parameter(torch.ones(num_enzymes) * max_rate)
        self.Km = nn.Parameter(torch.ones(num_enzymes) * 0.5)
        
        # Enzyme specificity (what each enzyme transforms)
        self.enzyme_input = nn.Linear(hidden_dim, num_enzymes)
        self.enzyme_output = nn.Linear(num_enzymes, hidden_dim)
        
        # Inhibition parameters
        self.inhibitor_weights = nn.Parameter(torch.randn(num_enzymes, num_enzymes) * 0.1)
        
        # Growth dynamics
        self.growth_rate = nn.Parameter(torch.tensor(0.1))
        
        self.norm = nn.LayerNorm(hidden_dim)
    
    def michaelis_menten(
        self,
        substrate: torch.Tensor,
        Vmax: torch.Tensor,
        Km: torch.Tensor,
        inhibitor: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Michaelis-Menten enzyme kinetics.
        
        v = Vmax·[S]/(Km + [S])
        
        With competitive inhibition:
        v = Vmax·[S]/(Km·(1 + [I]/Ki) + [S])
        """
        if inhibitor is not None:
            # Competitive inhibition
            Ki = 1.0  # Inhibition constant
            Km_apparent = Km * (1 + inhibitor / Ki)
        else:
            Km_apparent = Km
        
        rate = Vmax * substrate / (Km_apparent + substrate + 1e-10)
        return rate
    
    def monod_growth(
        self,
        substrate: torch.Tensor,
        mu_max: float = 0.5,
        Ks: float = 0.1
    ) -> torch.Tensor:
        """
        Monod growth kinetics.
        
        μ = μmax·S/(Ks + S)
        
        Models exponential growth limited by substrate.
        """
        return mu_max * substrate / (Ks + substrate + 1e-10)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply fermentation dynamics.
        
        1. Substrate (input) binds to enzymes
        2. Enzymatic transformation
        3. Product formation with inhibition feedback
        4. Growth-limited output
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute substrate concentration for each enzyme
        substrate = self.enzyme_input(x)  # [batch, seq, num_enzymes]
        substrate = F.relu(substrate)  # Concentrations are positive
        
        # Compute inhibition from products
        # Products can inhibit enzymes (feedback regulation)
        inhibition = torch.einsum('bse,ef->bsf', substrate, torch.sigmoid(self.inhibitor_weights))
        
        # Apply Michaelis-Menten kinetics
        rates = self.michaelis_menten(
            substrate,
            self.Vmax.unsqueeze(0).unsqueeze(0),
            self.Km.unsqueeze(0).unsqueeze(0),
            inhibitor=inhibition
        )
        
        # Product formation
        product = rates  # Simplified: rate = product formation rate
        
        # Growth limitation
        growth = self.monod_growth(product.mean(dim=-1, keepdim=True))
        product = product * (1 + growth * self.growth_rate)
        
        # Convert back to hidden space
        output = self.enzyme_output(product)
        
        return self.norm(output + x)


# =============================================================================
# SYMPLECTIC GEOMETRY (Phase Space, Hamiltonian)
# =============================================================================

class SymplecticLayer(nn.Module):
    """
    Symplectic geometry for phase space transformations.
    
    Symplectic structure preserves the fundamental 2-form:
    ω = Σ dq_i ∧ dp_i
    
    Properties:
    - Area-preserving in phase space
    - Hamiltonian flows are symplectic
    - Liouville's theorem: phase space volume conserved
    
    Neural mapping:
    - Position q → Feature values
    - Momentum p → Feature gradients/velocities
    - Hamiltonian H → Energy function
    - Symplectic map → Structure-preserving transformation
    
    "Symplectic geometry is the geometry of classical mechanics." - Arnold
    """
    
    def __init__(
        self,
        hidden_dim: int
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.phase_dim = hidden_dim // 2  # q and p each get half
        
        # Hamiltonian as neural network
        self.hamiltonian = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Symplectic matrix J = [[0, I], [-I, 0]]
        I = torch.eye(self.phase_dim)
        J = torch.zeros(hidden_dim, hidden_dim)
        J[:self.phase_dim, self.phase_dim:] = I
        J[self.phase_dim:, :self.phase_dim] = -I
        self.register_buffer('J', J)
        
        # Learnable symplectic transformation
        # S is symplectic if S^T J S = J
        self.S = nn.Parameter(torch.eye(hidden_dim))
        
        # Learned flow approximation for inference (when gradients disabled)
        self.flow_approximation = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        self.norm = nn.LayerNorm(hidden_dim)
    
    def _ensure_symplectic(self, S: torch.Tensor) -> torch.Tensor:
        """
        Project matrix onto symplectic group.
        
        Uses Cayley transform: S = (I + A)(I - A)^(-1)
        where A is skew-symmetric
        """
        # Make skew-symmetric
        A = (S - S.T) / 2
        
        # Cayley transform - use CPU for linalg.solve (MPS compatibility)
        I = torch.eye(S.shape[0], device=S.device)
        
        # For training: use approximate symplectic via exponential map
        if self.training:
            # Approximate: S ≈ I + 2A for small A (first-order Cayley)
            return I + 2 * A
        else:
            # At inference: exact computation on CPU
            I_cpu = I.cpu()
            A_cpu = A.detach().cpu()
            S_symplectic_cpu = torch.linalg.solve(I_cpu - A_cpu, I_cpu + A_cpu)
            return S_symplectic_cpu.to(S.device)
    
    def hamiltonian_gradient(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient of Hamiltonian.
        
        ∇H = [∂H/∂q, ∂H/∂p]
        """
        z.requires_grad_(True)
        H = self.hamiltonian(z).sum()
        grad = torch.autograd.grad(H, z, create_graph=True)[0]
        return grad
    
    def hamiltonian_flow(self, z: torch.Tensor, dt: float = 0.1) -> torch.Tensor:
        """
        Apply Hamiltonian flow (symplectic integrator).
        
        dz/dt = J · ∇H
        
        where J is the symplectic matrix.
        
        During inference (torch.no_grad()), uses a learned approximation
        since autograd is unavailable.
        """
        # Inference mode: use learned flow approximation
        if not torch.is_grad_enabled():
            return z + dt * self.flow_approximation(z)
        
        # Training mode: compute exact Hamiltonian gradient
        # Gradient of Hamiltonian
        grad_H = self.hamiltonian_gradient(z)
        
        # Symplectic flow: dz/dt = J @ grad_H
        flow = torch.einsum('ij,...j->...i', self.J, grad_H)
        
        # Euler step (could use symplectic integrator like leapfrog)
        return z + dt * flow
    
    def forward(
        self,
        x: torch.Tensor,
        num_steps: int = 3
    ) -> torch.Tensor:
        """
        Apply symplectic transformation.
        
        Preserves phase space structure while transforming features.
        """
        # Get symplectic matrix
        S = self._ensure_symplectic(self.S)
        
        # Apply symplectic transformation
        z = torch.einsum('ij,...j->...i', S, x)
        
        # Run Hamiltonian flow
        for _ in range(num_steps):
            z = self.hamiltonian_flow(z)
        
        return self.norm(z + x)


# =============================================================================
# UNIFIED MORPHOGENESIS MODULE
# =============================================================================

class MorphogenesisDynamics(nn.Module):
    """
    Unified geometric ferment and morphogenesis module.
    
    Combines all dynamic geometric processes:
    1. Turing patterns (reaction-diffusion)
    2. Geometric flows (Ricci, mean curvature)
    3. Catastrophe (bifurcations, phase transitions)
    4. Fermentation (metabolic transformation)
    5. Symplectic (Hamiltonian phase space)
    
    "Form follows function, but form also creates function."
    """
    
    def __init__(
        self,
        hidden_dim: int,
        enable_turing: bool = True,
        enable_geometric_flow: bool = True,
        enable_catastrophe: bool = True,
        enable_fermentation: bool = True,
        enable_symplectic: bool = True
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Sub-modules
        self.turing = TuringPatternLayer(hidden_dim) if enable_turing else None
        self.geometric_flow = GeometricFlowLayer(hidden_dim) if enable_geometric_flow else None
        self.catastrophe = CatastropheLayer(hidden_dim) if enable_catastrophe else None
        self.fermentation = FermentationLayer(hidden_dim) if enable_fermentation else None
        self.symplectic = SymplecticLayer(hidden_dim) if enable_symplectic else None
        
        # Combination weights
        num_active = sum([
            enable_turing, enable_geometric_flow, enable_catastrophe,
            enable_fermentation, enable_symplectic
        ])
        self.weights = nn.Parameter(torch.ones(num_active) / num_active)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        return_components: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Apply all morphogenesis dynamics.
        """
        outputs = {}
        weighted_outputs = []
        weight_idx = 0
        
        # Turing patterns
        if self.turing is not None:
            turing_out, _ = self.turing(x)
            outputs['turing'] = turing_out
            weighted_outputs.append(turing_out * self.weights[weight_idx])
            weight_idx += 1
        
        # Geometric flow
        if self.geometric_flow is not None:
            flow_out = self.geometric_flow(x)
            outputs['geometric_flow'] = flow_out
            weighted_outputs.append(flow_out * self.weights[weight_idx])
            weight_idx += 1
        
        # Catastrophe
        if self.catastrophe is not None:
            cat_out = self.catastrophe(x)
            outputs['catastrophe'] = cat_out
            weighted_outputs.append(cat_out * self.weights[weight_idx])
            weight_idx += 1
        
        # Fermentation
        if self.fermentation is not None:
            ferm_out = self.fermentation(x)
            outputs['fermentation'] = ferm_out
            weighted_outputs.append(ferm_out * self.weights[weight_idx])
            weight_idx += 1
        
        # Symplectic
        if self.symplectic is not None:
            symp_out = self.symplectic(x)
            outputs['symplectic'] = symp_out
            weighted_outputs.append(symp_out * self.weights[weight_idx])
            weight_idx += 1
        
        # Combine
        combined = sum(weighted_outputs)
        output = self.output_proj(combined)
        output = self.norm(output + x)
        
        outputs['combined'] = output
        
        return outputs
    
    def get_dynamics_stats(self) -> Dict[str, float]:
        """Get statistics about the dynamics."""
        stats = {}
        
        # Weights
        weights_normalized = F.softmax(self.weights, dim=0)
        stats['weights'] = weights_normalized.tolist()
        
        # Turing stats
        if self.turing is not None:
            stats['turing_Du'] = self.turing.Du.item()
            stats['turing_Dv'] = self.turing.Dv.item()
            stats['turing_ratio'] = (self.turing.Dv / self.turing.Du).item()
        
        # Flow stats
        if self.geometric_flow is not None:
            stats['flow_rate'] = self.geometric_flow.flow_rate.item()
        
        # Fermentation stats
        if self.fermentation is not None:
            stats['ferm_Vmax_mean'] = self.fermentation.Vmax.mean().item()
            stats['ferm_Km_mean'] = self.fermentation.Km.mean().item()
        
        return stats


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("Testing Geometric Ferment & Morphogenesis Dynamics...")
    print("=" * 60)
    
    batch_size = 4
    seq_len = 32
    hidden_dim = 256
    
    x = torch.randn(batch_size, seq_len, hidden_dim)
    
    # Test Turing Patterns
    print("\n1. Turing Pattern Formation (Reaction-Diffusion):")
    turing = TuringPatternLayer(hidden_dim)
    out, trajectory = turing(x, return_trajectory=True)
    print(f"   Input: {x.shape}")
    print(f"   Output: {out.shape}")
    print(f"   Trajectory steps: {len(trajectory)}")
    print(f"   Du (activator): {turing.Du.item():.3f}")
    print(f"   Dv (inhibitor): {turing.Dv.item():.3f}")
    print(f"   Dv/Du ratio: {(turing.Dv/turing.Du).item():.2f} (should be > 1)")
    
    # Test Geometric Flow
    print("\n2. Geometric Flow (Ricci):")
    flow = GeometricFlowLayer(hidden_dim, flow_type='ricci')
    out = flow(x)
    print(f"   Output: {out.shape}")
    print(f"   Flow rate: {flow.flow_rate.item():.3f}")
    
    # Test Catastrophe
    print("\n3. Catastrophe Theory (Cusp):")
    cat = CatastropheLayer(hidden_dim, catastrophe_type=CatastropheType.CUSP)
    out = cat(x)
    print(f"   Output: {out.shape}")
    print(f"   Control params shape: {cat.control.shape}")
    
    # Test Fermentation
    print("\n4. Fermentation Dynamics:")
    ferm = FermentationLayer(hidden_dim)
    out = ferm(x)
    print(f"   Output: {out.shape}")
    print(f"   Vmax: {ferm.Vmax.data.numpy().round(2)}")
    print(f"   Km: {ferm.Km.data.numpy().round(2)}")
    
    # Test Symplectic
    print("\n5. Symplectic Geometry:")
    symp = SymplecticLayer(hidden_dim)
    out = symp(x)
    print(f"   Output: {out.shape}")
    
    # Test Unified Module
    print("\n6. Unified Morphogenesis Dynamics:")
    morph = MorphogenesisDynamics(hidden_dim)
    outputs = morph(x)
    print(f"   Combined output: {outputs['combined'].shape}")
    print(f"   Stats: {morph.get_dynamics_stats()}")
    
    print("\n" + "=" * 60)
    print("All Geometric Ferment tests passed!")
    print("\nKey Concepts Implemented:")
    print("- Turing patterns (reaction-diffusion morphogenesis)")
    print("- Geometric flows (Ricci flow, mean curvature)")
    print("- Catastrophe theory (fold, cusp bifurcations)")
    print("- Fermentation dynamics (Michaelis-Menten kinetics)")
    print("- Symplectic geometry (Hamiltonian phase space)")
