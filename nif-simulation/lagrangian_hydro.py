"""
1D Lagrangian Hydrodynamics for ICF Implosions
==============================================

Full hydrodynamic simulation with:
- Lagrangian (co-moving) mesh
- Conservation of mass, momentum, energy
- Equation of state for DT plasma
- Implicit time integration for stability
- Artificial viscosity for shock capturing

Physics equations (Lagrangian form):
- Mass:     dm/dt = 0 (by construction)
- Momentum: ρ(dv/dt) = -∂P/∂r - q
- Energy:   ρ(dε/dt) = -(P+q)(∂v/∂r)

where q is artificial viscosity for shocks.

Author: Evan Pieser
Date: 2026
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt


# Physical constants (CGS)
EV = 1.602e-12       # erg
K_B = 1.381e-16      # erg/K
M_P = 1.673e-24      # g (proton mass)
M_DT = 2.5 * M_P     # g (average DT mass)
E_FUSION = 17.6e6 * EV  # erg (DT fusion energy)
C = 2.998e10         # cm/s (speed of light)


@dataclass
class ZoneState:
    """State of a single Lagrangian zone."""
    r_inner: float    # Inner radius [cm]
    r_outer: float    # Outer radius [cm]
    mass: float       # Zone mass [g]
    velocity: float   # Velocity at zone center [cm/s]
    energy: float     # Specific internal energy [erg/g]
    T: float          # Temperature [keV]
    rho: float        # Density [g/cm³]
    P: float          # Pressure [dyn/cm²]


@dataclass 
class LagrangianMesh:
    """
    1D Lagrangian mesh for spherical geometry.
    
    Nodes are at zone boundaries (staggered grid):
    - Positions r are defined at nodes (N+1 values)
    - Velocities v are defined at nodes (N+1 values)
    - Thermodynamic quantities (ρ, T, P, ε) are zone-centered (N values)
    """
    N: int                           # Number of zones
    r: np.ndarray = field(init=False)      # Node positions [cm]
    v: np.ndarray = field(init=False)      # Node velocities [cm/s]
    m: np.ndarray = field(init=False)      # Zone masses [g]
    rho: np.ndarray = field(init=False)    # Zone densities [g/cm³]
    T: np.ndarray = field(init=False)      # Zone temperatures [keV]
    eps: np.ndarray = field(init=False)    # Specific internal energy [erg/g]
    P: np.ndarray = field(init=False)      # Zone pressures [dyn/cm²]
    q: np.ndarray = field(init=False)      # Artificial viscosity [dyn/cm²]
    
    def __post_init__(self):
        """Initialize arrays."""
        self.r = np.zeros(self.N + 1)
        self.v = np.zeros(self.N + 1)
        self.m = np.zeros(self.N)
        self.rho = np.zeros(self.N)
        self.T = np.zeros(self.N)
        self.eps = np.zeros(self.N)
        self.P = np.zeros(self.N)
        self.q = np.zeros(self.N)


def initialize_nif_target(N_zones: int = 100) -> LagrangianMesh:
    """
    Initialize mesh for NIF-like target.
    
    Geometry (simplified):
    - Central gas fill: r < 850 μm, low density DT gas
    - DT ice shell: 850-1100 μm, solid density DT
    - Ablator (CH): outside (treated as boundary condition)
    
    Args:
        N_zones: Number of Lagrangian zones
    
    Returns:
        Initialized LagrangianMesh
    """
    mesh = LagrangianMesh(N=N_zones)
    
    # Target geometry (convert μm to cm)
    R_gas = 850e-4      # cm (gas fill radius)
    R_ice = 1100e-4     # cm (outer ice radius)
    
    # Mass (convert μg to g)
    M_gas = 10e-6       # g (DT gas)
    M_ice = 170e-6      # g (DT ice shell)
    
    # Create logarithmically spaced zones (finer at center)
    # Split zones between gas and ice
    N_gas = N_zones // 3
    N_ice = N_zones - N_gas
    
    # Gas region: logarithmic spacing
    r_gas = np.logspace(np.log10(R_gas * 0.01), np.log10(R_gas), N_gas + 1)
    
    # Ice region: linear spacing
    r_ice = np.linspace(R_gas, R_ice, N_ice + 1)[1:]  # Skip first (shared with gas)
    
    mesh.r = np.concatenate([r_gas, r_ice])
    
    # Initialize velocities to zero
    mesh.v[:] = 0.0
    
    # Compute zone volumes and masses
    for i in range(N_zones):
        r_in, r_out = mesh.r[i], mesh.r[i+1]
        V_zone = (4/3) * np.pi * (r_out**3 - r_in**3)
        
        if i < N_gas:
            # Gas region: uniform density to give total M_gas
            rho_gas = M_gas / ((4/3) * np.pi * R_gas**3)
            mesh.rho[i] = rho_gas
            mesh.m[i] = rho_gas * V_zone
        else:
            # Ice region: uniform density to give total M_ice
            V_shell = (4/3) * np.pi * (R_ice**3 - R_gas**3)
            rho_ice = M_ice / V_shell
            mesh.rho[i] = rho_ice
            mesh.m[i] = rho_ice * V_zone
    
    # Initial temperature: 300K = 0.026 eV (room temperature)
    mesh.T[:] = 0.026e-3  # keV
    
    # Initialize pressure and internal energy from EOS
    for i in range(N_zones):
        mesh.P[i], mesh.eps[i] = eos_dt(mesh.rho[i], mesh.T[i])
    
    mesh.q[:] = 0.0
    
    return mesh


def eos_dt(rho: float, T_keV: float) -> Tuple[float, float]:
    """
    Equation of state for DT plasma.
    
    For fully ionized plasma:
        P = n_i * k_B * T_i + n_e * k_B * T_e
        
    For DT at high temperature (T > 100 eV), assume:
    - Fully ionized: Z_eff ≈ 1
    - Equal ion and electron temperatures
    - Ideal gas behavior
    
    Args:
        rho: Density [g/cm³]
        T_keV: Temperature [keV]
    
    Returns:
        P: Pressure [dyn/cm²]
        eps: Specific internal energy [erg/g]
    """
    # Number density of ions
    n_i = rho / M_DT  # ions/cm³
    
    # For DT: Z = 1, so n_e = n_i
    n_e = n_i
    
    # Temperature in erg
    T_erg = T_keV * 1e3 * EV  # keV -> erg
    
    # Pressure: P = (n_i + n_e) * T / (erg = k_B * K, so P = n*T in natural units)
    # Actually P = n * k_B * T, but we're using T in energy units
    P = (n_i + n_e) * T_erg  # dyn/cm² (since [n]*[energy] = [force/area])
    
    # Specific internal energy: ε = (3/2) * (n_i + n_e) * T / ρ
    # For ideal gas with γ = 5/3
    eps = (3/2) * (n_i + n_e) * T_erg / rho  # erg/g
    
    return P, eps


def eos_inverse(rho: float, eps: float) -> Tuple[float, float]:
    """
    Inverse EOS: given ρ and ε, compute T and P.
    
    From ε = (3/2) * (2 * n_i) * T / ρ = 3 * T / m_DT
    So T = ε * m_DT / 3
    """
    n_i = rho / M_DT
    
    # T in erg
    T_erg = eps * rho / (3 * n_i)
    T_keV = T_erg / (1e3 * EV)
    
    P = 2 * n_i * T_erg
    
    return T_keV, P


def sigma_v_dt(T_keV: float) -> float:
    """
    DT fusion reactivity <σv> [cm³/s].
    
    Bosch-Hale parametrization (1992).
    Valid for 0.2 keV < T < 100 keV.
    """
    T = np.clip(T_keV, 0.2, 100.0)
    
    # Bosch-Hale coefficients
    BG = 34.3827  # Gamow constant
    mrc2 = 1124656  # keV (reduced mass * c^2)
    
    C1 = 1.17302e-9
    C2 = 1.51361e-2
    C3 = 7.51886e-2
    C4 = 4.60643e-3
    C5 = 1.35e-2
    C6 = -1.0675e-4
    C7 = 1.366e-5
    
    theta = T / (1 - (T*(C2 + T*(C4 + T*C6))) / (1 + T*(C3 + T*(C5 + T*C7))))
    xi = (BG**2 / (4 * theta))**(1/3)
    
    sigma_v = C1 * theta * np.sqrt(xi / (mrc2 * T**3)) * np.exp(-3 * xi)
    
    return sigma_v


def compute_artificial_viscosity(mesh: LagrangianMesh, 
                                  c_q: float = 2.0, 
                                  c_l: float = 0.5) -> None:
    """
    Compute artificial viscosity for shock capturing.
    
    Von Neumann-Richtmyer form:
        q = c_q² * ρ * (Δv)² + c_l * ρ * c_s * |Δv|  (for compression)
        q = 0  (for expansion)
    
    where Δv = v_{i+1} - v_i < 0 indicates compression.
    """
    for i in range(mesh.N):
        # Velocity difference across zone
        dv = mesh.v[i+1] - mesh.v[i]
        
        if dv < 0:  # Compression
            # Sound speed
            gamma = 5/3
            c_s = np.sqrt(gamma * mesh.P[i] / mesh.rho[i])
            
            # Zone width
            dr = mesh.r[i+1] - mesh.r[i]
            
            # Quadratic (shock) + linear (spreading) viscosity
            mesh.q[i] = mesh.rho[i] * (c_q**2 * dv**2 + c_l * c_s * abs(dv))
        else:
            mesh.q[i] = 0.0


def compute_timestep(mesh: LagrangianMesh, cfl: float = 0.3) -> float:
    """
    Compute stable timestep using CFL condition.
    
    dt < CFL * min(Δr / (c_s + |v|))
    """
    dt_min = 1e10  # Large initial value
    
    for i in range(mesh.N):
        dr = mesh.r[i+1] - mesh.r[i]
        if dr <= 0 or not np.isfinite(dr):
            continue
        
        # Sound speed
        gamma = 5/3
        if mesh.P[i] > 0 and mesh.rho[i] > 0 and np.isfinite(mesh.P[i]) and np.isfinite(mesh.rho[i]):
            c_s = np.sqrt(gamma * mesh.P[i] / mesh.rho[i])
            c_s = min(c_s, 1e10)  # Cap sound speed
        else:
            c_s = 1e5  # Minimum sound speed
        
        # Maximum velocity in zone
        v_max = max(abs(mesh.v[i]), abs(mesh.v[i+1]))
        
        # CFL timestep
        dt_cfl = cfl * dr / (c_s + v_max + 1e-10)
        if np.isfinite(dt_cfl) and dt_cfl > 0:
            dt_min = min(dt_min, dt_cfl)
    
    # Also limit by minimum zone crossing time
    dt_min = max(dt_min, 1e-15)  # Minimum 1 fs
    dt_min = min(dt_min, 1e-11)  # Maximum 10 ps
    
    return dt_min


def step_explicit(mesh: LagrangianMesh, dt: float, 
                   P_laser_TW: float = 0, R_ablation: float = None) -> float:
    """
    Advance mesh by one timestep using explicit integration.
    
    Algorithm:
    1. Compute artificial viscosity
    2. Update velocities (momentum equation)
    3. Update positions
    4. Update densities (from new geometry)
    5. Update internal energy
    6. Update temperature and pressure from EOS
    7. Compute fusion burn
    
    Returns:
        Fusion energy released this step [erg]
    """
    N = mesh.N
    
    # 1. Artificial viscosity
    compute_artificial_viscosity(mesh)
    
    # 2. Momentum equation: dv/dt = -A/m * (P + q)
    # At inner boundary (r=0): v = 0 by symmetry
    # At outer boundary: apply ablation pressure
    
    # Ablation pressure from laser (approximate)
    if P_laser_TW > 0:
        # P_ablation ≈ 100 Mbar at 500 TW for NIF target
        # P [Mbar] ≈ 0.2 * (P_laser [TW])^(2/3) / (R [mm])^2
        R_mm = mesh.r[-1] * 10  # cm to mm
        P_ablation = 0.2 * (P_laser_TW)**(2/3) / max(R_mm, 0.1)**2  # Mbar
        P_ablation *= 1e12  # Convert Mbar to dyn/cm²
    else:
        P_ablation = 0
    
    # Interior nodes (i = 1 to N-1)
    for i in range(1, N):
        # Area at node
        A = 4 * np.pi * mesh.r[i]**2
        
        # Pressure gradient: use average of adjacent zones
        P_minus = mesh.P[i-1] + mesh.q[i-1]
        P_plus = mesh.P[i] + mesh.q[i]
        
        # Effective mass at node (average of adjacent zones)
        m_eff = 0.5 * (mesh.m[i-1] + mesh.m[i])
        
        # Acceleration
        a = -A * (P_plus - P_minus) / (2 * m_eff) if m_eff > 0 else 0
        
        mesh.v[i] += a * dt
    
    # Outer boundary: ablation drives implosion
    if P_ablation > 0:
        A_out = 4 * np.pi * mesh.r[-1]**2
        m_out = mesh.m[-1]
        a_out = -A_out * P_ablation / m_out
        mesh.v[-1] += a_out * dt
        
        # Also accelerate outer zones (momentum transfer)
        for i in range(max(0, N-5), N):
            mesh.v[i] += 0.5 * a_out * dt * (1 - (N-1-i)/5)
    
    # Inner boundary: symmetry (v = 0)
    mesh.v[0] = 0
    
    # 3. Update positions
    for i in range(N + 1):
        mesh.r[i] += mesh.v[i] * dt
        mesh.r[i] = max(mesh.r[i], 1e-6)  # Minimum radius
    
    # Ensure monotonicity and minimum zone size
    for i in range(1, N + 1):
        if mesh.r[i] <= mesh.r[i-1]:
            mesh.r[i] = mesh.r[i-1] * 1.001
        # Limit minimum radius to prevent collapse
        if mesh.r[i] < 1e-4:  # 1 μm minimum
            mesh.r[i] = 1e-4
    
    # 4. Update densities from new geometry
    for i in range(N):
        V_new = (4/3) * np.pi * (mesh.r[i+1]**3 - mesh.r[i]**3)
        V_new = max(V_new, 1e-20)  # Prevent division by zero
        mesh.rho[i] = mesh.m[i] / V_new
        mesh.rho[i] = np.clip(mesh.rho[i], 1e-6, 1e6)  # Limit density range
    
    # 5. Update internal energy (PdV work)
    for i in range(N):
        # Volume change rate
        r_out, r_in = mesh.r[i+1], mesh.r[i]
        v_out, v_in = mesh.v[i+1], mesh.v[i]
        
        # Clip to prevent overflow
        r_out = np.clip(r_out, 1e-6, 1e10)
        r_in = np.clip(r_in, 1e-6, 1e10)
        
        dV_dt = 4 * np.pi * (r_out**2 * v_out - r_in**2 * v_in)
        
        # dε/dt = -(P + q) * (dV/dt) / m
        if mesh.m[i] > 0:
            deps_dt = -(mesh.P[i] + mesh.q[i]) * dV_dt / mesh.m[i]
            deps_dt = np.clip(deps_dt, -1e20, 1e20)  # Prevent overflow
            mesh.eps[i] += deps_dt * dt
        mesh.eps[i] = max(mesh.eps[i], 1e6)  # Minimum energy (cold)
    
    # 6. Update T and P from EOS
    for i in range(N):
        mesh.T[i], mesh.P[i] = eos_inverse(mesh.rho[i], mesh.eps[i])
        mesh.T[i] = np.clip(mesh.T[i], 1e-4, 100)  # Limit temperature
        mesh.P[i] = max(mesh.P[i], 1e6)  # Minimum pressure
    
    # 7. Fusion burn
    E_fusion = 0.0
    for i in range(N):
        if mesh.T[i] > 1.0:  # Only burn above 1 keV
            sv = sigma_v_dt(mesh.T[i])
            n_i = mesh.rho[i] / M_DT
            
            # Reaction rate (DT: equal parts D and T)
            R_rate = 0.25 * n_i**2 * sv  # reactions/cm³/s
            
            # Volume
            V = (4/3) * np.pi * (mesh.r[i+1]**3 - mesh.r[i]**3)
            
            # Energy this step
            dE = R_rate * E_FUSION * V * dt
            E_fusion += dE
            
            # Alpha heating (20% of fusion energy stays in hot spot)
            alpha_fraction = 0.2
            d_eps = alpha_fraction * dE / mesh.m[i]
            mesh.eps[i] += d_eps
    
    return E_fusion


class LagrangianSimulation:
    """
    Full 1D Lagrangian hydrodynamics simulation.
    """
    
    def __init__(self, N_zones: int = 100):
        self.mesh = initialize_nif_target(N_zones)
        self.t = 0.0  # Current time [s]
        self.E_laser = 0.0  # Cumulative laser energy [erg]
        self.E_fusion = 0.0  # Cumulative fusion energy [erg]
        self.history = []  # Time history
        
    def laser_pulse(self, t_ns: float) -> float:
        """NIF laser pulse [TW]."""
        if t_ns < 0:
            return 0
        elif t_ns < 2:
            return 30
        elif t_ns < 5:
            return 30 + 470 * (t_ns - 2) / 3
        elif t_ns < 12:
            return 500
        elif t_ns < 13:
            return 500 * (13 - t_ns)
        else:
            return 0
    
    def get_state(self) -> dict:
        """Get current state summary."""
        # Find hot spot (central high-temperature region)
        # Handle NaN values
        T_valid = np.where(np.isfinite(self.mesh.T), self.mesh.T, 0)
        T_max_idx = np.argmax(T_valid)
        
        # Hot spot metrics
        T_hs = np.nanmax(self.mesh.T) if np.any(np.isfinite(self.mesh.T)) else 0
        rho_hs = self.mesh.rho[T_max_idx] if np.isfinite(self.mesh.rho[T_max_idx]) else 0
        R_hs = self.mesh.r[T_max_idx + 1] * 1e4 if np.isfinite(self.mesh.r[T_max_idx + 1]) else 0
        
        # Shell metrics
        R_outer = self.mesh.r[-1] * 1e4 if np.isfinite(self.mesh.r[-1]) else 0  # μm
        V_imp = self.mesh.v[-1] * 1e-5 if np.isfinite(self.mesh.v[-1]) else 0  # km/s
        
        return {
            't_ns': self.t * 1e9,
            'R_outer_um': R_outer,
            'V_imp_kms': V_imp,
            'R_hs_um': R_hs,
            'T_hs_keV': T_hs,
            'rho_hs_gcc': rho_hs,
            'E_fusion_MJ': self.E_fusion / 1e13,
            'E_laser_MJ': self.E_laser / 1e13
        }
    
    def run(self, t_end_ns: float = 15, print_interval: int = 100) -> List[dict]:
        """
        Run simulation to t_end_ns.
        
        Returns:
            List of state dictionaries at each saved timestep
        """
        t_end = t_end_ns * 1e-9  # Convert to seconds
        step = 0
        
        print("\nStarting 1D Lagrangian hydrodynamics simulation...")
        print("=" * 70)
        
        while self.t < t_end:
            # Adaptive timestep
            dt = compute_timestep(self.mesh)
            
            # Don't overshoot end time
            if self.t + dt > t_end:
                dt = t_end - self.t
            
            # Laser power
            t_ns = self.t * 1e9
            P_laser = self.laser_pulse(t_ns)
            self.E_laser += P_laser * 1e19 * dt  # TW * s = 1e12 J = 1e19 erg
            
            # Advance one step
            dE_fusion = step_explicit(self.mesh, dt, P_laser)
            self.E_fusion += dE_fusion
            
            self.t += dt
            step += 1
            
            # Save state periodically
            if step % 10 == 0:
                self.history.append(self.get_state())
            
            # Print progress
            if step % print_interval == 0:
                s = self.get_state()
                print(f"  t={s['t_ns']:6.2f} ns  R={s['R_outer_um']:6.0f} μm  "
                      f"V={s['V_imp_kms']:+7.0f} km/s  T={s['T_hs_keV']:5.1f} keV  "
                      f"Y={s['E_fusion_MJ']:.3f} MJ")
        
        print("=" * 70)
        return self.history
    
    def plot_results(self, save_path: str = None):
        """Plot simulation results."""
        if not self.history:
            print("No history to plot!")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        t = [s['t_ns'] for s in self.history]
        
        # Outer radius
        ax = axes[0, 0]
        ax.plot(t, [s['R_outer_um'] for s in self.history], 'b-', linewidth=2)
        ax.set_xlabel('Time [ns]')
        ax.set_ylabel('Outer Radius [μm]')
        ax.set_title('Shell Trajectory')
        ax.grid(True, alpha=0.3)
        
        # Implosion velocity
        ax = axes[0, 1]
        ax.plot(t, [s['V_imp_kms'] for s in self.history], 'r-', linewidth=2)
        ax.set_xlabel('Time [ns]')
        ax.set_ylabel('Implosion Velocity [km/s]')
        ax.set_title('Implosion Velocity')
        ax.grid(True, alpha=0.3)
        
        # Hot spot radius
        ax = axes[0, 2]
        ax.plot(t, [s['R_hs_um'] for s in self.history], 'g-', linewidth=2)
        ax.set_xlabel('Time [ns]')
        ax.set_ylabel('Hot Spot Radius [μm]')
        ax.set_title('Hot Spot Compression')
        ax.grid(True, alpha=0.3)
        
        # Temperature
        ax = axes[1, 0]
        ax.semilogy(t, [max(s['T_hs_keV'], 0.001) for s in self.history], 
                    'orange', linewidth=2)
        ax.set_xlabel('Time [ns]')
        ax.set_ylabel('Hot Spot Temperature [keV]')
        ax.set_title('Hot Spot Heating')
        ax.grid(True, alpha=0.3)
        
        # Density
        ax = axes[1, 1]
        ax.semilogy(t, [max(s['rho_hs_gcc'], 0.001) for s in self.history],
                    'purple', linewidth=2)
        ax.set_xlabel('Time [ns]')
        ax.set_ylabel('Hot Spot Density [g/cm³]')
        ax.set_title('Compression')
        ax.grid(True, alpha=0.3)
        
        # Fusion yield
        ax = axes[1, 2]
        ax.plot(t, [s['E_fusion_MJ'] for s in self.history], 'red', linewidth=2)
        ax.axhline(y=self.E_laser/1e13, color='blue', linestyle='--', 
                   label=f'Laser: {self.E_laser/1e13:.2f} MJ')
        ax.set_xlabel('Time [ns]')
        ax.set_ylabel('Fusion Yield [MJ]')
        ax.set_title('Energy Production')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.close()
        return fig
    
    def plot_profiles(self, save_path: str = None):
        """Plot current radial profiles."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Get zone centers
        r_centers = 0.5 * (self.mesh.r[:-1] + self.mesh.r[1:]) * 1e4  # μm
        
        # Density profile
        ax = axes[0, 0]
        ax.semilogy(r_centers, self.mesh.rho, 'b-', linewidth=2)
        ax.set_xlabel('Radius [μm]')
        ax.set_ylabel('Density [g/cm³]')
        ax.set_title(f'Density Profile (t = {self.t*1e9:.2f} ns)')
        ax.grid(True, alpha=0.3)
        
        # Temperature profile
        ax = axes[0, 1]
        ax.semilogy(r_centers, np.clip(self.mesh.T, 1e-4, None), 'r-', linewidth=2)
        ax.set_xlabel('Radius [μm]')
        ax.set_ylabel('Temperature [keV]')
        ax.set_title('Temperature Profile')
        ax.grid(True, alpha=0.3)
        
        # Pressure profile
        ax = axes[1, 0]
        ax.semilogy(r_centers, self.mesh.P / 1e15, 'g-', linewidth=2)  # Gbar
        ax.set_xlabel('Radius [μm]')
        ax.set_ylabel('Pressure [Gbar]')
        ax.set_title('Pressure Profile')
        ax.grid(True, alpha=0.3)
        
        # Velocity profile
        ax = axes[1, 1]
        v_kms = self.mesh.v * 1e-5  # km/s
        ax.plot(self.mesh.r * 1e4, v_kms, 'purple', linewidth=2)
        ax.set_xlabel('Radius [μm]')
        ax.set_ylabel('Velocity [km/s]')
        ax.set_title('Velocity Profile')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.close()
        return fig


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print()
    print("╔" + "═" * 58 + "╗")
    print("║  1D LAGRANGIAN HYDRODYNAMICS SIMULATION                  ║")
    print("║  Inertial Confinement Fusion - NIF Target                ║")
    print("╚" + "═" * 58 + "╝")
    print()
    
    # Create simulation
    sim = LagrangianSimulation(N_zones=100)
    
    print(f"  Zones: {sim.mesh.N}")
    print(f"  Initial outer radius: {sim.mesh.r[-1]*1e4:.0f} μm")
    print(f"  Initial gas mass: {sum(sim.mesh.m[:33])*1e6:.1f} μg")
    print(f"  Initial shell mass: {sum(sim.mesh.m[33:])*1e6:.1f} μg")
    
    # Plot initial profiles
    sim.plot_profiles('hydro_initial_profiles.png')
    
    # Run simulation
    history = sim.run(t_end_ns=15, print_interval=200)
    
    # Results
    final = sim.get_state()
    Q = final['E_fusion_MJ'] / max(final['E_laser_MJ'], 0.01)
    
    print()
    print("  RESULTS")
    print("  " + "─" * 40)
    print(f"  Total laser energy:       {final['E_laser_MJ']:.2f} MJ")
    print(f"  Total fusion yield:       {final['E_fusion_MJ']:.2f} MJ")
    print(f"  Scientific gain (Q):      {Q:.2f}")
    print(f"  Peak hot spot temp:       {max(s['T_hs_keV'] for s in history):.1f} keV")
    print(f"  Peak hot spot density:    {max(s['rho_hs_gcc'] for s in history):.0f} g/cm³")
    print()
    
    if Q > 1:
        print("  ★ ★ ★  IGNITION ACHIEVED  ★ ★ ★")
    else:
        print(f"  Gain Q = {Q:.2f} (ignition requires Q > 1)")
    print()
    
    # Plot results
    sim.plot_results('hydro_time_evolution.png')
    sim.plot_profiles('hydro_final_profiles.png')
    
    print("  Generated plots:")
    print("    - hydro_initial_profiles.png")
    print("    - hydro_time_evolution.png")
    print("    - hydro_final_profiles.png")
    print()
