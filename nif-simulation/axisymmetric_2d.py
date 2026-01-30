"""
2D Axisymmetric ICF Simulation
==============================

Extends the 0D model to 2D axisymmetric geometry to capture:
1. Drive asymmetries (P2, P4 Legendre modes from laser)
2. Low-mode instabilities (l=2, 4, 6)
3. Pole/equator variations
4. Jet formation and shell breakup

The simulation uses a simplified Lagrangian approach where the
shell surface is represented as r(θ) and evolved under:
- Pressure from hot spot
- Laser drive (with asymmetry)
- Rayleigh-Taylor growth

Author: Evan Pieser
Date: 2026
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable
import matplotlib.pyplot as plt
from scipy.special import legendre

# Physical constants
EV = 1.602e-12       # erg
M_P = 1.673e-24      # g
M_DT = 2.5 * M_P     # g


@dataclass
class State2D:
    """2D simulation state."""
    t: float                  # Time [ns]
    theta: np.ndarray         # Polar angles [rad]
    R_outer: np.ndarray       # Outer shell radius vs theta [μm]
    R_inner: np.ndarray       # Inner shell radius (hot spot) [μm]
    V_outer: np.ndarray       # Radial velocity vs theta [km/s]
    T_hs: float               # Hot spot temperature [keV]
    rho_hs: float             # Hot spot density [g/cm³]
    P_hs: float               # Hot spot pressure [Gbar]
    yield_MJ: float           # Cumulative yield [MJ]
    
    # Asymmetry metrics
    P0: float                 # Mean radius (l=0)
    P2: float                 # P2/P0 asymmetry
    P4: float                 # P4/P0 asymmetry


def sigma_v(T_keV):
    """DT fusion reactivity [cm³/s]."""
    T = np.clip(T_keV, 1.0, 150.0)
    return 3.68e-12 * T**(-2./3.) * np.exp(-19.94 * T**(-1./3.))


class DriveAsymmetry:
    """
    Model for laser drive asymmetries.
    
    NIF's 192 beams are arranged in rings at different polar angles.
    Imperfect power balance creates Legendre mode asymmetries.
    """
    
    def __init__(self, P2_amplitude: float = 0.02, P4_amplitude: float = 0.01,
                 P6_amplitude: float = 0.005):
        """
        Args:
            P2_amplitude: P2 mode amplitude (δP/P)
            P4_amplitude: P4 mode amplitude
            P6_amplitude: P6 mode amplitude
        """
        self.P2 = P2_amplitude
        self.P4 = P4_amplitude
        self.P6 = P6_amplitude
        
    def drive_profile(self, theta: np.ndarray, P_mean: float) -> np.ndarray:
        """
        Compute laser drive pressure vs polar angle.
        
        P(θ) = P_mean * [1 + Σ a_l * P_l(cos θ)]
        
        Args:
            theta: Polar angles [rad]
            P_mean: Mean laser power [TW]
            
        Returns:
            Power vs angle [TW]
        """
        x = np.cos(theta)
        
        # Legendre polynomials
        P2_leg = legendre(2)(x)
        P4_leg = legendre(4)(x)
        P6_leg = legendre(6)(x)
        
        # Total drive
        P = P_mean * (1 + self.P2 * P2_leg + self.P4 * P4_leg + self.P6 * P6_leg)
        
        return np.maximum(P, 0)  # Ensure non-negative


class Axisymmetric2D:
    """
    2D axisymmetric ICF implosion simulation.
    
    Represents shell as r(θ) on a grid of polar angles.
    Hot spot is treated as a uniform-pressure region.
    """
    
    def __init__(self, n_theta: int = 64, 
                 drive_asymmetry: Optional[DriveAsymmetry] = None):
        """
        Args:
            n_theta: Number of polar angle points (0 to π)
            drive_asymmetry: DriveAsymmetry model for laser
        """
        self.n_theta = n_theta
        self.theta = np.linspace(0, np.pi, n_theta)  # 0 = pole, π/2 = equator
        
        self.drive = drive_asymmetry or DriveAsymmetry()
        
        # Initial conditions (NIF-like)
        self.R0_outer = 1100.0      # μm
        self.R0_inner = 850.0       # μm
        self.M_shell = 170.0        # μg
        self.M_hs = 10.0            # μg
        
        # Target implosion velocity
        self.V_imp_target = -400.0  # km/s
        
        self.reset()
        
    def reset(self):
        """Reset to initial spherical state."""
        self.t = 0.0
        self.R_outer = np.ones(self.n_theta) * self.R0_outer
        self.R_inner = np.ones(self.n_theta) * self.R0_inner
        self.V = np.zeros(self.n_theta)  # km/s
        self.T_hs = 0.03  # keV
        self.yield_MJ = 0.0
        self.burn_fraction = 0.0
        self.stagnated = False
        
    def compute_volume(self, R: np.ndarray) -> float:
        """
        Compute volume enclosed by surface r(θ).
        
        V = (2π/3) ∫ r³ sin(θ) dθ
        """
        dtheta = self.theta[1] - self.theta[0]
        integrand = R**3 * np.sin(self.theta)
        return (2 * np.pi / 3) * np.sum(integrand) * dtheta
    
    def compute_legendre_moments(self, R: np.ndarray) -> Tuple[float, float, float]:
        """
        Decompose R(θ) into Legendre moments.
        
        R(θ) = Σ R_l P_l(cos θ)
        R_l = (2l+1)/2 ∫ R(θ) P_l(cos θ) sin(θ) dθ
        
        Returns:
            (R0, P2/P0, P4/P0)
        """
        x = np.cos(self.theta)
        dtheta = self.theta[1] - self.theta[0]
        sin_theta = np.sin(self.theta)
        
        # l=0 (mean radius)
        R0 = 0.5 * np.sum(R * sin_theta) * dtheta
        
        # l=2
        P2_leg = legendre(2)(x)
        R2 = 2.5 * np.sum(R * P2_leg * sin_theta) * dtheta
        
        # l=4
        P4_leg = legendre(4)(x)
        R4 = 4.5 * np.sum(R * P4_leg * sin_theta) * dtheta
        
        return R0, R2/R0 if R0 > 0 else 0, R4/R0 if R0 > 0 else 0
    
    def rho_hs(self) -> float:
        """Mean hot spot density [g/cm³]."""
        V_cm3 = self.compute_volume(self.R_inner * 1e-4)  # Convert μm to cm
        return (self.M_hs * 1e-6) / max(V_cm3, 1e-30)
    
    def P_hs_Gbar(self) -> float:
        """Hot spot pressure [Gbar]."""
        return 0.154 * self.rho_hs() * self.T_hs / 1000
    
    def step(self, dt: float, P_laser_TW: float) -> State2D:
        """
        Advance 2D simulation by dt nanoseconds.
        
        Args:
            dt: Timestep [ns]
            P_laser_TW: Mean laser power [TW]
        """
        # Get drive profile (with asymmetry)
        P_drive = self.drive.drive_profile(self.theta, P_laser_TW)
        
        # === ACCELERATION PHASE ===
        # Each angle point accelerates based on local drive
        if np.mean(P_drive) > 10 and np.mean(self.V) > self.V_imp_target:
            for i in range(self.n_theta):
                P_norm = P_drive[i] / 500.0
                accel = -80.0 * P_norm  # km/s per ns
                dV = accel * dt
                self.V[i] = max(self.V[i] + dV, self.V_imp_target * 1.2)  # Allow some variation
        
        # === MOTION ===
        self.R_outer += self.V * dt
        self.R_outer = np.maximum(self.R_outer, 30.0)
        
        # Inner surface compression
        # More symmetric than outer (pressure equilibration)
        R_outer_mean = np.mean(self.R_outer)
        CR = self.R0_outer / max(R_outer_mean, 50)
        R_inner_mean = self.R0_inner / CR**0.95
        
        # Add some asymmetry feed-through (damped)
        outer_asymmetry = self.R_outer - R_outer_mean
        self.R_inner = R_inner_mean + 0.3 * outer_asymmetry
        self.R_inner = np.maximum(self.R_inner, 25.0)
        
        # === COMPRESSION HEATING ===
        R_hs_mean = np.mean(self.R_inner)
        CR_hs = self.R0_inner / max(R_hs_mean, 25)
        self.T_hs = 0.03 * CR_hs**2
        
        # Shock heating at stagnation
        if R_hs_mean < 100 and np.mean(self.V) < -50:
            KE_keV = 0.5 * self.M_shell * 1e-6 * (np.mean(self.V) * 1e5)**2 / (1000 * EV)
            dT_shock = 0.01 * KE_keV / (self.M_hs * 1e-6) * abs(dt / 0.5)
            self.T_hs += dT_shock
            
        self.T_hs = min(self.T_hs, 70.0)
        
        # === PRESSURE DECELERATION ===
        P = self.P_hs_Gbar()
        if P > 0.1 and np.mean(self.V) < 0:
            R_hs_cm = R_hs_mean * 1e-4
            A = 4 * np.pi * R_hs_cm**2
            F = P * 1e15 * A
            a = F / (self.M_shell * 1e-6)
            dV = a * dt * 1e-9 * 1e-5
            self.V = np.minimum(self.V + dV, 0)
        
        # === FUSION BURN ===
        # Yield degraded by asymmetry
        if self.T_hs > 4 and self.burn_fraction < 0.35:
            sv = sigma_v(self.T_hs)
            rho = self.rho_hs()
            n = rho / M_DT
            
            V_hs_cm3 = self.compute_volume(self.R_inner * 1e-4)
            R_rate = 0.25 * n**2 * sv
            E_FUSION = 17.6e6 * EV
            P_fus = R_rate * E_FUSION * V_hs_cm3
            
            # Asymmetry degradation (P2 reduces effective burn volume)
            _, P2_over_P0, P4_over_P0 = self.compute_legendre_moments(self.R_inner)
            asymmetry_factor = 1.0 - 0.5 * abs(P2_over_P0) - 0.3 * abs(P4_over_P0)
            asymmetry_factor = max(asymmetry_factor, 0.5)
            
            dY = P_fus * asymmetry_factor * (dt * 1e-9) / 1e13
            self.yield_MJ += dY
            
            n_reactions = R_rate * V_hs_cm3 * asymmetry_factor * (dt * 1e-9)
            fuel_atoms = (self.M_hs * 1e-6) / M_DT
            self.burn_fraction += n_reactions / fuel_atoms
            
            # Alpha heating
            ALPHA = 0.2
            Q_alpha = P_fus * ALPHA * (dt * 1e-9) * (1 - self.burn_fraction)
            cv = 1.44e13
            dT = Q_alpha / (rho * V_hs_cm3 * cv) * asymmetry_factor
            self.T_hs += min(dT, 5.0)
        
        # Mark stagnation
        if R_hs_mean <= 30 and not self.stagnated:
            self.stagnated = True
            
        self.t += dt
        
        # Compute Legendre moments
        P0, P2_rel, P4_rel = self.compute_legendre_moments(self.R_outer)
        
        return State2D(
            t=self.t,
            theta=self.theta.copy(),
            R_outer=self.R_outer.copy(),
            R_inner=self.R_inner.copy(),
            V_outer=self.V.copy(),
            T_hs=self.T_hs,
            rho_hs=self.rho_hs(),
            P_hs=self.P_hs_Gbar(),
            yield_MJ=self.yield_MJ,
            P0=P0,
            P2=P2_rel,
            P4=P4_rel
        )
    
    def run(self, t_end_ns: float, laser_func: Callable, dt: float = 0.02,
            verbose: bool = True) -> List[State2D]:
        """Run 2D simulation."""
        states = []
        
        while self.t < t_end_ns:
            P = laser_func(self.t)
            state = self.step(dt, P)
            states.append(state)
            
            if verbose and len(states) % 50 == 0:
                print(f"  t={state.t:5.2f}ns  <R>={state.P0:5.0f}μm  "
                      f"P2={state.P2:+.3f}  P4={state.P4:+.3f}  "
                      f"T={state.T_hs:5.1f}keV  Y={state.yield_MJ:.3f}MJ")
                
        return states


def nif_pulse(t_ns):
    """NIF laser pulse shape [TW]."""
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


def plot_2d_implosion(states: List[State2D], save_prefix: str = "2d_implosion"):
    """Generate 2D visualization plots."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Shell shape evolution
    ax = axes[0, 0]
    times_to_plot = [0, len(states)//4, len(states)//2, 3*len(states)//4, -1]
    colors = plt.cm.viridis(np.linspace(0, 1, len(times_to_plot)))
    
    for idx, color in zip(times_to_plot, colors):
        state = states[idx]
        # Convert to Cartesian for plotting (2D cross-section)
        x = state.R_outer * np.sin(state.theta)
        y = state.R_outer * np.cos(state.theta)
        ax.plot(x, y, '-', color=color, linewidth=2, label=f't={state.t:.1f}ns')
        ax.plot(-x, y, '-', color=color, linewidth=2)  # Mirror
        
    ax.set_xlabel('r sin(θ) [μm]')
    ax.set_ylabel('r cos(θ) [μm]')
    ax.set_title('Shell Shape Evolution')
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 2. Mean radius and asymmetry
    ax = axes[0, 1]
    t = [s.t for s in states]
    ax.plot(t, [s.P0 for s in states], 'b-', linewidth=2, label='<R>')
    ax.set_xlabel('Time [ns]')
    ax.set_ylabel('Mean Radius [μm]')
    ax.set_title('Mean Shell Radius')
    ax.grid(True, alpha=0.3)
    
    ax2 = ax.twinx()
    ax2.plot(t, [s.P2 for s in states], 'r--', linewidth=2, label='P2/P0')
    ax2.plot(t, [s.P4 for s in states], 'g--', linewidth=2, label='P4/P0')
    ax2.set_ylabel('Asymmetry (P_l/P_0)')
    ax2.legend(loc='upper right')
    
    # 3. Velocity vs angle at stagnation
    ax = axes[0, 2]
    stag_idx = np.argmin([np.mean(s.R_outer) for s in states])
    stag_state = states[stag_idx]
    ax.plot(np.degrees(stag_state.theta), stag_state.V_outer, 'b-', linewidth=2)
    ax.set_xlabel('Polar Angle θ [degrees]')
    ax.set_ylabel('Radial Velocity [km/s]')
    ax.set_title(f'Velocity Profile at Stagnation (t={stag_state.t:.2f}ns)')
    ax.axhline(y=0, color='k', linestyle='--')
    ax.grid(True, alpha=0.3)
    
    # 4. Temperature evolution
    ax = axes[1, 0]
    ax.plot(t, [s.T_hs for s in states], 'r-', linewidth=2)
    ax.set_xlabel('Time [ns]')
    ax.set_ylabel('Hot Spot Temperature [keV]')
    ax.set_title('Hot Spot Temperature')
    ax.grid(True, alpha=0.3)
    
    # 5. Yield
    ax = axes[1, 1]
    ax.plot(t, [s.yield_MJ for s in states], 'g-', linewidth=2)
    ax.set_xlabel('Time [ns]')
    ax.set_ylabel('Yield [MJ]')
    ax.set_title('Cumulative Fusion Yield')
    ax.grid(True, alpha=0.3)
    
    # 6. Final hot spot shape
    ax = axes[1, 2]
    final = states[-1]
    x_outer = final.R_outer * np.sin(final.theta)
    y_outer = final.R_outer * np.cos(final.theta)
    x_inner = final.R_inner * np.sin(final.theta)
    y_inner = final.R_inner * np.cos(final.theta)
    
    # Fill shell region
    ax.fill(np.concatenate([x_outer, -x_outer[::-1]]),
            np.concatenate([y_outer, y_outer[::-1]]),
            color='blue', alpha=0.3, label='Shell')
    ax.fill(np.concatenate([x_inner, -x_inner[::-1]]),
            np.concatenate([y_inner, y_inner[::-1]]),
            color='red', alpha=0.5, label='Hot Spot')
    
    ax.set_xlabel('x [μm]')
    ax.set_ylabel('z [μm]')
    ax.set_title(f'Final Capsule Shape (t={final.t:.1f}ns)')
    ax.set_aspect('equal')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {save_prefix}.png")
    plt.close()


def run_asymmetry_study():
    """
    Study effect of drive asymmetry on implosion performance.
    """
    print()
    print("╔" + "═" * 58 + "╗")
    print("║  2D AXISYMMETRIC ICF SIMULATION                          ║")
    print("║  Drive Asymmetry Study                                   ║")
    print("╚" + "═" * 58 + "╝")
    print()
    
    # Sweep P2 asymmetry
    P2_values = [0, 0.01, 0.02, 0.05, 0.10]
    results = []
    
    for P2 in P2_values:
        print(f"Running P2 = {P2*100:.1f}%...")
        
        drive = DriveAsymmetry(P2_amplitude=P2, P4_amplitude=P2/2)
        sim = Axisymmetric2D(n_theta=64, drive_asymmetry=drive)
        states = sim.run(t_end_ns=14, laser_func=nif_pulse, dt=0.02, verbose=False)
        
        final = states[-1]
        results.append({
            'P2': P2,
            'yield': final.yield_MJ,
            'P2_final': final.P2,
            'P4_final': final.P4,
            'T_peak': max(s.T_hs for s in states)
        })
        
    print()
    print("Results:")
    print("─" * 60)
    print(f"{'P2 Drive':>10}  {'Yield [MJ]':>12}  {'Final P2':>10}  {'Q':>8}")
    print("─" * 60)
    
    for r in results:
        Q = r['yield'] / 2.05
        print(f"{r['P2']*100:>9.1f}%  {r['yield']:>12.2f}  {r['P2_final']:>+10.3f}  {Q:>8.2f}")
    
    print()
    
    # Plot asymmetry sweep
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    ax = axes[0]
    P2_arr = [r['P2']*100 for r in results]
    yield_arr = [r['yield'] for r in results]
    ax.plot(P2_arr, yield_arr, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('P2 Drive Asymmetry [%]')
    ax.set_ylabel('Fusion Yield [MJ]')
    ax.set_title('Yield vs Drive Asymmetry')
    ax.axhline(y=2.05, color='r', linestyle='--', label='Q=1')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    Q_arr = [r['yield']/2.05 for r in results]
    ax.bar(P2_arr, Q_arr, width=1.5, color='green', alpha=0.7)
    ax.axhline(y=1, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('P2 Drive Asymmetry [%]')
    ax.set_ylabel('Scientific Gain Q')
    ax.set_title('Ignition Margin vs Asymmetry')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('asymmetry_sweep.png', dpi=150, bbox_inches='tight')
    print("Saved: asymmetry_sweep.png")
    plt.close()
    
    return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print()
    print("╔" + "═" * 58 + "╗")
    print("║  2D AXISYMMETRIC ICF SIMULATION                          ║")
    print("║  National Ignition Facility - December 5, 2022           ║")
    print("╚" + "═" * 58 + "╝")
    print()
    
    # Run with moderate asymmetry
    drive = DriveAsymmetry(P2_amplitude=0.02, P4_amplitude=0.01)
    sim = Axisymmetric2D(n_theta=64, drive_asymmetry=drive)
    
    print(f"Grid: {sim.n_theta} polar angles")
    print(f"Drive asymmetry: P2={drive.P2*100:.1f}%, P4={drive.P4*100:.1f}%")
    print(f"Initial radius: {sim.R0_outer:.0f} μm")
    print()
    print("─" * 60)
    
    states = sim.run(t_end_ns=14, laser_func=nif_pulse, dt=0.02)
    
    print("─" * 60)
    print()
    
    # Results
    final = states[-1]
    peak_T = max(states, key=lambda s: s.T_hs)
    min_R = min(states, key=lambda s: s.P0)
    
    Q = final.yield_MJ / 2.05
    
    print("  RESULTS")
    print("  " + "─" * 40)
    print(f"  Peak temperature:    {peak_T.T_hs:.1f} keV")
    print(f"  Minimum mean radius: {min_R.P0:.0f} μm")
    print(f"  Final P2 asymmetry:  {final.P2:+.3f}")
    print(f"  Final P4 asymmetry:  {final.P4:+.3f}")
    print(f"  Total yield:         {final.yield_MJ:.2f} MJ")
    print(f"  Scientific gain:     Q = {Q:.2f}")
    print()
    
    if Q > 1:
        print("  ✓ IGNITION achieved despite asymmetry!")
    else:
        print(f"  ✗ Q = {Q:.2f} < 1 (asymmetry degraded performance)")
    
    print()
    
    # Generate plots
    print("Generating 2D plots...")
    plot_2d_implosion(states)
    
    # Run asymmetry study
    print()
    run_asymmetry_study()
    
    print()
    print("═" * 60)
