"""
Rayleigh-Taylor Instabilities in ICF Implosions
================================================

The Rayleigh-Taylor (RT) instability occurs when a heavy fluid is
accelerated into a light fluid, or equivalently, when a light fluid
supports a heavy fluid against gravity.

In ICF implosions, RT instabilities occur at:
1. Ablation front (during acceleration phase)
2. Hot spot / shell interface (during deceleration)

These instabilities can:
- Mix cold fuel into the hot spot, reducing temperature
- Break up the shell, reducing compression
- Create jets that penetrate the hot spot

Key physics:
- Classical RT growth rate: γ = sqrt(k * g * A)
  where A = (ρ_heavy - ρ_light)/(ρ_heavy + ρ_light) is Atwood number
- Ablation stabilization: γ_abl = sqrt(k*g) - β*k*V_a
  where V_a is ablation velocity
- Finite shell thickness limits maximum mode number

Author: Evan Pieser
Date: 2026
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt


# Physical constants
PI = np.pi


@dataclass
class RTMode:
    """Single Rayleigh-Taylor mode."""
    l: int            # Mode number (spherical harmonic degree)
    amplitude: float  # Perturbation amplitude [μm]
    phase: float      # Phase [radians]
    growth_rate: float  # Linear growth rate [1/ns]


@dataclass
class RTState:
    """State of RT instabilities at a given time."""
    t: float                    # Time [ns]
    R: float                    # Shell radius [μm]
    total_amplitude: float      # RMS amplitude [μm]
    max_amplitude: float        # Maximum amplitude [μm]
    mix_width: float           # Mix region width [μm]
    mix_fraction: float        # Fraction of shell mixed
    modes: List[RTMode]        # Individual modes


def classical_rt_growth_rate(k: float, g: float, A: float) -> float:
    """
    Classical Rayleigh-Taylor growth rate.
    
    γ = sqrt(k * g * A)
    
    Args:
        k: Wavenumber [1/cm]
        g: Acceleration [cm/s²]
        A: Atwood number (ρ₂-ρ₁)/(ρ₂+ρ₁)
    
    Returns:
        Growth rate [1/s]
    """
    if k <= 0 or g <= 0 or A <= 0:
        return 0.0
    return np.sqrt(k * g * A)


def ablation_stabilized_growth_rate(k: float, g: float, A: float,
                                     V_abl: float, beta: float = 1.5) -> float:
    """
    Ablation-stabilized RT growth rate (Takabe formula).
    
    γ = α * sqrt(k*g) - β * k * V_a
    
    where α ≈ 0.9 and β ≈ 1-3 depending on conditions.
    
    Ablation carries away perturbed material, reducing growth.
    Short wavelengths (high k) are more strongly stabilized.
    
    Args:
        k: Wavenumber [1/cm]
        g: Acceleration [cm/s²]
        A: Atwood number
        V_abl: Ablation velocity [cm/s]
        beta: Ablation stabilization coefficient (1-3)
    
    Returns:
        Growth rate [1/s]
    """
    alpha = 0.9
    gamma_classical = alpha * np.sqrt(k * g)
    gamma_abl = beta * k * V_abl
    
    gamma = max(gamma_classical - gamma_abl, 0.0)
    return gamma


def spherical_mode_to_wavenumber(l: int, R: float) -> float:
    """
    Convert spherical harmonic mode number to wavenumber.
    
    k ≈ l / R for spherical geometry
    
    Args:
        l: Mode number (spherical harmonic degree)
        R: Radius [cm]
    
    Returns:
        Wavenumber [1/cm]
    """
    if R <= 0:
        return 0.0
    return l / R


def cutoff_mode_number(R: float, delta_R: float) -> int:
    """
    Maximum unstable mode number limited by shell thickness.
    
    Modes with wavelength < shell thickness are stabilized
    because they can't "fit" in the shell.
    
    l_max ≈ R / δR
    
    Args:
        R: Radius [μm]
        delta_R: Shell thickness [μm]
    
    Returns:
        Maximum mode number
    """
    if delta_R <= 0:
        return 1000  # No limit
    return max(int(R / delta_R), 2)


def initial_perturbation_spectrum(l_max: int, amplitude_rms: float,
                                   spectrum_type: str = 'NIF') -> List[RTMode]:
    """
    Generate initial perturbation spectrum.
    
    Real capsules have surface roughness from:
    - Manufacturing defects
    - Fill tube
    - Mounting features
    - Ice layer non-uniformity
    
    Args:
        l_max: Maximum mode number
        amplitude_rms: RMS surface roughness [μm]
        spectrum_type: 'flat', 'NIF', or 'power_law'
    
    Returns:
        List of RTMode objects
    """
    modes = []
    
    # Mode numbers to include
    l_values = list(range(2, min(l_max, 200) + 1))
    
    for l in l_values:
        # Amplitude spectrum
        if spectrum_type == 'flat':
            # Equal power per mode
            amp = amplitude_rms / np.sqrt(len(l_values))
            
        elif spectrum_type == 'NIF':
            # NIF capsules have ~1 μm RMS roughness
            # Power spectrum peaks around l~10-20
            # Approximate: a_l ∝ l^(-1) for l > 10
            if l < 10:
                amp = amplitude_rms * 0.1 / np.sqrt(len(l_values))
            else:
                amp = amplitude_rms * (10/l) / np.sqrt(len(l_values))
                
        elif spectrum_type == 'power_law':
            # Power law: a_l ∝ l^(-1)
            amp = amplitude_rms / (l * np.sqrt(sum(1/ll**2 for ll in l_values)))
            
        else:
            amp = amplitude_rms / np.sqrt(len(l_values))
        
        # Random phase
        phase = np.random.uniform(0, 2*PI)
        
        modes.append(RTMode(
            l=l,
            amplitude=amp,
            phase=phase,
            growth_rate=0.0  # Will be updated during evolution
        ))
    
    return modes


class RTInstabilityModel:
    """
    Model for Rayleigh-Taylor instability evolution during implosion.
    
    Tracks perturbation growth at both ablation front (acceleration)
    and hot spot interface (deceleration).
    """
    
    def __init__(self, initial_roughness_um: float = 1.0,
                 ablation_front: bool = True,
                 hot_spot_interface: bool = True):
        """
        Args:
            initial_roughness_um: Initial RMS surface roughness [μm]
            ablation_front: Include ablation front instabilities
            hot_spot_interface: Include hot spot interface instabilities
        """
        self.initial_roughness = initial_roughness_um
        self.include_ablation = ablation_front
        self.include_hot_spot = hot_spot_interface
        
        # Initialize modes
        self.ablation_modes = initial_perturbation_spectrum(
            100, initial_roughness_um, 'NIF'
        )
        self.interface_modes = initial_perturbation_spectrum(
            100, initial_roughness_um * 0.5, 'NIF'  # Inner surface smoother
        )
        
        self.history = []
    
    def compute_growth_rates(self, R_um: float, V_kms: float, 
                              a_kms2: float, rho_shell: float,
                              rho_ablator: float, rho_hot_spot: float,
                              V_abl_kms: float, shell_thickness_um: float):
        """
        Update growth rates for all modes given current conditions.
        
        Args:
            R_um: Shell radius [μm]
            V_kms: Implosion velocity [km/s]
            a_kms2: Acceleration [km/s²]  (negative for implosion)
            rho_shell: Shell density [g/cm³]
            rho_ablator: Ablator density [g/cm³]
            rho_hot_spot: Hot spot density [g/cm³]
            V_abl_kms: Ablation velocity [km/s]
            shell_thickness_um: Shell thickness [μm]
        """
        R_cm = R_um * 1e-4
        a_cms2 = abs(a_kms2) * 1e10  # km/s² to cm/s²
        V_abl_cms = V_abl_kms * 1e5
        
        # Cutoff mode
        l_max = cutoff_mode_number(R_um, shell_thickness_um)
        
        # Ablation front (during acceleration, a < 0 means shell being pushed)
        if self.include_ablation and a_kms2 < 0:
            A_abl = (rho_shell - rho_ablator) / (rho_shell + rho_ablator + 1e-10)
            A_abl = max(A_abl, 0)  # Must be positive for instability
            
            for mode in self.ablation_modes:
                if mode.l > l_max:
                    mode.growth_rate = 0.0
                    continue
                    
                k = spherical_mode_to_wavenumber(mode.l, R_cm)
                gamma = ablation_stabilized_growth_rate(k, a_cms2, A_abl, V_abl_cms)
                mode.growth_rate = gamma * 1e-9  # Convert to 1/ns
        
        # Hot spot interface (during deceleration)
        if self.include_hot_spot and a_kms2 > 0:
            A_hs = (rho_shell - rho_hot_spot) / (rho_shell + rho_hot_spot + 1e-10)
            A_hs = max(A_hs, 0)
            
            for mode in self.interface_modes:
                if mode.l > l_max:
                    mode.growth_rate = 0.0
                    continue
                    
                k = spherical_mode_to_wavenumber(mode.l, R_cm)
                # No ablation stabilization at hot spot interface
                gamma = classical_rt_growth_rate(k, a_cms2, A_hs)
                mode.growth_rate = gamma * 1e-9  # Convert to 1/ns
    
    def evolve(self, dt_ns: float):
        """
        Evolve perturbation amplitudes by one timestep.
        
        Uses exponential growth in linear regime.
        Saturates when amplitude ~ wavelength (nonlinear).
        """
        for mode in self.ablation_modes + self.interface_modes:
            if mode.growth_rate > 0:
                # Linear growth
                mode.amplitude *= np.exp(mode.growth_rate * dt_ns)
                
                # Nonlinear saturation: amplitude can't exceed wavelength
                # λ = 2πR/l, so a_max ~ R/l
                # We use a_max ~ 0.1 * R / l as typical saturation
                # For l=10, R=500μm: a_max ~ 5 μm
    
    def get_state(self, t_ns: float, R_um: float, 
                   shell_thickness_um: float) -> RTState:
        """
        Get current RT instability state.
        """
        # Compute statistics
        abl_amps = [m.amplitude for m in self.ablation_modes if m.growth_rate > 0]
        hs_amps = [m.amplitude for m in self.interface_modes if m.growth_rate > 0]
        all_amps = abl_amps + hs_amps
        
        if all_amps:
            total_rms = np.sqrt(sum(a**2 for a in all_amps))
            max_amp = max(all_amps)
        else:
            total_rms = 0
            max_amp = 0
        
        # Mix width (where instabilities have penetrated)
        # Approximate as 2 * max amplitude
        mix_width = 2 * max_amp
        
        # Mix fraction (fraction of shell affected)
        mix_fraction = min(mix_width / shell_thickness_um, 1.0) if shell_thickness_um > 0 else 0
        
        return RTState(
            t=t_ns,
            R=R_um,
            total_amplitude=total_rms,
            max_amplitude=max_amp,
            mix_width=mix_width,
            mix_fraction=mix_fraction,
            modes=self.ablation_modes + self.interface_modes
        )
    
    def step(self, t_ns: float, dt_ns: float, R_um: float, V_kms: float,
             a_kms2: float, rho_shell: float = 100, rho_ablator: float = 1,
             rho_hot_spot: float = 10, V_abl_kms: float = 50,
             shell_thickness_um: float = 50) -> RTState:
        """
        Advance RT model by one timestep.
        
        Args:
            t_ns: Current time [ns]
            dt_ns: Timestep [ns]
            R_um: Shell radius [μm]
            V_kms: Implosion velocity [km/s]
            a_kms2: Acceleration [km/s²]
            rho_shell: Shell density [g/cm³]
            rho_ablator: Ablator density [g/cm³]
            rho_hot_spot: Hot spot density [g/cm³]
            V_abl_kms: Ablation velocity [km/s]
            shell_thickness_um: Shell thickness [μm]
        
        Returns:
            RTState after this step
        """
        # Update growth rates
        self.compute_growth_rates(
            R_um, V_kms, a_kms2, rho_shell, rho_ablator,
            rho_hot_spot, V_abl_kms, shell_thickness_um
        )
        
        # Evolve amplitudes
        self.evolve(dt_ns)
        
        # Get state
        state = self.get_state(t_ns, R_um, shell_thickness_um)
        self.history.append(state)
        
        return state
    
    def degradation_factor(self) -> float:
        """
        Estimate yield degradation due to mix.
        
        Mix reduces the effective hot spot volume and temperature.
        Approximate degradation as:
        
        Y_actual / Y_1D ≈ (1 - f_mix)³
        
        where f_mix is the mix fraction.
        
        Returns:
            Yield degradation factor (0 to 1)
        """
        if not self.history:
            return 1.0
        
        # Use peak mix fraction
        max_mix = max(s.mix_fraction for s in self.history)
        
        # Degradation model
        # 3 because yield ~ ρR² * T³ ~ V * T³
        degradation = (1 - max_mix)**3
        
        return max(degradation, 0.0)
    
    def plot_evolution(self, save_path: str = None):
        """Plot RT instability evolution."""
        if not self.history:
            print("No history to plot!")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        t = [s.t for s in self.history]
        
        # Total amplitude
        ax = axes[0, 0]
        ax.semilogy(t, [s.total_amplitude for s in self.history], 'b-', linewidth=2)
        ax.set_xlabel('Time [ns]')
        ax.set_ylabel('RMS Amplitude [μm]')
        ax.set_title('RT Amplitude Growth')
        ax.grid(True, alpha=0.3)
        
        # Max amplitude
        ax = axes[0, 1]
        ax.semilogy(t, [max(s.max_amplitude, 0.01) for s in self.history], 
                    'r-', linewidth=2)
        ax.set_xlabel('Time [ns]')
        ax.set_ylabel('Max Amplitude [μm]')
        ax.set_title('Maximum Perturbation')
        ax.grid(True, alpha=0.3)
        
        # Mix width
        ax = axes[1, 0]
        ax.plot(t, [s.mix_width for s in self.history], 'g-', linewidth=2)
        ax.set_xlabel('Time [ns]')
        ax.set_ylabel('Mix Width [μm]')
        ax.set_title('Mix Region Width')
        ax.grid(True, alpha=0.3)
        
        # Mix fraction
        ax = axes[1, 1]
        ax.plot(t, [s.mix_fraction * 100 for s in self.history], 
                'purple', linewidth=2)
        ax.set_xlabel('Time [ns]')
        ax.set_ylabel('Mix Fraction [%]')
        ax.set_title('Shell Mix Fraction')
        ax.axhline(y=50, color='r', linestyle='--', label='Critical (50%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.close()
        return fig
    
    def plot_mode_spectrum(self, save_path: str = None):
        """Plot mode amplitude spectrum at final time."""
        if not self.history:
            print("No history to plot!")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Get final amplitudes
        final_state = self.history[-1]
        
        l_vals = [m.l for m in final_state.modes]
        amps = [m.amplitude for m in final_state.modes]
        
        ax.semilogy(l_vals, amps, 'bo-', markersize=4, linewidth=1)
        ax.set_xlabel('Mode Number l')
        ax.set_ylabel('Amplitude [μm]')
        ax.set_title('RT Mode Spectrum (Final)')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max(l_vals) + 10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.close()
        return fig


def simulate_rt_with_implosion():
    """
    Simulate RT growth during a NIF-like implosion.
    """
    print()
    print("╔" + "═" * 58 + "╗")
    print("║  RAYLEIGH-TAYLOR INSTABILITY SIMULATION                  ║")
    print("╚" + "═" * 58 + "╝")
    print()
    
    # Initialize RT model
    rt = RTInstabilityModel(
        initial_roughness_um=1.0,  # 1 μm RMS surface roughness
        ablation_front=True,
        hot_spot_interface=True
    )
    
    print(f"Initial surface roughness: {rt.initial_roughness} μm RMS")
    print(f"Number of modes: {len(rt.ablation_modes)}")
    print()
    
    # Simulate implosion trajectory (simplified)
    # Based on NIF parameters
    R0 = 1100       # Initial radius [μm]
    V_final = -400  # Final velocity [km/s]
    t_accel = 10    # Acceleration time [ns]
    t_coast = 2     # Coast time [ns]
    t_stag = 1      # Stagnation time [ns]
    
    dt = 0.05  # Timestep [ns]
    
    print("Simulating implosion with RT growth...")
    print("─" * 50)
    
    t = 0
    R = R0
    V = 0
    
    # Track trajectory
    trajectory = []
    
    while t < t_accel + t_coast + t_stag:
        # Compute acceleration and position
        if t < t_accel:
            # Acceleration phase
            # V goes from 0 to V_final over t_accel
            # a = V_final / t_accel (negative since inward)
            a = V_final / t_accel  # km/s per ns
            V = a * t  # km/s
            # R decreases: dR/dt = V (V is negative)
            # R(t) = R0 + integral(V dt) = R0 + 0.5*a*t² (units: km/s * ns = μm)
            R = R0 + 0.5 * a * t**2  # a is negative, so R decreases
            phase = "accel"
        elif t < t_accel + t_coast:
            # Coast phase - constant velocity
            t_coast_elapsed = t - t_accel
            a = 0
            V = V_final
            R_at_accel_end = R0 + 0.5 * (V_final / t_accel) * t_accel**2
            R = R_at_accel_end + V * t_coast_elapsed  # V is negative
            phase = "coast"
        else:
            # Stagnation (deceleration)
            t_stag_elapsed = t - t_accel - t_coast
            # Decelerate from V_final to 0
            a = -V_final / t_stag  # Positive (opposing V)
            V = V_final + a * t_stag_elapsed
            R = max(30, R - abs(V) * dt)  # Just track position decrease
            phase = "stag"
        
        # Make sure R stays positive
        R = max(R, 30)
        
        # Estimate shell parameters
        shell_thickness = max(R * 0.1, 20)  # Shell is ~10% of R
        rho_shell = 10 * (R0 / max(R, 50))**2  # Compress as R shrinks
        rho_ablator = 1
        rho_hot_spot = 1 * (R0 / max(R, 50))**2
        V_abl = 50 if t < t_accel else 0  # Ablation only during acceleration
        
        # Step RT model
        state = rt.step(
            t, dt, max(R, 50), V, a,
            rho_shell, rho_ablator, rho_hot_spot,
            V_abl, shell_thickness
        )
        
        trajectory.append({
            't': t, 'R': R, 'V': V, 'a': a, 'phase': phase,
            'amp': state.total_amplitude
        })
        
        if int(t / 1.0) > int((t - dt) / 1.0):  # Print every 1 ns
            print(f"  t={t:5.1f} ns  R={R:5.0f} μm  V={V:+6.0f} km/s  "
                  f"amp={state.total_amplitude:5.2f} μm  phase={phase}")
        
        t += dt
    
    print("─" * 50)
    print()
    
    # Results
    final_state = rt.history[-1]
    degradation = rt.degradation_factor()
    
    print("  RESULTS")
    print("  " + "─" * 40)
    print(f"  Initial roughness:     {rt.initial_roughness:.1f} μm")
    print(f"  Final RMS amplitude:   {final_state.total_amplitude:.1f} μm")
    print(f"  Maximum amplitude:     {final_state.max_amplitude:.1f} μm")
    print(f"  Mix width:             {final_state.mix_width:.1f} μm")
    print(f"  Mix fraction:          {final_state.mix_fraction*100:.1f}%")
    print(f"  Yield degradation:     {(1-degradation)*100:.1f}%")
    print()
    
    if degradation > 0.5:
        print("  ✓ Mix is manageable - ignition possible")
    else:
        print("  ✗ Severe mix - ignition compromised")
    
    # Generate plots
    print("\nGenerating RT plots...")
    rt.plot_evolution('rt_evolution.png')
    rt.plot_mode_spectrum('rt_spectrum.png')
    
    # Plot implosion trajectory with RT
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    ax = axes[0]
    ax.plot([d['t'] for d in trajectory], [d['R'] for d in trajectory], 
            'b-', linewidth=2)
    ax.fill_between(
        [d['t'] for d in trajectory],
        [d['R'] - d['amp'] for d in trajectory],
        [d['R'] + d['amp'] for d in trajectory],
        alpha=0.3, color='red', label='RT amplitude'
    )
    ax.set_xlabel('Time [ns]')
    ax.set_ylabel('Radius [μm]')
    ax.set_title('Shell Trajectory with RT Perturbations')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    ax.semilogy([d['t'] for d in trajectory], 
                [max(d['amp'], 0.01) for d in trajectory], 
                'r-', linewidth=2)
    ax.axhline(y=1, color='k', linestyle='--', label='Initial')
    ax.set_xlabel('Time [ns]')
    ax.set_ylabel('RMS Amplitude [μm]')
    ax.set_title('RT Amplitude Growth')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('rt_trajectory.png', dpi=150, bbox_inches='tight')
    print("Saved: rt_trajectory.png")
    plt.close()
    
    print()
    print("Generated plots:")
    print("  - rt_evolution.png")
    print("  - rt_spectrum.png")
    print("  - rt_trajectory.png")
    print()
    
    return rt, trajectory


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    np.random.seed(42)  # Reproducibility
    rt, trajectory = simulate_rt_with_implosion()
