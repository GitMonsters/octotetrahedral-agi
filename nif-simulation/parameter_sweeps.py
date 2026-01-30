"""
Parameter Sweep Studies for ICF Ignition
=========================================

Explores the ignition cliff and parameter space around NIF's
December 2022 shot to understand:
1. Minimum laser energy for ignition
2. Impact of target size
3. Effect of pulse shapes
4. Ignition threshold contours

Author: Evan Pieser
Date: 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from icf_simulation import NIF, State
from dataclasses import dataclass
from typing import Callable, List, Tuple
import warnings
warnings.filterwarnings('ignore')


@dataclass
class SweepResult:
    """Result from a single parameter point."""
    laser_energy_MJ: float
    peak_power_TW: float
    target_radius_um: float
    fuel_mass_ug: float
    yield_MJ: float
    gain_Q: float
    peak_T_keV: float
    peak_rho_gcc: float
    convergence_ratio: float
    ignited: bool


def make_scaled_pulse(energy_MJ: float, peak_power_TW: float) -> Callable:
    """
    Create a laser pulse function with specified energy and peak power.
    
    The NIF baseline is 2.05 MJ at 500 TW peak.
    We scale the pulse duration to achieve the desired energy.
    """
    # Baseline pulse delivers ~2.05 MJ
    # Energy = integral of power over time
    # For trapezoidal pulse: E ≈ P_peak * t_peak * 0.8 (accounting for ramps)
    
    baseline_energy = 2.05  # MJ
    baseline_peak = 500     # TW
    baseline_t_peak = 7     # ns at peak (from t=5 to t=12)
    
    # Scale duration to achieve desired energy at given power
    energy_ratio = energy_MJ / baseline_energy
    power_ratio = peak_power_TW / baseline_peak
    
    # If we change power, adjust duration to get desired energy
    duration_scale = energy_ratio / power_ratio
    
    def pulse(t_ns):
        # Scale time to adjust duration
        t_scaled = t_ns / duration_scale
        
        if t_scaled < 0:
            return 0
        elif t_scaled < 2:
            # Foot pulse (scaled)
            return 30 * power_ratio
        elif t_scaled < 5:
            # Rise
            return (30 + 470 * (t_scaled - 2) / 3) * power_ratio
        elif t_scaled < 12:
            # Peak
            return peak_power_TW
        elif t_scaled < 13:
            # Fall
            return peak_power_TW * (13 - t_scaled)
        else:
            return 0
    
    return pulse, 14 * duration_scale  # Return pulse and adjusted end time


def run_single_point(laser_energy_MJ: float, peak_power_TW: float = 500,
                     target_scale: float = 1.0) -> SweepResult:
    """
    Run simulation for a single parameter point.
    
    Args:
        laser_energy_MJ: Laser energy in MJ
        peak_power_TW: Peak laser power in TW
        target_scale: Scale factor for target size (1.0 = NIF baseline)
    
    Returns:
        SweepResult with all metrics
    """
    sim = NIF()
    
    # Scale target
    sim.R0 *= target_scale
    sim.R_hs0 *= target_scale
    sim.M_shell *= target_scale**3  # Mass scales with volume
    sim.M_hs *= target_scale**3
    
    # Scale acceleration to match power
    base_accel = 80.0
    
    # Create pulse
    pulse, t_end = make_scaled_pulse(laser_energy_MJ, peak_power_TW)
    
    # Run simulation (suppress output)
    sim.reset()
    states = []
    dt = 0.01
    
    while sim.t < t_end:
        P = pulse(sim.t)
        
        # Adjust acceleration based on power and target size
        # Smaller targets accelerate faster
        P_norm = P / (500.0 * target_scale)
        accel = -base_accel * P_norm
        
        if P > 10 and sim.V > sim.V_imp_target / target_scale**0.5:
            dV = accel * dt
            sim.V = max(sim.V + dV, sim.V_imp_target / target_scale**0.5)
        
        s = sim.step(dt, P)
        states.append(s)
    
    # Extract results
    peak_rho = max(states, key=lambda s: s.rho_hs)
    peak_T = max(states, key=lambda s: s.T_hs)
    final = states[-1]
    
    Q = final.yield_MJ / laser_energy_MJ if laser_energy_MJ > 0 else 0
    
    return SweepResult(
        laser_energy_MJ=laser_energy_MJ,
        peak_power_TW=peak_power_TW,
        target_radius_um=sim.R0,
        fuel_mass_ug=sim.M_shell + sim.M_hs,
        yield_MJ=final.yield_MJ,
        gain_Q=Q,
        peak_T_keV=peak_T.T_hs,
        peak_rho_gcc=peak_rho.rho_hs,
        convergence_ratio=peak_rho.CR,
        ignited=Q > 1.0
    )


def sweep_laser_energy(energies: np.ndarray = None) -> List[SweepResult]:
    """Sweep laser energy at fixed power and target size."""
    if energies is None:
        energies = np.linspace(0.5, 3.0, 26)
    
    print("Sweeping laser energy...")
    results = []
    for i, E in enumerate(energies):
        result = run_single_point(E)
        results.append(result)
        status = "IGNITION" if result.ignited else ""
        print(f"  E={E:.2f} MJ -> Q={result.gain_Q:.3f} {status}")
    
    return results


def sweep_2d_energy_power(energies: np.ndarray = None, 
                          powers: np.ndarray = None) -> np.ndarray:
    """2D sweep of energy and power."""
    if energies is None:
        energies = np.linspace(0.5, 3.0, 20)
    if powers is None:
        powers = np.linspace(200, 700, 20)
    
    print(f"2D sweep: {len(energies)} x {len(powers)} = {len(energies)*len(powers)} points")
    
    Q_grid = np.zeros((len(powers), len(energies)))
    
    for i, P in enumerate(powers):
        for j, E in enumerate(energies):
            result = run_single_point(E, P)
            Q_grid[i, j] = result.gain_Q
        print(f"  Power {P:.0f} TW complete")
    
    return energies, powers, Q_grid


def sweep_2d_energy_target(energies: np.ndarray = None,
                           scales: np.ndarray = None) -> Tuple:
    """2D sweep of energy and target size."""
    if energies is None:
        energies = np.linspace(0.5, 3.0, 20)
    if scales is None:
        scales = np.linspace(0.7, 1.3, 20)
    
    print(f"2D sweep: {len(energies)} x {len(scales)} = {len(energies)*len(scales)} points")
    
    Q_grid = np.zeros((len(scales), len(energies)))
    
    for i, s in enumerate(scales):
        for j, E in enumerate(energies):
            result = run_single_point(E, target_scale=s)
            Q_grid[i, j] = result.gain_Q
        print(f"  Scale {s:.2f}x complete")
    
    return energies, scales, Q_grid


def plot_energy_sweep(results: List[SweepResult], save_path: str = None):
    """Plot gain vs laser energy."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    E = [r.laser_energy_MJ for r in results]
    Q = [r.gain_Q for r in results]
    Y = [r.yield_MJ for r in results]
    T = [r.peak_T_keV for r in results]
    rho = [r.peak_rho_gcc for r in results]
    
    # Gain vs Energy
    ax = axes[0, 0]
    ax.plot(E, Q, 'b-o', linewidth=2, markersize=6)
    ax.axhline(y=1, color='r', linestyle='--', label='Ignition threshold (Q=1)')
    ax.axvline(x=2.05, color='g', linestyle=':', alpha=0.7, label='NIF Dec 2022 (2.05 MJ)')
    ax.set_xlabel('Laser Energy [MJ]', fontsize=12)
    ax.set_ylabel('Scientific Gain Q', fontsize=12)
    ax.set_title('Ignition Cliff', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(Q) * 1.1)
    
    # Yield vs Energy
    ax = axes[0, 1]
    ax.semilogy(E, [max(y, 0.001) for y in Y], 'r-o', linewidth=2, markersize=6)
    ax.axhline(y=2.05, color='g', linestyle=':', alpha=0.7, label='Input energy')
    ax.set_xlabel('Laser Energy [MJ]', fontsize=12)
    ax.set_ylabel('Fusion Yield [MJ]', fontsize=12)
    ax.set_title('Yield Amplification', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Temperature vs Energy
    ax = axes[1, 0]
    ax.plot(E, T, 'orange', marker='o', linewidth=2, markersize=6)
    ax.axhline(y=4, color='purple', linestyle='--', alpha=0.7, label='Burn threshold (~4 keV)')
    ax.set_xlabel('Laser Energy [MJ]', fontsize=12)
    ax.set_ylabel('Peak Hot Spot Temperature [keV]', fontsize=12)
    ax.set_title('Hot Spot Heating', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Density vs Energy  
    ax = axes[1, 1]
    ax.plot(E, rho, 'purple', marker='o', linewidth=2, markersize=6)
    ax.set_xlabel('Laser Energy [MJ]', fontsize=12)
    ax.set_ylabel('Peak Hot Spot Density [g/cm³]', fontsize=12)
    ax.set_title('Compression Achievement', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()
    return fig


def plot_2d_sweep(energies, param2, Q_grid, param2_label: str, 
                  save_path: str = None):
    """Plot 2D parameter sweep as contour map."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create contour plot
    E_mesh, P_mesh = np.meshgrid(energies, param2)
    
    # Clip for log scale
    Q_plot = np.clip(Q_grid, 0.01, None)
    
    # Filled contours
    levels = np.array([0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.2, 1.5, 2.0, 3.0])
    cf = ax.contourf(E_mesh, P_mesh, Q_plot, levels=levels, 
                     cmap='RdYlGn', extend='both')
    
    # Ignition contour (Q=1)
    cs = ax.contour(E_mesh, P_mesh, Q_grid, levels=[1.0], 
                    colors='black', linewidths=3)
    ax.clabel(cs, fmt='Q=1', fontsize=12)
    
    # NIF point
    if 'Power' in param2_label:
        ax.plot(2.05, 500, 'w*', markersize=20, markeredgecolor='black', 
                markeredgewidth=2, label='NIF Dec 2022')
    elif 'Scale' in param2_label or 'Size' in param2_label:
        ax.plot(2.05, 1.0, 'w*', markersize=20, markeredgecolor='black',
                markeredgewidth=2, label='NIF Dec 2022')
    
    ax.set_xlabel('Laser Energy [MJ]', fontsize=14)
    ax.set_ylabel(param2_label, fontsize=14)
    ax.set_title('Ignition Parameter Space', fontsize=16)
    
    cbar = plt.colorbar(cf, ax=ax, label='Scientific Gain Q')
    ax.legend(loc='upper left', fontsize=12)
    ax.grid(True, alpha=0.3, color='white')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()
    return fig


def find_ignition_threshold(power_TW: float = 500, target_scale: float = 1.0,
                            E_low: float = 0.5, E_high: float = 3.0,
                            tol: float = 0.01) -> float:
    """
    Binary search to find ignition threshold energy.
    
    Returns:
        Laser energy (MJ) at which Q ≈ 1
    """
    while E_high - E_low > tol:
        E_mid = (E_low + E_high) / 2
        result = run_single_point(E_mid, power_TW, target_scale)
        
        if result.gain_Q > 1.0:
            E_high = E_mid
        else:
            E_low = E_mid
    
    return (E_low + E_high) / 2


def pulse_shape_study():
    """
    Compare different pulse shapes.
    """
    print("\nPulse Shape Study")
    print("=" * 50)
    
    results = {}
    
    # 1. Standard NIF pulse
    print("\n1. Standard NIF pulse (foot + main)...")
    result = run_single_point(2.05, 500)
    results['Standard'] = result
    print(f"   Q = {result.gain_Q:.3f}")
    
    # 2. Square pulse (constant power)
    print("\n2. Square pulse...")
    def square_pulse_factory(energy_MJ, peak_TW):
        duration = energy_MJ / (peak_TW * 1e-6)  # ns
        def pulse(t):
            return peak_TW if 0 <= t <= duration else 0
        return pulse, duration * 1.5
    
    sim = NIF()
    sim.reset()
    pulse, t_end = square_pulse_factory(2.05, 500)
    states = []
    while sim.t < t_end:
        P = pulse(sim.t)
        s = sim.step(0.01, P)
        states.append(s)
    final = states[-1]
    results['Square'] = SweepResult(
        laser_energy_MJ=2.05, peak_power_TW=500, target_radius_um=1100,
        fuel_mass_ug=180, yield_MJ=final.yield_MJ, gain_Q=final.yield_MJ/2.05,
        peak_T_keV=max(s.T_hs for s in states),
        peak_rho_gcc=max(s.rho_hs for s in states),
        convergence_ratio=max(s.CR for s in states),
        ignited=final.yield_MJ/2.05 > 1
    )
    print(f"   Q = {results['Square'].gain_Q:.3f}")
    
    # 3. High-foot pulse (larger foot, shorter main)
    print("\n3. High-foot pulse...")
    def high_foot_pulse(t_ns):
        if t_ns < 0:
            return 0
        elif t_ns < 4:
            return 100  # Higher foot
        elif t_ns < 6:
            return 100 + 400 * (t_ns - 4) / 2
        elif t_ns < 11:
            return 500
        elif t_ns < 12:
            return 500 * (12 - t_ns)
        else:
            return 0
    
    sim = NIF()
    sim.reset()
    states = []
    while sim.t < 14:
        P = high_foot_pulse(sim.t)
        s = sim.step(0.01, P)
        states.append(s)
    final = states[-1]
    results['High-foot'] = SweepResult(
        laser_energy_MJ=2.05, peak_power_TW=500, target_radius_um=1100,
        fuel_mass_ug=180, yield_MJ=final.yield_MJ, gain_Q=final.yield_MJ/2.05,
        peak_T_keV=max(s.T_hs for s in states),
        peak_rho_gcc=max(s.rho_hs for s in states),
        convergence_ratio=max(s.CR for s in states),
        ignited=final.yield_MJ/2.05 > 1
    )
    print(f"   Q = {results['High-foot'].gain_Q:.3f}")
    
    # 4. Adiabat-shaped pulse (gradual rise)
    print("\n4. Adiabat-shaped (gradual rise)...")
    def adiabat_pulse(t_ns):
        if t_ns < 0 or t_ns > 14:
            return 0
        elif t_ns < 10:
            # Gradual rise: P(t) = P_max * (t/t_rise)^2
            return 500 * (t_ns / 10)**2
        elif t_ns < 12:
            return 500
        else:
            return 500 * (14 - t_ns) / 2
    
    sim = NIF()
    sim.reset()
    states = []
    while sim.t < 16:
        P = adiabat_pulse(sim.t)
        s = sim.step(0.01, P)
        states.append(s)
    final = states[-1]
    results['Adiabat'] = SweepResult(
        laser_energy_MJ=2.05, peak_power_TW=500, target_radius_um=1100,
        fuel_mass_ug=180, yield_MJ=final.yield_MJ, gain_Q=final.yield_MJ/2.05,
        peak_T_keV=max(s.T_hs for s in states),
        peak_rho_gcc=max(s.rho_hs for s in states),
        convergence_ratio=max(s.CR for s in states),
        ignited=final.yield_MJ/2.05 > 1
    )
    print(f"   Q = {results['Adiabat'].gain_Q:.3f}")
    
    return results


def plot_pulse_comparison(results: dict, save_path: str = None):
    """Plot comparison of different pulse shapes."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Bar chart of Q values
    ax = axes[0]
    names = list(results.keys())
    Q_vals = [results[n].gain_Q for n in names]
    colors = ['green' if q > 1 else 'red' for q in Q_vals]
    
    bars = ax.bar(names, Q_vals, color=colors, edgecolor='black', linewidth=2)
    ax.axhline(y=1, color='black', linestyle='--', linewidth=2, label='Ignition (Q=1)')
    ax.set_ylabel('Scientific Gain Q', fontsize=14)
    ax.set_title('Pulse Shape Comparison (2.05 MJ)', fontsize=14)
    ax.legend()
    ax.set_ylim(0, max(Q_vals) * 1.2)
    
    for bar, q in zip(bars, Q_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{q:.2f}', ha='center', fontsize=12, fontweight='bold')
    
    # Right: Pulse shapes
    ax = axes[1]
    t = np.linspace(0, 16, 500)
    
    from icf_simulation import nif_pulse
    ax.plot(t, [nif_pulse(ti) for ti in t], 'b-', linewidth=2, label='Standard')
    ax.plot(t, [500 if 0 <= ti <= 4.1 else 0 for ti in t], 'r--', 
            linewidth=2, label='Square')
    
    def high_foot(ti):
        if ti < 0: return 0
        elif ti < 4: return 100
        elif ti < 6: return 100 + 400 * (ti - 4) / 2
        elif ti < 11: return 500
        elif ti < 12: return 500 * (12 - ti)
        else: return 0
    ax.plot(t, [high_foot(ti) for ti in t], 'g-.', linewidth=2, label='High-foot')
    
    def adiabat(ti):
        if ti < 0 or ti > 14: return 0
        elif ti < 10: return 500 * (ti / 10)**2
        elif ti < 12: return 500
        else: return 500 * (14 - ti) / 2
    ax.plot(t, [adiabat(ti) for ti in t], 'm:', linewidth=2, label='Adiabat')
    
    ax.set_xlabel('Time [ns]', fontsize=14)
    ax.set_ylabel('Laser Power [TW]', fontsize=14)
    ax.set_title('Pulse Shape Profiles', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 16)
    
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
    print("=" * 60)
    print("  ICF PARAMETER SWEEP STUDIES")
    print("=" * 60)
    print()
    
    # 1. Laser energy sweep
    print("\n" + "─" * 60)
    print("1. LASER ENERGY SWEEP (find ignition cliff)")
    print("─" * 60)
    energies = np.linspace(0.5, 3.0, 26)
    energy_results = sweep_laser_energy(energies)
    plot_energy_sweep(energy_results, 'energy_sweep.png')
    
    # Find threshold
    threshold = find_ignition_threshold()
    print(f"\n  >>> Ignition threshold: {threshold:.2f} MJ <<<")
    
    # 2. 2D Energy-Power sweep
    print("\n" + "─" * 60)
    print("2. ENERGY vs POWER PARAMETER SPACE")
    print("─" * 60)
    E, P, Q_EP = sweep_2d_energy_power(
        energies=np.linspace(0.5, 3.0, 15),
        powers=np.linspace(200, 700, 15)
    )
    plot_2d_sweep(E, P, Q_EP, 'Peak Laser Power [TW]', 'parameter_space_EP.png')
    
    # 3. 2D Energy-Target sweep
    print("\n" + "─" * 60)
    print("3. ENERGY vs TARGET SIZE PARAMETER SPACE")
    print("─" * 60)
    E, S, Q_ES = sweep_2d_energy_target(
        energies=np.linspace(0.5, 3.0, 15),
        scales=np.linspace(0.7, 1.3, 15)
    )
    plot_2d_sweep(E, S, Q_ES, 'Target Scale Factor', 'parameter_space_ET.png')
    
    # 4. Pulse shape study
    print("\n" + "─" * 60)
    print("4. PULSE SHAPE COMPARISON")
    print("─" * 60)
    pulse_results = pulse_shape_study()
    plot_pulse_comparison(pulse_results, 'pulse_shapes.png')
    
    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"\n  Ignition threshold energy: {threshold:.2f} MJ")
    print(f"  NIF December 2022 used:    2.05 MJ")
    print(f"  Margin above threshold:    {(2.05 - threshold)/threshold * 100:.1f}%")
    print()
    print("  Generated plots:")
    print("    - energy_sweep.png")
    print("    - parameter_space_EP.png")
    print("    - parameter_space_ET.png")
    print("    - pulse_shapes.png")
    print()
