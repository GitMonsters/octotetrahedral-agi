"""
Integrated ICF Simulation with Full Physics
============================================

Couples all physics modules into a unified simulation:
1. Core implosion dynamics (0D model)
2. Radiation losses (Bremsstrahlung with opacity)
3. Rayleigh-Taylor instabilities
4. Neutron diagnostics
5. Manufacturing defects

This provides a more accurate prediction of yield by accounting
for all the loss mechanisms that reduce performance from ideal 1D.

Author: Evan Pieser
Date: 2026
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable
import sys

# Import existing modules
from icf_simulation import NIF, State, sigma_v, nif_pulse, EV, M_P, M_DT, E_FUSION, ALPHA
from radiation import (RadiationModel, radiation_loss_rate, bremsstrahlung_simplified,
                       conduction_loss_rate, K_B)
from rayleigh_taylor import RTInstabilityModel, RTState


@dataclass
class IntegratedState:
    """Complete simulation state including all physics."""
    # Core implosion
    t: float              # Time [ns]
    R: float              # Outer shell radius [μm]
    V: float              # Implosion velocity [km/s]
    R_hs: float           # Hot spot radius [μm]
    T_hs: float           # Hot spot temperature [keV]
    rho_hs: float         # Hot spot density [g/cm³]
    P_hs: float           # Hot spot pressure [Gbar]
    yield_MJ: float       # Cumulative yield [MJ]
    CR: float             # Convergence ratio
    
    # Radiation
    P_rad_TW: float       # Radiation power loss [TW]
    tau_optical: float    # Optical depth
    
    # RT instabilities
    rt_amplitude: float   # RMS RT amplitude [μm]
    mix_fraction: float   # Fraction of shell mixed
    
    # Neutron diagnostics
    neutron_rate: float   # Neutron production rate [1/s]
    dsr: float            # Down-scatter ratio
    
    # Yield accounting
    yield_1D: float       # Ideal 1D yield [MJ]
    yield_actual: float   # Degraded yield [MJ]
    degradation: float    # Total degradation factor


@dataclass  
class NeutronDiagnostics:
    """Neutron diagnostic observables."""
    total_yield: float        # Total neutron yield (14.1 MeV)
    bang_time: float          # Time of peak neutron production [ns]
    burn_width: float         # FWHM of neutron pulse [ns]
    burn_weighted_T: float    # Burn-weighted ion temperature [keV]
    burn_weighted_rho: float  # Burn-weighted density [g/cm³]
    rhoR_total: float         # Areal density at bang time [g/cm²]
    dsr: float                # Down-scatter ratio (ρR diagnostic)
    neutron_spectrum: np.ndarray  # Energy spectrum [MeV], [counts]


class IntegratedICF:
    """
    Integrated ICF simulation coupling all physics modules.
    
    This is an enhanced version of the NIF class that includes:
    - Radiation losses subtracted from hot spot energy
    - RT degradation applied to fusion yield
    - Neutron diagnostic calculations
    - Manufacturing defect impacts
    """
    
    def __init__(self, 
                 include_radiation: bool = True,
                 include_rt: bool = True,
                 include_conduction: bool = True,
                 initial_roughness_um: float = 1.0,
                 defect_model: Optional[dict] = None):
        """
        Args:
            include_radiation: Include radiation cooling
            include_rt: Include RT instability degradation
            include_conduction: Include thermal conduction losses
            initial_roughness_um: Surface roughness for RT seeding
            defect_model: Dict with defect parameters (fill_tube, seams, etc.)
        """
        # Core simulation
        self.core = NIF()
        
        # Physics flags
        self.include_radiation = include_radiation
        self.include_rt = include_rt
        self.include_conduction = include_conduction
        
        # Radiation model
        self.radiation = RadiationModel(include_trapping=True) if include_radiation else None
        
        # RT model
        self.rt = RTInstabilityModel(
            initial_roughness_um=initial_roughness_um,
            ablation_front=True,
            hot_spot_interface=True
        ) if include_rt else None
        
        # Defect model
        self.defects = defect_model or {}
        
        # Neutron tracking
        self.neutron_history = []
        self.burn_history = []
        
        # State history
        self.history: List[IntegratedState] = []
        
        # Yield tracking
        self.yield_1D = 0.0  # Ideal yield
        self.yield_actual = 0.0  # Actual degraded yield
        
    def reset(self):
        """Reset simulation to initial conditions."""
        self.core.reset()
        if self.rt:
            self.rt = RTInstabilityModel(
                initial_roughness_um=self.rt.initial_roughness,
                ablation_front=True,
                hot_spot_interface=True
            )
        if self.radiation:
            self.radiation = RadiationModel(include_trapping=True)
        self.neutron_history = []
        self.burn_history = []
        self.history = []
        self.yield_1D = 0.0
        self.yield_actual = 0.0
        
    def compute_radiation_loss(self, dt_ns: float) -> Tuple[float, float, float]:
        """
        Compute radiation energy loss for this timestep.
        
        Returns:
            (dT_rad, P_rad_TW, tau): Temperature change, power, optical depth
        """
        if not self.include_radiation or self.core.T_hs < 1:
            return 0.0, 0.0, 0.0
            
        T_keV = self.core.T_hs
        rho = self.core.rho_hs()
        R_cm = self.core.R_hs * 1e-4
        V_cm3 = (4/3) * np.pi * R_cm**3
        
        # Get radiation loss rate
        P_loss_density, tau = radiation_loss_rate(rho, T_keV, R_cm)
        P_loss = P_loss_density * V_cm3  # erg/s
        P_rad_TW = P_loss / 1e19  # Convert to TW
        
        # Energy loss
        E_loss = P_loss * (dt_ns * 1e-9)  # erg
        
        # Temperature change
        m_ion = 2.5 * M_P
        N = rho * V_cm3 / m_ion
        E_thermal = 1.5 * N * T_keV * 1e3 * EV
        
        if E_thermal > 0:
            dT_rad = -T_keV * E_loss / E_thermal
            dT_rad = max(dT_rad, -0.5 * T_keV)  # Don't lose more than 50% per step
        else:
            dT_rad = 0
            
        return dT_rad, P_rad_TW, tau
    
    def compute_conduction_loss(self, dt_ns: float) -> float:
        """
        Compute electron conduction cooling.
        
        Returns:
            dT_cond: Temperature change due to conduction [keV]
        """
        if not self.include_conduction or self.core.T_hs < 5:
            return 0.0
            
        T_hs_keV = self.core.T_hs
        T_shell_keV = 0.5  # Cold shell temperature
        rho = self.core.rho_hs()
        R_cm = self.core.R_hs * 1e-4
        dr_cm = 10e-4  # Interface width ~10 μm
        V_cm3 = (4/3) * np.pi * R_cm**3
        
        P_cond = conduction_loss_rate(T_hs_keV, T_shell_keV, rho, R_cm, dr_cm)
        E_loss = P_cond * (dt_ns * 1e-9)
        
        # Temperature change
        m_ion = 2.5 * M_P
        N = rho * V_cm3 / m_ion
        E_thermal = 1.5 * N * T_hs_keV * 1e3 * EV
        
        if E_thermal > 0:
            dT_cond = -T_hs_keV * E_loss / E_thermal
            dT_cond = max(dT_cond, -0.3 * T_hs_keV)  # Limit
        else:
            dT_cond = 0
            
        return dT_cond
    
    def compute_rt_state(self, dt_ns: float) -> Tuple[float, float]:
        """
        Update RT instabilities and get mix state.
        
        Returns:
            (rt_amplitude, mix_fraction)
        """
        if not self.include_rt:
            return 0.0, 0.0
            
        # Estimate acceleration
        # During laser: a ~ 80 km/s/ns = 8e13 cm/s²
        # During stagnation: positive (deceleration)
        if self.core.V < -50:
            a_kms2 = -80  # Accelerating inward
        elif self.core.stagnated:
            a_kms2 = 100  # Decelerating
        else:
            a_kms2 = 0
            
        # Shell parameters
        shell_thickness = max(self.core.R - self.core.R_hs, 20)
        rho_shell = 10 * (self.core.R0 / max(self.core.R, 50))**2
        rho_ablator = 1
        rho_hot_spot = self.core.rho_hs()
        V_abl = 50 if self.core.V < -50 else 0
        
        rt_state = self.rt.step(
            self.core.t, dt_ns, self.core.R, self.core.V, a_kms2,
            rho_shell, rho_ablator, rho_hot_spot, V_abl, shell_thickness
        )
        
        return rt_state.total_amplitude, rt_state.mix_fraction
    
    def compute_neutron_rate(self) -> Tuple[float, float]:
        """
        Compute neutron production rate and down-scatter ratio.
        
        Returns:
            (neutron_rate, dsr)
        """
        if self.core.T_hs < 4:
            return 0.0, 0.0
            
        sv = sigma_v(self.core.T_hs)
        rho = self.core.rho_hs()
        n = rho / M_DT
        
        R_hs_cm = self.core.R_hs * 1e-4
        V_hs = (4/3) * np.pi * R_hs_cm**3
        
        # Neutron rate = reaction rate (one 14.1 MeV neutron per DT fusion)
        reaction_rate = 0.25 * n**2 * sv * V_hs  # reactions/s
        neutron_rate = reaction_rate
        
        # Down-scatter ratio (DSR) - fraction of neutrons that scatter
        # DSR ≈ 0.05 * ρR / (g/cm²) for DT
        shell_thickness_cm = (self.core.R - self.core.R_hs) * 1e-4
        rho_shell = 10 * (self.core.R0 / max(self.core.R, 50))**2
        rhoR = rho_shell * shell_thickness_cm
        dsr = 0.05 * rhoR
        
        return neutron_rate, dsr
    
    def compute_yield_degradation(self) -> float:
        """
        Compute total yield degradation from all sources.
        
        Returns:
            degradation_factor (0 to 1, where 1 = no degradation)
        """
        degradation = 1.0
        
        # RT mix degradation
        if self.include_rt and self.rt and self.rt.history:
            rt_deg = self.rt.degradation_factor()
            degradation *= rt_deg
            
        # Defect degradation
        if self.defects:
            # Fill tube: ~5-10% degradation
            if 'fill_tube' in self.defects:
                degradation *= (1 - self.defects['fill_tube'].get('degradation', 0.05))
            # Tent/mount: ~2-5%
            if 'tent' in self.defects:
                degradation *= (1 - self.defects['tent'].get('degradation', 0.03))
            # Surface roughness additional: based on RMS
            if 'extra_roughness' in self.defects:
                rms = self.defects['extra_roughness'].get('rms_um', 0)
                degradation *= max(1 - 0.1 * rms, 0.5)  # 10% per μm extra
                
        return degradation
    
    def step(self, dt: float, P_laser_TW: float) -> IntegratedState:
        """
        Advance simulation by dt nanoseconds with all physics.
        
        Args:
            dt: Timestep [ns]
            P_laser_TW: Laser power [TW]
            
        Returns:
            IntegratedState with all quantities
        """
        # Store pre-step yield for 1D comparison
        yield_pre = self.core.yield_MJ
        
        # Core implosion step (computes ideal yield)
        core_state = self.core.step(dt, P_laser_TW)
        
        # 1D yield increment
        d_yield_1D = self.core.yield_MJ - yield_pre
        self.yield_1D += d_yield_1D
        
        # Radiation cooling
        dT_rad, P_rad_TW, tau = self.compute_radiation_loss(dt)
        self.core.T_hs = max(self.core.T_hs + dT_rad, 0.1)
        
        # Conduction cooling
        dT_cond = self.compute_conduction_loss(dt)
        self.core.T_hs = max(self.core.T_hs + dT_cond, 0.1)
        
        # RT instabilities
        rt_amp, mix_frac = self.compute_rt_state(dt)
        
        # Neutron diagnostics
        neutron_rate, dsr = self.compute_neutron_rate()
        
        # Track neutron history for diagnostics
        if neutron_rate > 0:
            self.neutron_history.append({
                't': self.core.t,
                'rate': neutron_rate,
                'T': self.core.T_hs,
                'rho': self.core.rho_hs(),
                'R_hs': self.core.R_hs,
                'dsr': dsr
            })
        
        # Compute degradation factor
        degradation = self.compute_yield_degradation()
        
        # Actual yield (degraded)
        self.yield_actual = self.yield_1D * degradation
        
        # Create integrated state
        state = IntegratedState(
            t=self.core.t,
            R=self.core.R,
            V=self.core.V,
            R_hs=self.core.R_hs,
            T_hs=self.core.T_hs,
            rho_hs=self.core.rho_hs(),
            P_hs=self.core.P_hs_Gbar(),
            yield_MJ=self.core.yield_MJ,
            CR=core_state.CR,
            P_rad_TW=P_rad_TW,
            tau_optical=tau,
            rt_amplitude=rt_amp,
            mix_fraction=mix_frac,
            neutron_rate=neutron_rate,
            dsr=dsr,
            yield_1D=self.yield_1D,
            yield_actual=self.yield_actual,
            degradation=degradation
        )
        
        self.history.append(state)
        return state
    
    def run(self, t_end_ns: float, laser_func: Callable, dt: float = 0.01,
            verbose: bool = True) -> List[IntegratedState]:
        """
        Run integrated simulation.
        
        Args:
            t_end_ns: End time [ns]
            laser_func: Laser power function P(t) in TW
            dt: Timestep [ns]
            verbose: Print progress
            
        Returns:
            List of IntegratedState objects
        """
        while self.core.t < t_end_ns:
            P = laser_func(self.core.t)
            state = self.step(dt, P)
            
            if verbose and len(self.history) % 100 == 0:
                print(f"  t={state.t:5.2f}ns  R_hs={state.R_hs:5.0f}μm  "
                      f"T={state.T_hs:5.1f}keV  Y_1D={state.yield_1D:.3f}MJ  "
                      f"Y_act={state.yield_actual:.3f}MJ  deg={state.degradation:.2f}")
                
        return self.history
    
    def compute_neutron_diagnostics(self) -> NeutronDiagnostics:
        """
        Compute neutron diagnostic observables from history.
        
        Returns:
            NeutronDiagnostics object
        """
        if not self.neutron_history:
            return NeutronDiagnostics(
                total_yield=0, bang_time=0, burn_width=0,
                burn_weighted_T=0, burn_weighted_rho=0,
                rhoR_total=0, dsr=0, neutron_spectrum=np.array([])
            )
        
        # Extract arrays
        t = np.array([h['t'] for h in self.neutron_history])
        rate = np.array([h['rate'] for h in self.neutron_history])
        T = np.array([h['T'] for h in self.neutron_history])
        rho = np.array([h['rho'] for h in self.neutron_history])
        dsr = np.array([h['dsr'] for h in self.neutron_history])
        
        # Total yield (integrate rate)
        dt = t[1] - t[0] if len(t) > 1 else 0.01
        total_yield = np.sum(rate) * dt * 1e-9  # Convert ns to s
        
        # Bang time (time of peak rate)
        bang_idx = np.argmax(rate)
        bang_time = t[bang_idx]
        
        # Burn width (FWHM)
        half_max = 0.5 * rate[bang_idx]
        above_half = rate > half_max
        if np.any(above_half):
            burn_width = (t[above_half][-1] - t[above_half][0])
        else:
            burn_width = dt
            
        # Burn-weighted quantities (weighted by reaction rate)
        weights = rate / np.sum(rate) if np.sum(rate) > 0 else np.ones_like(rate)
        burn_weighted_T = np.sum(T * weights)
        burn_weighted_rho = np.sum(rho * weights)
        burn_weighted_dsr = np.sum(dsr * weights)
        
        # ρR at bang time
        R_hs = self.neutron_history[bang_idx]['R_hs'] * 1e-4  # cm
        rhoR_total = burn_weighted_rho * R_hs
        
        # Neutron energy spectrum
        # 14.1 MeV primary, broadened by ion temperature
        # FWHM ≈ 177 * sqrt(Ti [keV]) keV
        E_primary = 14.1  # MeV
        sigma_E = 0.177 * np.sqrt(burn_weighted_T) / 2.355  # MeV (convert FWHM to sigma)
        E_range = np.linspace(10, 17, 100)
        spectrum = np.exp(-0.5 * ((E_range - E_primary) / sigma_E)**2)
        
        # Add down-scattered neutrons (10-12 MeV peak)
        E_ds = 11.5  # MeV (single-scatter)
        sigma_ds = 1.0  # MeV
        spectrum += burn_weighted_dsr * np.exp(-0.5 * ((E_range - E_ds) / sigma_ds)**2)
        
        # Normalize
        spectrum /= np.max(spectrum)
        
        return NeutronDiagnostics(
            total_yield=total_yield,
            bang_time=bang_time,
            burn_width=burn_width,
            burn_weighted_T=burn_weighted_T,
            burn_weighted_rho=burn_weighted_rho,
            rhoR_total=rhoR_total,
            dsr=burn_weighted_dsr,
            neutron_spectrum=np.column_stack([E_range, spectrum])
        )


def run_integrated_simulation(include_radiation=True, include_rt=True,
                               defects=None, verbose=True):
    """
    Run the integrated ICF simulation with all physics.
    """
    print()
    print("╔" + "═" * 62 + "╗")
    print("║  INTEGRATED ICF SIMULATION WITH FULL PHYSICS                 ║")
    print("║  National Ignition Facility - December 5, 2022               ║")
    print("╚" + "═" * 62 + "╝")
    print()
    
    print("Physics modules:")
    print(f"  Radiation losses:    {'ON' if include_radiation else 'OFF'}")
    print(f"  RT instabilities:    {'ON' if include_rt else 'OFF'}")
    print(f"  Manufacturing defects: {'ON' if defects else 'OFF'}")
    print()
    
    # Create simulation
    sim = IntegratedICF(
        include_radiation=include_radiation,
        include_rt=include_rt,
        include_conduction=True,
        initial_roughness_um=1.0,
        defect_model=defects
    )
    
    print(f"  Target capsule:  {sim.core.R0:.0f} μm outer radius")
    print(f"  DT fuel mass:    {sim.core.M_shell:.0f} μg shell, {sim.core.M_hs:.1f} μg hot spot")
    print(f"  Laser energy:    2.05 MJ")
    print(f"  Peak power:      500 TW")
    print()
    print("─" * 64)
    
    # Run simulation
    states = sim.run(t_end_ns=14, laser_func=nif_pulse, dt=0.01, verbose=verbose)
    
    print("─" * 64)
    print()
    
    # Find key moments
    peak_rho = max(states, key=lambda s: s.rho_hs)
    peak_T = max(states, key=lambda s: s.T_hs)
    min_V = min(states, key=lambda s: s.V)
    final = states[-1]
    
    # Compute neutron diagnostics
    neutron_diag = sim.compute_neutron_diagnostics()
    
    print("  IMPLOSION RESULTS")
    print("  " + "─" * 50)
    print(f"  Peak implosion velocity:  {min_V.V:.0f} km/s")
    print(f"  Stagnation time:          {peak_rho.t:.2f} ns")
    print(f"  Convergence ratio:        {peak_rho.CR:.0f}×")
    print(f"  Minimum hot spot radius:  {peak_rho.R_hs:.0f} μm")
    print(f"  Peak hot spot temperature:{peak_T.T_hs:.1f} keV")
    print(f"  Peak hot spot density:    {peak_rho.rho_hs:.0f} g/cm³")
    print()
    
    print("  YIELD ANALYSIS")
    print("  " + "─" * 50)
    print(f"  Ideal 1D yield:           {final.yield_1D:.2f} MJ")
    print(f"  Actual (degraded) yield:  {final.yield_actual:.2f} MJ")
    print(f"  Total degradation:        {(1-final.degradation)*100:.1f}%")
    
    if include_rt:
        rt_deg = sim.rt.degradation_factor()
        print(f"    - RT mix:               {(1-rt_deg)*100:.1f}%")
    if defects:
        print(f"    - Defects:              included in total")
    print()
    
    Q_1D = final.yield_1D / 2.05
    Q_actual = final.yield_actual / 2.05
    
    print(f"  Scientific gain (Q):")
    print(f"    - Ideal:   Q = {Q_1D:.2f}")
    print(f"    - Actual:  Q = {Q_actual:.2f}")
    print()
    
    print("  NEUTRON DIAGNOSTICS")
    print("  " + "─" * 50)
    print(f"  Bang time:              {neutron_diag.bang_time:.2f} ns")
    print(f"  Burn width (FWHM):      {neutron_diag.burn_width:.2f} ns")
    print(f"  Burn-weighted Ti:       {neutron_diag.burn_weighted_T:.1f} keV")
    print(f"  Burn-weighted ρ:        {neutron_diag.burn_weighted_rho:.0f} g/cm³")
    print(f"  Down-scatter ratio:     {neutron_diag.dsr:.3f}")
    print()
    
    if Q_actual > 1:
        print("  ╔════════════════════════════════════════════════════╗")
        print("  ║                                                    ║")
        print("  ║            ★ ★ ★  IGNITION  ★ ★ ★                ║")
        print("  ║                                                    ║")
        print("  ║   Fusion gain Q > 1 achieved with full physics!   ║")
        print("  ║                                                    ║")
        print("  ╚════════════════════════════════════════════════════╝")
    else:
        print(f"  Gain Q = {Q_actual:.2f} (ignition requires Q > 1)")
    
    print()
    print("═" * 64)
    
    return sim, states, neutron_diag


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Run with all physics enabled
    sim, states, neutron_diag = run_integrated_simulation(
        include_radiation=True,
        include_rt=True,
        defects=None  # Can add {'fill_tube': {'degradation': 0.05}}
    )
    
    # Generate comparison plots
    print("\nGenerating integrated physics plots...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    t = [s.t for s in states]
    
    # 1. Temperature with radiation cooling
    ax = axes[0, 0]
    ax.plot(t, [s.T_hs for s in states], 'r-', linewidth=2)
    ax.set_xlabel('Time [ns]')
    ax.set_ylabel('Temperature [keV]')
    ax.set_title('Hot Spot Temperature (with radiation)')
    ax.grid(True, alpha=0.3)
    
    # 2. Radiation power
    ax = axes[0, 1]
    ax.semilogy(t, [max(s.P_rad_TW, 1e-6) for s in states], 'b-', linewidth=2)
    ax.set_xlabel('Time [ns]')
    ax.set_ylabel('Radiation Power [TW]')
    ax.set_title('Radiation Losses')
    ax.grid(True, alpha=0.3)
    
    # 3. RT amplitude
    ax = axes[0, 2]
    ax.semilogy(t, [max(s.rt_amplitude, 0.01) for s in states], 'g-', linewidth=2)
    ax.set_xlabel('Time [ns]')
    ax.set_ylabel('RMS Amplitude [μm]')
    ax.set_title('RT Instability Growth')
    ax.grid(True, alpha=0.3)
    
    # 4. Yield comparison
    ax = axes[1, 0]
    ax.plot(t, [s.yield_1D for s in states], 'b-', linewidth=2, label='Ideal (1D)')
    ax.plot(t, [s.yield_actual for s in states], 'r-', linewidth=2, label='Actual')
    ax.set_xlabel('Time [ns]')
    ax.set_ylabel('Yield [MJ]')
    ax.set_title('Yield: Ideal vs Actual')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Degradation factor
    ax = axes[1, 1]
    ax.plot(t, [s.degradation for s in states], 'purple', linewidth=2)
    ax.set_xlabel('Time [ns]')
    ax.set_ylabel('Degradation Factor')
    ax.set_title('Yield Degradation (1 = no loss)')
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)
    
    # 6. Neutron spectrum
    ax = axes[1, 2]
    if len(neutron_diag.neutron_spectrum) > 0:
        E = neutron_diag.neutron_spectrum[:, 0]
        counts = neutron_diag.neutron_spectrum[:, 1]
        ax.plot(E, counts, 'k-', linewidth=2)
        ax.axvline(x=14.1, color='r', linestyle='--', label='Primary (14.1 MeV)')
        ax.axvline(x=11.5, color='b', linestyle='--', label='Down-scattered')
    ax.set_xlabel('Energy [MeV]')
    ax.set_ylabel('Intensity (normalized)')
    ax.set_title('Neutron Energy Spectrum')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('integrated_physics.png', dpi=150, bbox_inches='tight')
    print("Saved: integrated_physics.png")
    plt.close()
    
    # Run comparison: with and without physics
    print("\n" + "=" * 64)
    print("COMPARISON: PHYSICS MODULE EFFECTS")
    print("=" * 64)
    
    results = {}
    
    # 1D (no degradation)
    sim_1d, states_1d, _ = run_integrated_simulation(
        include_radiation=False, include_rt=False, verbose=False
    )
    results['1D (ideal)'] = states_1d[-1].yield_1D
    
    # With radiation only
    sim_rad, states_rad, _ = run_integrated_simulation(
        include_radiation=True, include_rt=False, verbose=False
    )
    results['+ Radiation'] = states_rad[-1].yield_1D
    
    # With RT only
    sim_rt, states_rt, _ = run_integrated_simulation(
        include_radiation=False, include_rt=True, verbose=False
    )
    results['+ RT mix'] = states_rt[-1].yield_actual
    
    # Full physics
    results['Full physics'] = states[-1].yield_actual
    
    print("\nYield comparison:")
    print("─" * 40)
    for name, yield_val in results.items():
        Q = yield_val / 2.05
        print(f"  {name:20s}: {yield_val:.2f} MJ  (Q = {Q:.2f})")
    print()
