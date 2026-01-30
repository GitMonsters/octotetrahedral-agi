"""
Radiation Physics for ICF Simulations
======================================

Implements radiation loss and transport mechanisms:
1. Bremsstrahlung (free-free) radiation
2. Line radiation (bound-bound)
3. Recombination radiation (free-bound)
4. Radiation transport (diffusion approximation)

At high temperatures relevant to ICF (T > 1 keV), Bremsstrahlung dominates.
This is the primary energy loss mechanism that must be overcome for ignition.

Key physics:
- P_Brem ∝ n_e² * T^(1/2) * Z_eff
- Radiation trapping at high ρR (optical depth τ >> 1)
- Rosseland mean opacity for diffusion

Author: Evan Pieser
Date: 2026
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple

# Physical constants (CGS)
C = 2.998e10         # cm/s (speed of light)
H = 6.626e-27        # erg·s (Planck constant)
K_B = 1.381e-16      # erg/K (Boltzmann constant)
M_E = 9.109e-28      # g (electron mass)
E_CHARGE = 4.803e-10 # esu (electron charge)
SIGMA_SB = 5.67e-5   # erg/cm²/s/K⁴ (Stefan-Boltzmann)
EV = 1.602e-12       # erg
M_P = 1.673e-24      # g


@dataclass
class RadiationState:
    """Radiation field state for a zone."""
    E_rad: float      # Radiation energy density [erg/cm³]
    T_rad: float      # Radiation temperature [keV]
    P_brem: float     # Bremsstrahlung power loss [erg/cm³/s]
    P_line: float     # Line radiation power [erg/cm³/s]
    kappa_R: float    # Rosseland mean opacity [cm²/g]
    tau: float        # Optical depth
    F_rad: float      # Radiation flux [erg/cm²/s]


def bremsstrahlung_power(rho: float, T_keV: float, Z_eff: float = 1.0) -> float:
    """
    Bremsstrahlung (free-free) emission power density.
    
    P_ff = 1.42e-27 * n_e * n_i * Z² * g_ff * T^(1/2)  [erg/cm³/s]
    
    where g_ff ≈ 1.2 is the Gaunt factor.
    
    For DT plasma: Z = 1, so Z² = 1
    
    Args:
        rho: Density [g/cm³]
        T_keV: Temperature [keV]
        Z_eff: Effective charge (1 for DT)
    
    Returns:
        Power density [erg/cm³/s]
    """
    if T_keV <= 0 or rho <= 0:
        return 0.0
    
    # Number densities
    m_ion = 2.5 * M_P  # DT average mass
    n_i = rho / m_ion  # ion density
    n_e = Z_eff * n_i  # electron density (fully ionized)
    
    # Temperature in Kelvin
    T_K = T_keV * 1e3 * EV / K_B
    
    # Gaunt factor (approximate)
    g_ff = 1.2
    
    # Bremsstrahlung power
    P_ff = 1.42e-27 * n_e * n_i * Z_eff**2 * g_ff * np.sqrt(T_K)
    
    return P_ff


def bremsstrahlung_simplified(rho: float, T_keV: float) -> float:
    """
    Simplified Bremsstrahlung formula for DT.
    
    P_brem ≈ 1.69e32 * Z² * n_i * n_e * T^(1/2) [erg/cm³/s]
    
    For DT (Z=1, A≈2.5):
    n_i = n_e = ρ / (2.5 * m_p) ≈ 2.4e23 * ρ cm⁻³
    
    P_brem ≈ 1.69e32 * (2.4e23 * ρ)² * T^(1/2)
           ≈ 9.7e78 * ρ² * T^(1/2) [erg/cm³/s]
    
    Wait, that's way too high. Let me use the standard formula:
    
    P_ff = 1.42e-27 * g_ff * Z² * n_e * n_i * T^(1/2) [erg/cm³/s]
    
    where T is in Kelvin.
    
    At T = 50 keV = 5.8e8 K, ρ = 100 g/cm³:
    n = 2.4e25 cm⁻³
    P = 1.42e-27 * 1.2 * 1 * (2.4e25)² * (5.8e8)^0.5
      ≈ 1.42e-27 * 1.2 * 5.76e50 * 2.4e4
      ≈ 2.4e28 erg/cm³/s
    """
    if T_keV <= 0 or rho <= 0:
        return 0.0
    
    # Number density
    m_ion = 2.5 * M_P
    n = rho / m_ion  # cm⁻³
    
    # Temperature in Kelvin
    T_K = T_keV * 1e3 * EV / K_B
    
    # Gaunt factor
    g_ff = 1.2
    
    # Bremsstrahlung power density
    # P = 1.42e-27 * g_ff * Z² * n_e * n_i * sqrt(T_K)
    P_brem = 1.42e-27 * g_ff * n * n * np.sqrt(T_K)
    
    return P_brem


def line_radiation_power(rho: float, T_keV: float, Z_eff: float = 1.0) -> float:
    """
    Line radiation (bound-bound transitions) power.
    
    For fully ionized DT (Z=1), there is no line radiation since
    there are no bound electrons. However, at lower temperatures
    or with impurities (C, O from ablator), line radiation can
    be significant.
    
    For pure DT, returns 0.
    """
    # Pure DT has no line radiation
    # Could add impurity model here
    return 0.0


def recombination_power(rho: float, T_keV: float, Z_eff: float = 1.0) -> float:
    """
    Recombination (free-bound) radiation power.
    
    P_fb ∝ n_e * n_i * Z² * T^(-1/2)
    
    Typically smaller than Bremsstrahlung at high T.
    Becomes important as plasma cools.
    """
    if T_keV <= 0.01 or rho <= 0:  # Only relevant at low T
        return 0.0
    
    m_ion = 2.5 * M_P
    n_i = rho / m_ion
    n_e = Z_eff * n_i
    
    T_K = T_keV * 1e3 * EV / K_B
    
    # Approximate recombination power
    # Much smaller than Bremsstrahlung at ICF temperatures
    P_fb = 1.7e-27 * n_e * n_i * Z_eff**2 / np.sqrt(T_K)
    
    return P_fb


def rosseland_opacity(rho: float, T_keV: float, Z_eff: float = 1.0) -> float:
    """
    Rosseland mean opacity for DT plasma.
    
    At high T (fully ionized), dominated by free-free absorption:
    
    κ_R ≈ 0.4 * ρ * T^(-3.5) [cm²/g]  (simplified Kramers for H/He)
    
    At ICF temperatures (T > 10 keV), the opacity is very low
    and the hot spot is nearly transparent.
    
    Returns:
        Rosseland mean opacity [cm²/g]
    """
    if T_keV <= 0 or rho <= 0:
        return 1e-10  # Minimum opacity
    
    # Temperature in keV
    T = max(T_keV, 0.1)  # Prevent divide by zero
    
    # Simplified Kramers opacity for fully ionized light elements
    # At T = 50 keV, ρ = 100 g/cm³: κ ≈ 0.4 * 100 * 50^(-3.5) ≈ 0.0007 cm²/g
    kappa_ff = 0.4 * rho * T**(-3.5)
    
    # Minimum opacity (prevents κ → 0 at very high T)
    kappa = max(kappa_ff, 1e-6)
    
    return kappa


def optical_depth(rho: float, T_keV: float, R: float, Z_eff: float = 1.0) -> float:
    """
    Estimate optical depth of hot spot.
    
    τ = κ_R * ρ * R
    
    where R is the characteristic size.
    
    Args:
        rho: Density [g/cm³]
        T_keV: Temperature [keV]
        R: Radius [cm]
        Z_eff: Effective charge
    
    Returns:
        Optical depth (dimensionless)
    """
    kappa = rosseland_opacity(rho, T_keV, Z_eff)
    tau = kappa * rho * R
    return tau


def radiation_loss_rate(rho: float, T_keV: float, R: float, 
                        Z_eff: float = 1.0) -> Tuple[float, float]:
    """
    Net radiation energy loss rate accounting for trapping.
    
    In optically thin limit (τ << 1): all radiation escapes
    In optically thick limit (τ >> 1): radiation is trapped,
        only escapes from surface at blackbody rate
    
    For ICF hot spots, τ is typically >> 1, so surface losses dominate.
    
    Args:
        rho: Density [g/cm³]
        T_keV: Temperature [keV]
        R: Hot spot radius [cm]
        Z_eff: Effective charge
    
    Returns:
        (P_loss, tau): Power loss density [erg/cm³/s] and optical depth
    """
    # Optically thin emission
    P_thin = bremsstrahlung_simplified(rho, T_keV)
    
    # Optical depth
    tau = optical_depth(rho, T_keV, R, Z_eff)
    
    # Volume
    V = (4/3) * np.pi * R**3
    
    if tau < 0.1:
        # Optically thin: all radiation escapes
        return P_thin, tau
    
    elif tau > 1:
        # Optically thick: radiation diffusion / surface emission
        # Surface emission is blackbody-limited
        # Total power = 4πR² * σ * T⁴
        # Power density = P_total / V = 3 * σ * T⁴ / R
        
        T_K = T_keV * 1e3 * EV / K_B
        P_surface = 3 * SIGMA_SB * T_K**4 / R
        
        # The actual loss is the minimum of thin emission and surface limit
        # In practice, surface emission is MUCH smaller at high τ
        return min(P_thin, P_surface), tau
    
    else:
        # Intermediate: interpolate using escape probability
        # P_esc ≈ (1 - exp(-τ)) / τ for slab geometry
        # This smoothly transitions between thin and thick limits
        P_esc = (1 - np.exp(-tau)) / tau
        return P_thin * P_esc, tau


def radiation_diffusion_flux(T1_keV: float, T2_keV: float, 
                              rho: float, dr: float) -> float:
    """
    Radiation energy flux in diffusion approximation.
    
    F_rad = -c / (3 * κ_R * ρ) * ∂(aT⁴)/∂r
    
    where a = 4σ_SB/c is the radiation constant.
    
    Args:
        T1_keV, T2_keV: Temperatures at adjacent zones [keV]
        rho: Average density [g/cm³]
        dr: Zone spacing [cm]
    
    Returns:
        Radiation flux [erg/cm²/s]
    """
    if dr <= 0 or rho <= 0:
        return 0.0
    
    T_avg_keV = 0.5 * (T1_keV + T2_keV)
    kappa = rosseland_opacity(rho, T_avg_keV)
    
    # Radiation constant
    a_rad = 4 * SIGMA_SB / C
    
    # Temperature gradient (in K)
    T1_K = T1_keV * 1e3 * EV / K_B
    T2_K = T2_keV * 1e3 * EV / K_B
    
    # Flux
    D = C / (3 * kappa * rho)  # Diffusion coefficient
    dT4_dr = (T2_K**4 - T1_K**4) / dr
    
    F_rad = -D * a_rad * dT4_dr
    
    return F_rad


class RadiationModel:
    """
    Radiation loss model for ICF hot spot.
    
    Combines Bremsstrahlung emission with opacity effects
    to give net radiation loss that can be subtracted from
    the hot spot energy balance.
    """
    
    def __init__(self, include_trapping: bool = True):
        """
        Args:
            include_trapping: If True, account for optical depth effects
        """
        self.include_trapping = include_trapping
        self.history = []
    
    def compute_loss(self, rho: float, T_keV: float, R_cm: float) -> RadiationState:
        """
        Compute radiation state for given plasma conditions.
        
        Args:
            rho: Hot spot density [g/cm³]
            T_keV: Hot spot temperature [keV]
            R_cm: Hot spot radius [cm]
        
        Returns:
            RadiationState with all computed quantities
        """
        # Basic emission rates
        P_brem = bremsstrahlung_simplified(rho, T_keV)
        P_line = line_radiation_power(rho, T_keV)
        
        # Opacity and optical depth
        kappa_R = rosseland_opacity(rho, T_keV)
        tau = optical_depth(rho, T_keV, R_cm)
        
        # Net loss accounting for trapping
        if self.include_trapping:
            P_loss, _ = radiation_loss_rate(rho, T_keV, R_cm)
        else:
            P_loss = P_brem + P_line
        
        # Radiation temperature (assume LTE)
        T_rad = T_keV
        
        # Radiation energy density (if optically thick)
        a_rad = 4 * SIGMA_SB / C
        T_K = T_keV * 1e3 * EV / K_B
        E_rad = a_rad * T_K**4
        
        # Approximate flux (surface emission)
        F_rad = SIGMA_SB * T_K**4
        
        state = RadiationState(
            E_rad=E_rad,
            T_rad=T_rad,
            P_brem=P_brem,
            P_line=P_line,
            kappa_R=kappa_R,
            tau=tau,
            F_rad=F_rad
        )
        
        self.history.append({
            'rho': rho,
            'T_keV': T_keV,
            'R_cm': R_cm,
            'P_loss': P_loss,
            'tau': tau
        })
        
        return state
    
    def energy_loss_rate(self, rho: float, T_keV: float, R_cm: float, 
                          V_cm3: float) -> float:
        """
        Total radiation energy loss rate from hot spot.
        
        Args:
            rho: Density [g/cm³]
            T_keV: Temperature [keV]
            R_cm: Radius [cm]
            V_cm3: Volume [cm³]
        
        Returns:
            Energy loss rate [erg/s]
        """
        P_loss, _ = radiation_loss_rate(rho, T_keV, R_cm)
        return P_loss * V_cm3


def conduction_loss_rate(T_hs_keV: float, T_shell_keV: float,
                          rho: float, R_hs_cm: float, dr_cm: float) -> float:
    """
    Electron thermal conduction loss rate.
    
    For ICF hot spots, electron conduction can also cool the hot spot
    by transporting energy to the cold shell.
    
    Spitzer conductivity:
        κ_e = 1.8e-10 * T^(5/2) / (Z * ln(Λ)) [erg/cm/s/K]
    
    where ln(Λ) ≈ 5-10 is the Coulomb logarithm.
    
    Args:
        T_hs_keV: Hot spot temperature [keV]
        T_shell_keV: Shell temperature [keV]
        rho: Density at interface [g/cm³]
        R_hs_cm: Hot spot radius [cm]
        dr_cm: Interface width [cm]
    
    Returns:
        Conduction power loss [erg/s]
    """
    if T_hs_keV <= T_shell_keV or dr_cm <= 0:
        return 0.0
    
    # Convert to Kelvin
    T_hs_K = T_hs_keV * 1e3 * EV / K_B
    T_shell_K = T_shell_keV * 1e3 * EV / K_B
    
    # Coulomb logarithm (approximate)
    ln_Lambda = 7.0
    
    # Spitzer conductivity at average temperature
    T_avg_K = 0.5 * (T_hs_K + T_shell_K)
    kappa_e = 1.8e-10 * T_avg_K**(2.5) / ln_Lambda  # Z=1 for DT
    
    # Heat flux (with flux limiter to prevent unphysical values)
    dT_dr = (T_hs_K - T_shell_K) / dr_cm
    q_cond = kappa_e * dT_dr
    
    # Flux limiter: q < f * n_e * v_th * k_B * T
    # where f ≈ 0.1 is the flux limit factor
    m_ion = 2.5 * M_P
    n_e = rho / m_ion
    v_th = np.sqrt(K_B * T_hs_K / M_E)
    q_max = 0.1 * n_e * v_th * K_B * T_hs_K
    
    q_cond = min(q_cond, q_max)
    
    # Total power through surface
    A_surface = 4 * np.pi * R_hs_cm**2
    P_cond = q_cond * A_surface
    
    return P_cond


# =============================================================================
# INTEGRATION WITH ICF SIMULATION
# =============================================================================

def add_radiation_to_simulation(sim_state, dt: float, 
                                 radiation_model: RadiationModel = None) -> float:
    """
    Compute radiation cooling for a simulation timestep.
    
    Args:
        sim_state: Simulation state with T_hs, rho_hs, R_hs
        dt: Timestep [s]
        radiation_model: RadiationModel instance
    
    Returns:
        Temperature change due to radiation [keV]
    """
    if radiation_model is None:
        radiation_model = RadiationModel()
    
    # Get state
    T_keV = sim_state.T_hs
    rho = sim_state.rho_hs
    R_cm = sim_state.R_hs * 1e-4  # μm to cm
    V_cm3 = (4/3) * np.pi * R_cm**3
    
    # Energy loss
    E_loss = radiation_model.energy_loss_rate(rho, T_keV, R_cm, V_cm3) * dt
    
    # Convert to temperature change
    # E = (3/2) * N * k_B * T
    # dT/T = dE/E
    m_ion = 2.5 * M_P
    N = rho * V_cm3 / m_ion
    E_thermal = 1.5 * N * T_keV * 1e3 * EV
    
    if E_thermal > 0:
        dT = -T_keV * E_loss / E_thermal
    else:
        dT = 0
    
    return dT


# =============================================================================
# MAIN - DEMONSTRATE RADIATION PHYSICS
# =============================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    print()
    print("╔" + "═" * 58 + "╗")
    print("║  RADIATION PHYSICS FOR ICF                               ║")
    print("╚" + "═" * 58 + "╝")
    print()
    
    # Test radiation at NIF-like conditions
    print("NIF-like hot spot conditions:")
    print("─" * 40)
    
    rho_test = 100  # g/cm³
    T_test = 50     # keV
    R_test = 30e-4  # cm (30 μm)
    
    print(f"  Density:     {rho_test} g/cm³")
    print(f"  Temperature: {T_test} keV")
    print(f"  Radius:      {R_test*1e4:.0f} μm")
    print()
    
    # Compute radiation
    P_brem = bremsstrahlung_simplified(rho_test, T_test)
    kappa = rosseland_opacity(rho_test, T_test)
    tau = optical_depth(rho_test, T_test, R_test)
    P_loss, _ = radiation_loss_rate(rho_test, T_test, R_test)
    
    print("Radiation properties:")
    print("─" * 40)
    print(f"  Bremsstrahlung power: {P_brem:.2e} erg/cm³/s")
    print(f"  Rosseland opacity:    {kappa:.2e} cm²/g")
    print(f"  Optical depth:        {tau:.2f}")
    print(f"  Net loss rate:        {P_loss:.2e} erg/cm³/s")
    print()
    
    # Total power
    V = (4/3) * np.pi * R_test**3
    P_rad_total = P_loss * V
    print(f"  Total power loss: {P_rad_total:.2e} erg/s = {P_rad_total/1e19:.2f} TW")
    print()
    
    # Compare to fusion power at same conditions
    # P_fus ≈ n² * <σv> * E_fus * V
    n = rho_test / (2.5 * M_P)
    from icf_simulation import sigma_v
    sv = sigma_v(T_test)
    E_fus = 17.6e6 * EV
    P_fusion_density = 0.25 * n**2 * sv * E_fus  # per cm³
    P_fusion_total = P_fusion_density * V
    
    print(f"  Fusion power:     {P_fusion_total:.2e} erg/s = {P_fusion_total/1e19:.2f} TW")
    print(f"  Ratio (fusion/rad): {P_fusion_total/P_rad_total:.1f}")
    print()
    
    if P_fusion_total > P_rad_total:
        print("  ✓ Fusion exceeds radiation - self-heating possible!")
    else:
        print("  ✗ Radiation exceeds fusion - no self-heating")
    
    # Plot radiation vs temperature
    print("\nGenerating radiation plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Bremsstrahlung vs temperature
    ax = axes[0, 0]
    T_range = np.logspace(-1, 2, 100)
    P_brem_arr = [bremsstrahlung_simplified(100, T) for T in T_range]
    ax.loglog(T_range, P_brem_arr, 'b-', linewidth=2)
    ax.set_xlabel('Temperature [keV]')
    ax.set_ylabel('Bremsstrahlung Power [erg/cm³/s]')
    ax.set_title('Bremsstrahlung vs Temperature (ρ = 100 g/cm³)')
    ax.grid(True, alpha=0.3)
    
    # 2. Optical depth vs density
    ax = axes[0, 1]
    rho_range = np.logspace(-1, 3, 100)
    tau_arr = [optical_depth(rho, 50, 30e-4) for rho in rho_range]
    ax.loglog(rho_range, tau_arr, 'r-', linewidth=2)
    ax.axhline(y=1, color='k', linestyle='--', label='τ = 1')
    ax.set_xlabel('Density [g/cm³]')
    ax.set_ylabel('Optical Depth τ')
    ax.set_title('Optical Depth vs Density (T=50 keV, R=30 μm)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Net loss with and without trapping
    ax = axes[1, 0]
    rho_range = np.logspace(0, 3, 50)
    P_thin = [bremsstrahlung_simplified(rho, 50) for rho in rho_range]
    P_thick = [radiation_loss_rate(rho, 50, 30e-4)[0] for rho in rho_range]
    
    ax.loglog(rho_range, P_thin, 'b-', linewidth=2, label='Optically thin')
    ax.loglog(rho_range, P_thick, 'r--', linewidth=2, label='With trapping')
    ax.set_xlabel('Density [g/cm³]')
    ax.set_ylabel('Radiation Loss [erg/cm³/s]')
    ax.set_title('Effect of Radiation Trapping')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Ignition condition: fusion vs radiation
    ax = axes[1, 1]
    T_range = np.logspace(0, 2, 100)
    
    for rho in [10, 100, 500]:
        R = 30e-4  # cm
        P_rad = [radiation_loss_rate(rho, T, R)[0] for T in T_range]
        
        n = rho / (2.5 * M_P)
        P_fus = [0.25 * n**2 * sigma_v(T) * E_fus for T in T_range]
        
        ratio = [pf/pr if pr > 0 else 0 for pf, pr in zip(P_fus, P_rad)]
        ax.semilogy(T_range, ratio, linewidth=2, label=f'ρ = {rho} g/cm³')
    
    ax.axhline(y=1, color='k', linestyle='--', linewidth=2)
    ax.set_xlabel('Temperature [keV]')
    ax.set_ylabel('Fusion / Radiation Power Ratio')
    ax.set_title('Ignition Condition (Fusion > Radiation)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, 100)
    ax.set_ylim(0.01, 100)
    
    plt.tight_layout()
    plt.savefig('radiation_physics.png', dpi=150, bbox_inches='tight')
    print("Saved: radiation_physics.png")
    plt.close()
    
    print()
    print("=" * 60)
