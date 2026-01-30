"""
Alternative Fusion Fuel Cycles
==============================

Implements multiple fusion fuel combinations beyond D-T:

1. D-D (Deuterium-Deuterium)
   - No tritium required (abundant fuel)
   - Two branches: D+D → T+p or D+D → ³He+n
   - Lower reactivity, higher ignition threshold
   
2. D-³He (Deuterium-Helium-3)
   - "Aneutronic" - primary reaction produces no neutrons
   - Charged products enable direct energy conversion
   - Requires ~5× higher temperature than D-T
   - ³He is scarce on Earth (lunar mining?)
   
3. p-¹¹B (Proton-Boron-11)
   - Truly aneutronic
   - All energy in charged alphas
   - Requires ~20× higher temperature than D-T
   - Very challenging for ICF due to radiation losses

Each fuel has different:
- Reactivity <σv>(T)
- Energy release per reaction
- Radiation losses (Z² dependence)
- Product spectrum

Author: Evan Pieser
Date: 2026
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Callable
import matplotlib.pyplot as plt

# Physical constants
EV = 1.602e-12        # erg
KEV = 1e3 * EV        # erg
MEV = 1e6 * EV        # erg
M_P = 1.673e-24       # g (proton mass)
C = 2.998e10          # cm/s


@dataclass
class FusionReaction:
    """Describes a fusion reaction."""
    name: str
    reactant1: str
    reactant2: str
    products: List[str]
    Q_MeV: float           # Total energy release [MeV]
    E_charged_MeV: float   # Energy in charged particles [MeV]
    E_neutron_MeV: float   # Energy in neutrons [MeV]
    m1_amu: float          # Reactant 1 mass [amu]
    m2_amu: float          # Reactant 2 mass [amu]
    Z1: int                # Reactant 1 charge
    Z2: int                # Reactant 2 charge


# Define fusion reactions
DT_REACTION = FusionReaction(
    name="D-T",
    reactant1="D", reactant2="T",
    products=["⁴He (3.5 MeV)", "n (14.1 MeV)"],
    Q_MeV=17.6,
    E_charged_MeV=3.5,
    E_neutron_MeV=14.1,
    m1_amu=2.0, m2_amu=3.0,
    Z1=1, Z2=1
)

DD_REACTION_1 = FusionReaction(
    name="D-D → T+p",
    reactant1="D", reactant2="D",
    products=["T (1.01 MeV)", "p (3.02 MeV)"],
    Q_MeV=4.03,
    E_charged_MeV=4.03,
    E_neutron_MeV=0,
    m1_amu=2.0, m2_amu=2.0,
    Z1=1, Z2=1
)

DD_REACTION_2 = FusionReaction(
    name="D-D → ³He+n",
    reactant1="D", reactant2="D",
    products=["³He (0.82 MeV)", "n (2.45 MeV)"],
    Q_MeV=3.27,
    E_charged_MeV=0.82,
    E_neutron_MeV=2.45,
    m1_amu=2.0, m2_amu=2.0,
    Z1=1, Z2=1
)

DHE3_REACTION = FusionReaction(
    name="D-³He",
    reactant1="D", reactant2="³He",
    products=["⁴He (3.6 MeV)", "p (14.7 MeV)"],
    Q_MeV=18.3,
    E_charged_MeV=18.3,
    E_neutron_MeV=0,
    m1_amu=2.0, m2_amu=3.0,
    Z1=1, Z2=2
)

PB11_REACTION = FusionReaction(
    name="p-¹¹B",
    reactant1="p", reactant2="¹¹B",
    products=["3 × ⁴He (8.7 MeV total)"],
    Q_MeV=8.7,
    E_charged_MeV=8.7,
    E_neutron_MeV=0,
    m1_amu=1.0, m2_amu=11.0,
    Z1=1, Z2=5
)


def sigma_v_DT(T_keV: float) -> float:
    """
    D-T fusion reactivity [cm³/s].
    
    Bosch-Hale parameterization.
    """
    T = np.clip(T_keV, 0.2, 200.0)
    
    # Bosch-Hale coefficients
    BG = 34.3827
    mrc2 = 1124656  # keV (reduced mass × c²)
    
    C1 = 1.17302e-9
    C2 = 1.51361e-2
    C3 = 7.51886e-2
    C4 = 4.60643e-3
    C5 = 1.35000e-2
    C6 = -1.06750e-4
    C7 = 1.36600e-5
    
    theta = T / (1 - T*(C2 + T*(C4 + T*C6)) / (1 + T*(C3 + T*(C5 + T*C7))))
    xi = (BG**2 / (4*theta))**(1/3)
    
    sigma_v = C1 * theta * np.sqrt(xi / (mrc2 * T**3)) * np.exp(-3*xi)
    
    return sigma_v


def sigma_v_DD(T_keV: float) -> float:
    """
    D-D fusion reactivity (both branches combined) [cm³/s].
    
    Total D-D ≈ 2× each branch at high T.
    """
    T = np.clip(T_keV, 0.2, 200.0)
    
    # Simplified fit (less accurate than D-T)
    # D-D is roughly 100× lower than D-T at 20 keV
    # Peaks around 1000 keV
    BG = 31.4  # Gamow factor
    
    # Approximate parameterization
    sigma_v = 2.33e-14 * T**(-2/3) * np.exp(-18.76 * T**(-1/3))
    
    return sigma_v


def sigma_v_DHe3(T_keV: float) -> float:
    """
    D-³He fusion reactivity [cm³/s].
    
    Peaks at higher temperature than D-T.
    """
    T = np.clip(T_keV, 0.5, 500.0)
    
    # Bosch-Hale style fit
    BG = 68.7508
    mrc2 = 1124572
    
    C1 = 5.51036e-10
    C2 = 6.41918e-3
    C3 = -2.02896e-3
    C4 = -1.91080e-5
    C5 = 1.35776e-4
    
    theta = T / (1 - T*(C2 + T*C4) / (1 + T*(C3 + T*C5)))
    xi = (BG**2 / (4*theta))**(1/3)
    
    sigma_v = C1 * theta * np.sqrt(xi / (mrc2 * T**3)) * np.exp(-3*xi)
    
    return sigma_v


def sigma_v_pB11(T_keV: float) -> float:
    """
    p-¹¹B fusion reactivity [cm³/s].
    
    Has resonance structure. Simplified fit here.
    Requires very high temperatures (100+ keV).
    """
    T = np.clip(T_keV, 10, 1000.0)
    
    # Approximate fit including resonance at ~150 keV
    # Much lower than D-T at all temperatures
    base = 2.1e-14 * T**(-2/3) * np.exp(-148 * T**(-1/3))
    
    # Add resonance contribution
    T_res = 150  # keV
    width = 50   # keV
    resonance = 1.5e-16 * np.exp(-((T - T_res)/width)**2)
    
    sigma_v = base + resonance
    
    return max(sigma_v, 1e-30)


def bremsstrahlung_loss(rho: float, T_keV: float, Z_eff: float) -> float:
    """
    Bremsstrahlung power density [erg/cm³/s].
    
    P_brem ∝ n_e² × Z_eff² × T^(1/2)
    
    Higher Z fuels have much higher radiation losses!
    """
    if T_keV <= 0 or rho <= 0:
        return 0.0
    
    # Approximate mass per ion (varies by fuel)
    m_ion = 2.5 * M_P  # Use D-T average as baseline
    n_i = rho / m_ion
    n_e = Z_eff * n_i  # For charge neutrality
    
    T_K = T_keV * 1e3 * EV / (1.381e-16)  # Convert to K
    
    # Gaunt factor
    g_ff = 1.2
    
    # Bremsstrahlung (scales as Z_eff² from both electron density and Z dependence)
    P_brem = 1.42e-27 * g_ff * Z_eff**2 * n_e * n_i * np.sqrt(T_K)
    
    return P_brem


@dataclass
class FuelState:
    """State for a fuel cycle simulation."""
    t: float           # Time [ns]
    T: float           # Temperature [keV]
    rho: float         # Density [g/cm³]
    R: float           # Radius [cm]
    n1: float          # Reactant 1 number density [cm⁻³]
    n2: float          # Reactant 2 number density [cm⁻³]
    P_fusion: float    # Fusion power [TW]
    P_brem: float      # Bremsstrahlung power [TW]
    yield_MJ: float    # Cumulative yield [MJ]


class AlternativeFuel:
    """
    Simulation of alternative fusion fuel cycles.
    
    Uses simplified hot spot model to compare different fuels.
    """
    
    def __init__(self, reaction: FusionReaction, 
                 sigma_v_func: Callable[[float], float],
                 initial_T_keV: float = 50.0,
                 initial_rho: float = 100.0,
                 R_cm: float = 30e-4):
        """
        Args:
            reaction: FusionReaction describing the fuel
            sigma_v_func: Reactivity function σv(T)
            initial_T_keV: Initial temperature [keV]
            initial_rho: Initial density [g/cm³]
            R_cm: Hot spot radius [cm]
        """
        self.reaction = reaction
        self.sigma_v = sigma_v_func
        self.T = initial_T_keV
        self.rho = initial_rho
        self.R = R_cm
        
        # Compute number densities (assume 50-50 mix)
        m_avg = 0.5 * (reaction.m1_amu + reaction.m2_amu) * M_P
        n_total = self.rho / m_avg
        self.n1 = 0.5 * n_total
        self.n2 = 0.5 * n_total
        
        # Effective charge for Bremsstrahlung
        self.Z_eff = 0.5 * (reaction.Z1 + reaction.Z2)
        
        self.t = 0
        self.yield_MJ = 0
        self.history: List[FuelState] = []
        
    def step(self, dt_ns: float) -> FuelState:
        """Advance simulation by one timestep."""
        V_cm3 = (4/3) * np.pi * self.R**3
        
        # Fusion power
        sv = self.sigma_v(self.T)
        reaction_rate = self.n1 * self.n2 * sv  # per cm³ per s
        E_per_reaction = self.reaction.Q_MeV * MEV  # erg
        P_fusion = reaction_rate * E_per_reaction * V_cm3  # erg/s
        P_fusion_TW = P_fusion / 1e19
        
        # Bremsstrahlung loss
        P_brem_density = bremsstrahlung_loss(self.rho, self.T, self.Z_eff)
        P_brem = P_brem_density * V_cm3
        P_brem_TW = P_brem / 1e19
        
        # Alpha (charged particle) heating
        # Fraction of energy that stays in plasma
        f_alpha = self.reaction.E_charged_MeV / self.reaction.Q_MeV
        P_heat = P_fusion * f_alpha
        
        # Net heating/cooling
        P_net = P_heat - P_brem
        
        # Temperature change
        # E_thermal = (3/2) * n * k_B * T * V
        n_total = self.n1 + self.n2
        E_thermal = 1.5 * n_total * self.T * KEV * V_cm3
        
        if E_thermal > 0:
            dE = P_net * (dt_ns * 1e-9)
            dT = self.T * dE / E_thermal
            self.T += dT
            self.T = np.clip(self.T, 1.0, 500.0)  # Physical limits
        
        # Yield
        dY = P_fusion * (dt_ns * 1e-9) / 1e13  # MJ
        self.yield_MJ += dY
        
        # Fuel depletion (burn-up)
        dn = reaction_rate * (dt_ns * 1e-9)
        self.n1 = max(self.n1 - dn, 0)
        self.n2 = max(self.n2 - dn, 0)
        
        self.t += dt_ns
        
        state = FuelState(
            t=self.t,
            T=self.T,
            rho=self.rho,
            R=self.R,
            n1=self.n1,
            n2=self.n2,
            P_fusion=P_fusion_TW,
            P_brem=P_brem_TW,
            yield_MJ=self.yield_MJ
        )
        self.history.append(state)
        return state
    
    def run(self, t_end_ns: float, dt_ns: float = 0.01) -> List[FuelState]:
        """Run simulation."""
        while self.t < t_end_ns:
            self.step(dt_ns)
        return self.history
    
    def ignition_margin(self) -> float:
        """
        Compute ignition margin: P_fusion / P_brem.
        
        Ignition requires this ratio > 1.
        """
        if self.history:
            last = self.history[-1]
            if last.P_brem > 0:
                return last.P_fusion / last.P_brem
        return 0


def compare_fuel_cycles():
    """
    Compare all fuel cycles at the same conditions.
    """
    print()
    print("╔" + "═" * 62 + "╗")
    print("║  ALTERNATIVE FUSION FUEL CYCLES                              ║")
    print("║  Comparison at Hot Spot Conditions                           ║")
    print("╚" + "═" * 62 + "╝")
    print()
    
    # Define fuels
    fuels = [
        ("D-T", DT_REACTION, sigma_v_DT),
        ("D-D", DD_REACTION_1, sigma_v_DD),
        ("D-³He", DHE3_REACTION, sigma_v_DHe3),
        ("p-¹¹B", PB11_REACTION, sigma_v_pB11),
    ]
    
    # Test conditions (NIF-like)
    T_test = 50   # keV
    rho_test = 100  # g/cm³
    R_test = 30e-4  # cm
    
    print(f"Test conditions: T = {T_test} keV, ρ = {rho_test} g/cm³, R = {R_test*1e4:.0f} μm")
    print()
    print("─" * 70)
    print(f"{'Fuel':>10}  {'Q [MeV]':>8}  {'<σv>':>12}  {'P_fus [TW]':>12}  {'P_brem [TW]':>12}  {'Margin':>8}")
    print("─" * 70)
    
    results = []
    
    for name, reaction, sv_func in fuels:
        sim = AlternativeFuel(reaction, sv_func, T_test, rho_test, R_test)
        
        # Get instantaneous values
        sv = sv_func(T_test)
        V = (4/3) * np.pi * R_test**3
        
        rate = sim.n1 * sim.n2 * sv
        P_fus = rate * reaction.Q_MeV * MEV * V / 1e19  # TW
        P_brem = bremsstrahlung_loss(rho_test, T_test, sim.Z_eff) * V / 1e19
        
        margin = P_fus / P_brem if P_brem > 0 else float('inf')
        
        results.append({
            'name': name,
            'reaction': reaction,
            'sv_func': sv_func,
            'sv': sv,
            'P_fus': P_fus,
            'P_brem': P_brem,
            'margin': margin
        })
        
        margin_str = f"{margin:.1f}" if margin < 1000 else ">1000"
        print(f"{name:>10}  {reaction.Q_MeV:>8.1f}  {sv:>12.2e}  {P_fus:>12.1f}  {P_brem:>12.1f}  {margin_str:>8}")
    
    print("─" * 70)
    print()
    print("Ignition margin = P_fusion / P_brem (must be > 1 for self-heating)")
    print()
    
    return results


def plot_reactivity_comparison():
    """Plot reactivity curves for all fuels."""
    
    T_range = np.logspace(0, 3, 200)  # 1 to 1000 keV
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Reactivity vs temperature
    ax = axes[0]
    
    sv_DT = [sigma_v_DT(T) for T in T_range]
    sv_DD = [sigma_v_DD(T) for T in T_range]
    sv_DHe3 = [sigma_v_DHe3(T) for T in T_range]
    sv_pB11 = [sigma_v_pB11(T) for T in T_range]
    
    ax.loglog(T_range, sv_DT, 'b-', linewidth=2, label='D-T')
    ax.loglog(T_range, sv_DD, 'g-', linewidth=2, label='D-D')
    ax.loglog(T_range, sv_DHe3, 'r-', linewidth=2, label='D-³He')
    ax.loglog(T_range, sv_pB11, 'm-', linewidth=2, label='p-¹¹B')
    
    # Mark ICF operating region
    ax.axvspan(10, 100, alpha=0.1, color='yellow', label='ICF range')
    
    ax.set_xlabel('Temperature [keV]')
    ax.set_ylabel('Reactivity <σv> [cm³/s]')
    ax.set_title('Fusion Reactivity Comparison')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, 1000)
    ax.set_ylim(1e-26, 1e-14)
    
    # 2. Power balance vs temperature
    ax = axes[1]
    
    rho = 100  # g/cm³
    R = 30e-4  # cm
    V = (4/3) * np.pi * R**3
    
    fuels = [
        ("D-T", DT_REACTION, sigma_v_DT, 'b'),
        ("D-D", DD_REACTION_1, sigma_v_DD, 'g'),
        ("D-³He", DHE3_REACTION, sigma_v_DHe3, 'r'),
        ("p-¹¹B", PB11_REACTION, sigma_v_pB11, 'm'),
    ]
    
    for name, reaction, sv_func, color in fuels:
        margins = []
        for T in T_range:
            m_avg = 0.5 * (reaction.m1_amu + reaction.m2_amu) * M_P
            n = rho / m_avg / 2
            Z_eff = 0.5 * (reaction.Z1 + reaction.Z2)
            
            sv = sv_func(T)
            rate = n * n * sv
            P_fus = rate * reaction.Q_MeV * MEV * V
            P_brem = bremsstrahlung_loss(rho, T, Z_eff) * V
            
            margin = P_fus / P_brem if P_brem > 1 else 1e-10
            margins.append(margin)
        
        ax.semilogy(T_range, margins, color=color, linewidth=2, label=name)
    
    ax.axhline(y=1, color='k', linestyle='--', linewidth=2, label='Ignition threshold')
    ax.set_xlabel('Temperature [keV]')
    ax.set_ylabel('P_fusion / P_bremsstrahlung')
    ax.set_title(f'Ignition Margin (ρ={rho} g/cm³)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, 500)
    ax.set_ylim(1e-3, 1e4)
    
    plt.tight_layout()
    plt.savefig('fuel_cycles_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved: fuel_cycles_comparison.png")
    plt.close()


def simulate_burn(fuel_name: str, T_init: float = 50.0):
    """
    Simulate a burn pulse for a given fuel.
    """
    fuels = {
        "D-T": (DT_REACTION, sigma_v_DT),
        "D-D": (DD_REACTION_1, sigma_v_DD),
        "D-³He": (DHE3_REACTION, sigma_v_DHe3),
        "p-¹¹B": (PB11_REACTION, sigma_v_pB11),
    }
    
    if fuel_name not in fuels:
        print(f"Unknown fuel: {fuel_name}")
        return None
    
    reaction, sv_func = fuels[fuel_name]
    
    print(f"\nSimulating {fuel_name} burn at T_init = {T_init} keV...")
    
    sim = AlternativeFuel(reaction, sv_func, T_init, 100, 30e-4)
    states = sim.run(t_end_ns=1.0, dt_ns=0.001)  # Short burn
    
    final = states[-1]
    margin = sim.ignition_margin()
    
    print(f"  Final T: {final.T:.1f} keV")
    print(f"  Yield: {final.yield_MJ:.4f} MJ")
    print(f"  Ignition margin: {margin:.2f}")
    
    return sim, states


def find_ignition_temperature():
    """
    Find minimum temperature for ignition for each fuel.
    
    Ignition = P_fusion > P_bremsstrahlung
    """
    print()
    print("Finding ignition temperatures...")
    print("─" * 50)
    
    fuels = [
        ("D-T", DT_REACTION, sigma_v_DT),
        ("D-D", DD_REACTION_1, sigma_v_DD),
        ("D-³He", DHE3_REACTION, sigma_v_DHe3),
        ("p-¹¹B", PB11_REACTION, sigma_v_pB11),
    ]
    
    rho = 100  # g/cm³
    R = 30e-4  # cm
    V = (4/3) * np.pi * R**3
    
    for name, reaction, sv_func in fuels:
        # Binary search for ignition temperature
        T_low, T_high = 1, 500
        T_ign = None
        
        for _ in range(50):  # Iterations
            T_mid = 0.5 * (T_low + T_high)
            
            m_avg = 0.5 * (reaction.m1_amu + reaction.m2_amu) * M_P
            n = rho / m_avg / 2
            Z_eff = 0.5 * (reaction.Z1 + reaction.Z2)
            
            sv = sv_func(T_mid)
            rate = n * n * sv
            P_fus = rate * reaction.Q_MeV * MEV * V
            P_brem = bremsstrahlung_loss(rho, T_mid, Z_eff) * V
            
            if P_fus > P_brem:
                T_ign = T_mid
                T_high = T_mid
            else:
                T_low = T_mid
                
            if T_high - T_low < 0.1:
                break
        
        if T_ign:
            print(f"  {name:>8}: T_ignition = {T_ign:.1f} keV")
        else:
            print(f"  {name:>8}: No ignition at ρ = {rho} g/cm³ (T > 500 keV required)")
    
    print()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Compare all fuel cycles
    results = compare_fuel_cycles()
    
    # Find ignition temperatures
    find_ignition_temperature()
    
    # Generate plots
    print("\nGenerating fuel cycle comparison plots...")
    plot_reactivity_comparison()
    
    # Simulate burns
    print()
    print("═" * 62)
    print("BURN SIMULATIONS")
    print("═" * 62)
    
    # D-T at 50 keV (should ignite)
    simulate_burn("D-T", T_init=50)
    
    # D-³He at 100 keV
    simulate_burn("D-³He", T_init=100)
    
    # p-¹¹B at 200 keV (marginal)
    simulate_burn("p-¹¹B", T_init=200)
    
    print()
    print("═" * 62)
    print()
    print("Key findings:")
    print("─" * 50)
    print("• D-T: Easiest to ignite, but produces 14.1 MeV neutrons")
    print("• D-D: 100× lower reactivity, requires higher T")
    print("• D-³He: 'Aneutronic' but needs ~5× higher T than D-T")
    print("• p-¹¹B: Truly aneutronic, but very hard to ignite")
    print("         (radiation losses from Z=5 boron are severe)")
    print()
