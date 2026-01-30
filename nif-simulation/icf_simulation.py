"""
Inertial Confinement Fusion (ICF) Simulation
=============================================

Empirically-calibrated model matching NIF's December 2022 ignition shot.
Uses measured/reported implosion parameters as constraints.

Key NIF parameters (N221204):
- Laser: 2.05 MJ, ~500 TW peak
- Implosion velocity: ~400 km/s
- Convergence ratio: ~30-40
- Hot spot: ~50 μm at stagnation
- Bang time: ~10 ns after laser start
- Yield: 3.15 MJ

Author: Evan Pieser
Date: 2026
"""

import numpy as np
from dataclasses import dataclass
from typing import List

# Physical constants
EV = 1.602e-12       # erg
M_P = 1.673e-24      # g
M_DT = 2.5 * M_P     # g
E_FUSION = 17.6e6 * EV  # erg
ALPHA = 0.2          # alpha particle energy fraction


@dataclass
class State:
    """Implosion state snapshot."""
    t: float          # Time [ns]
    R: float          # Outer shell radius [μm]
    V: float          # Implosion velocity [km/s]
    R_hs: float       # Hot spot radius [μm]
    T_hs: float       # Hot spot temperature [keV]
    rho_hs: float     # Hot spot density [g/cm³]
    P_hs: float       # Hot spot pressure [Gbar]
    yield_MJ: float   # Cumulative yield [MJ]
    CR: float         # Convergence ratio


def sigma_v(T_keV):
    """DT fusion reactivity [cm³/s]."""
    T = np.clip(T_keV, 1.0, 150.0)
    return 3.68e-12 * T**(-2./3.) * np.exp(-19.94 * T**(-1./3.))


class NIF:
    """
    NIF implosion model calibrated to experimental data.
    
    The implosion follows three phases:
    1. Acceleration (0-10 ns): Laser ablates outer shell, drives implosion
    2. Coast (10-12 ns): Shell coasts inward at ~400 km/s
    3. Stagnation (12-13 ns): Hot spot forms, burn occurs
    """
    
    def __init__(self):
        # Initial target geometry
        self.R0 = 1100.0       # Initial outer radius [μm]
        self.R_hs0 = 850.0     # Initial hot spot radius [μm]
        
        # Shell mass (total DT is ~170 μg for NIF)
        self.M_shell = 170.0   # [μg] - DT fuel
        self.M_hs = 10.0       # [μg] - DT gas fill (becomes hot spot)
        
        # Target implosion velocity (from NIF data)
        self.V_imp_target = -400.0  # km/s
        
        self.reset()
        
    def reset(self):
        """Reset to initial conditions."""
        self.t = 0.0           # [ns]
        self.R = self.R0       # [μm]
        self.R_hs = self.R_hs0 # [μm]
        self.V = 0.0           # [km/s]
        self.T_hs = 0.03       # [keV]
        self.yield_MJ = 0.0
        self.burn_fraction = 0.0  # Fraction of fuel burned
        self.stagnated = False
        
    def rho_hs(self):
        """Hot spot density [g/cm³]."""
        V_cm3 = (4./3.) * np.pi * (self.R_hs * 1e-4)**3
        return (self.M_hs * 1e-6) / max(V_cm3, 1e-30)
    
    def P_hs_Gbar(self):
        """Hot spot pressure [Gbar]."""
        # P = 2 * n * k * T (fully ionized)
        # In convenient units: P[Gbar] = 0.154 * ρ[g/cc] * T[keV] / 1000
        return 0.154 * self.rho_hs() * self.T_hs / 1000
    
    def step(self, dt, P_laser_TW):
        """
        Advance simulation by dt nanoseconds.
        """
        # === PHASE 1: ACCELERATION ===
        # Model: V increases toward target velocity during laser pulse
        
        if P_laser_TW > 10 and self.V > self.V_imp_target:
            # Accelerate toward target implosion velocity
            # Use empirical acceleration that gives ~400 km/s in ~8 ns
            P_norm = P_laser_TW / 500.0  # Normalized to peak power
            accel = -80.0 * P_norm  # km/s per ns (empirical)
            
            dV = accel * dt
            self.V = max(self.V + dV, self.V_imp_target)
        
        # === PHASE 2: MOTION ===
        # dR/dt = V
        dR = self.V * dt  # μm (V in km/s, dt in ns, 1 km/s * 1 ns = 1 μm)
        self.R += dR
        self.R = max(self.R, 30.0)  # Minimum radius
        
        # Hot spot compression
        # In ideal implosion: R_hs / R_hs0 ≈ (R / R0)^0.9 (shell compresses more)
        if self.V < 0:
            self.R_hs = self.R_hs0 * (self.R / self.R0)**0.95
            self.R_hs = max(self.R_hs, 25.0)  # NIF hot spot ~25-30 μm at stagnation
        
        # === PHASE 3: COMPRESSION HEATING ===
        # Adiabatic: T ∝ ρ^(γ-1) = ρ^(2/3)
        # ρ ∝ R_hs^(-3), so T ∝ R_hs^(-2) = (R_hs0/R_hs)^2
        
        CR = self.R_hs0 / self.R_hs
        self.T_hs = 0.03 * CR**2
        
        # Additional heating at stagnation (shock heating)
        if self.R_hs < 100 and self.V < -50:
            # Kinetic energy thermalizes
            KE_keV = 0.5 * self.M_shell * 1e-6 * (self.V * 1e5)**2 / (1000 * EV)
            # Small fraction heats hot spot per timestep
            dT_shock = 0.01 * KE_keV / (self.M_hs * 1e-6) * abs(dt / 0.5)
            self.T_hs += dT_shock
        
        self.T_hs = min(self.T_hs, 70.0)  # Cap at 70 keV
        
        # === PHASE 4: STAGNATION DECELERATION ===
        P = self.P_hs_Gbar()
        if P > 0.1 and self.V < 0:
            # Pressure decelerates shell
            A = 4 * np.pi * (self.R_hs * 1e-4)**2  # cm²
            F = P * 1e15 * A  # dyn
            a = F / (self.M_shell * 1e-6)  # cm/s²
            dV = a * dt * 1e-9 * 1e-5  # Convert to km/s change per ns
            self.V = min(self.V + dV, 0)  # Decelerate, don't reverse
        
        # === PHASE 5: FUSION BURN ===
        if self.T_hs > 4 and self.burn_fraction < 0.35:  # Max ~35% burn-up
            sv = sigma_v(self.T_hs)
            rho = self.rho_hs()
            n = rho / M_DT
            
            # Reaction rate
            R_rate = 0.25 * n**2 * sv  # reactions/cm³/s
            V_hs = (4./3.) * np.pi * (self.R_hs * 1e-4)**3  # cm³
            
            # Power
            P_fus = R_rate * E_FUSION * V_hs  # erg/s
            
            # Yield this step
            dY = P_fus * (dt * 1e-9) / 1e13  # MJ
            self.yield_MJ += dY
            
            # Track burn fraction (reactions consume fuel)
            n_reactions = R_rate * V_hs * (dt * 1e-9)
            fuel_atoms = (self.M_hs * 1e-6) / M_DT
            self.burn_fraction += n_reactions / fuel_atoms
            
            # Alpha heating (bootstrap) - reduced as fuel depletes
            Q_alpha = P_fus * ALPHA * (dt * 1e-9) * (1 - self.burn_fraction)  # erg
            cv = 1.44e13  # erg/(g·keV)
            dT = Q_alpha / (rho * V_hs * cv)
            self.T_hs += min(dT, 5.0)  # Limit alpha heating per step
            
            # Disassembly: hot spot expands after stagnation
            if self.stagnated and self.R_hs < 200:
                # Sound speed expansion
                cs = np.sqrt(self.T_hs * 1e3 * EV * 2 / M_DT) * 1e-5  # km/s
                self.R_hs += 0.3 * cs * dt  # Expand at ~30% sound speed
                
        # Mark stagnation
        if self.R_hs <= 30 and not self.stagnated:
            self.stagnated = True
        
        self.t += dt
        
        return State(
            t=self.t,
            R=self.R,
            V=self.V,
            R_hs=self.R_hs,
            T_hs=self.T_hs,
            rho_hs=self.rho_hs(),
            P_hs=self.P_hs_Gbar(),
            yield_MJ=self.yield_MJ,
            CR=CR
        )
    
    def run(self, t_end_ns, laser_func, dt=0.01):
        """Run simulation to t_end_ns."""
        states = []
        
        while self.t < t_end_ns:
            P = laser_func(self.t)
            s = self.step(dt, P)
            states.append(s)
            
            if len(states) % 50 == 0:
                print(f"  t={s.t:5.2f}ns  R={s.R:6.0f}μm  V={s.V:+7.0f}km/s  "
                      f"R_hs={s.R_hs:5.0f}μm  T={s.T_hs:5.1f}keV  "
                      f"ρ={s.rho_hs:6.0f}g/cc  Y={s.yield_MJ:.3f}MJ")
        
        return states


def nif_pulse(t_ns):
    """
    NIF laser pulse shape [TW].
    Approximately matches shot N221204 (Dec 5, 2022).
    """
    if t_ns < 0:
        return 0
    elif t_ns < 2:
        # Foot pulse
        return 30
    elif t_ns < 5:
        # Rise
        return 30 + 470 * (t_ns - 2) / 3
    elif t_ns < 12:
        # Peak
        return 500
    elif t_ns < 13:
        # Fall
        return 500 * (13 - t_ns)
    else:
        return 0


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print()
    print("╔" + "═" * 58 + "╗")
    print("║  INERTIAL CONFINEMENT FUSION SIMULATION                  ║")
    print("║  National Ignition Facility - December 5, 2022           ║")
    print("╚" + "═" * 58 + "╝")
    print()
    
    sim = NIF()
    
    print(f"  Target capsule:  {sim.R0:.0f} μm outer radius")
    print(f"  DT fuel mass:    {sim.M_shell:.0f} μg shell, {sim.M_hs:.1f} μg hot spot")
    print(f"  Laser energy:    2.05 MJ")
    print(f"  Peak power:      500 TW")
    print()
    print("─" * 60)
    
    states = sim.run(t_end_ns=14, laser_func=nif_pulse, dt=0.01)
    
    print("─" * 60)
    print()
    
    # Find key moments
    peak_rho = max(states, key=lambda s: s.rho_hs)
    peak_T = max(states, key=lambda s: s.T_hs)
    min_V = min(states, key=lambda s: s.V)
    final = states[-1]
    
    Q = final.yield_MJ / 2.05
    
    print("  RESULTS")
    print("  " + "─" * 40)
    print(f"  Peak implosion velocity:  {min_V.V:.0f} km/s")
    print(f"  Stagnation time:          {peak_rho.t:.2f} ns")
    print(f"  Convergence ratio:        {peak_rho.CR:.0f}×")
    print(f"  Minimum hot spot radius:  {peak_rho.R_hs:.0f} μm")
    print(f"  Peak hot spot temperature:{peak_T.T_hs:.1f} keV")
    print(f"  Peak hot spot density:    {peak_rho.rho_hs:.0f} g/cm³")
    print(f"  Peak pressure:            {peak_rho.P_hs:.1f} Gbar")
    print(f"  Total fusion yield:       {final.yield_MJ:.2f} MJ")
    print(f"  Scientific gain (Q):      {Q:.2f}")
    print()
    
    if Q > 1:
        print("  ╔════════════════════════════════════════════════╗")
        print("  ║                                                ║")
        print("  ║          ★ ★ ★  IGNITION  ★ ★ ★              ║")
        print("  ║                                                ║")
        print("  ║   For the first time in history, a fusion     ║")
        print("  ║   reaction produced more energy than the      ║")
        print("  ║   laser energy used to initiate it.           ║")
        print("  ║                                                ║")
        print("  ║   This is the moment humanity proved it       ║")
        print("  ║   could create a star.                        ║")
        print("  ║                                                ║")
        print("  ╚════════════════════════════════════════════════╝")
    else:
        print(f"  Gain Q = {Q:.2f} (ignition requires Q > 1)")
    
    print()
    print("═" * 60)
