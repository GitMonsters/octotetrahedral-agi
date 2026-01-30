"""
Manufacturing Defects Model for ICF Capsules
=============================================

Models various manufacturing defects that degrade ICF implosion performance:

1. Fill Tube
   - 10 μm diameter glass tube for DT gas fill
   - Seeds ~100 μm diameter perturbation
   - Creates high-density jet during implosion
   - ~5-10% yield degradation

2. Capsule Support (Tent/Stalk)
   - Thin (~50 nm) membrane to position capsule
   - Seeds low-mode (l~6) perturbation
   - ~2-5% yield degradation

3. Surface Roughness
   - Outer surface: ~1 μm RMS (HDC ablator)
   - Inner surface: ~0.5 μm RMS (DT ice)
   - Seeds multi-mode RT instabilities

4. Ice Layer Non-uniformity
   - DT ice layer should be spherical
   - Grooves, bubbles, crystal defects
   - Seeds low-mode perturbations

5. Ablator Defects
   - Voids, inclusions, delaminations
   - High-Z dopant non-uniformity
   - Localized opacity variations

Each defect is modeled as:
- Initial perturbation amplitude
- Mode spectrum (l values)
- Growth during implosion
- Yield degradation factor

Author: Evan Pieser
Date: 2026
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
from scipy.special import legendre


@dataclass
class Defect:
    """Base class for a manufacturing defect."""
    name: str
    description: str
    amplitude_um: float         # Initial perturbation amplitude [μm]
    modes: List[int]            # Spherical harmonic modes seeded
    azimuthal_modes: List[int]  # m values for each l (0 for axisymmetric)
    growth_rate: float          # Relative growth rate factor
    yield_degradation: float    # Direct yield degradation factor


class FillTube:
    """
    Fill tube defect model.
    
    The fill tube is a 5-10 μm glass capillary used to fill the capsule
    with DT gas. It creates a localized high-density perturbation that
    grows into a jet during implosion.
    """
    def __init__(self, diameter_um: float = 10.0, length_um: float = 500.0,
                 position_theta: float = 0.0):
        self.diameter_um = diameter_um
        self.length_um = length_um
        self.position_theta = position_theta
        
        self.name = "Fill Tube"
        self.description = f"{self.diameter_um:.0f} μm diameter fill tube"
        # Fill tube seeds modes l~50-100 (localized perturbation)
        self.modes = list(range(30, 120, 5))
        self.azimuthal_modes = [0] * len(self.modes)
        self.amplitude_um = 2 * self.diameter_um  # Effective perturbation
        self.growth_rate = 2.0  # Grows faster than random roughness
        self.yield_degradation = 0.05  # ~5% direct degradation


class TentSupport:
    """
    Tent/membrane support defect.
    
    A thin (~50 nm) plastic membrane holds the capsule in place.
    It burns through early but seeds a large-scale perturbation.
    """
    def __init__(self, thickness_nm: float = 50.0, attachment_angle: float = 45.0):
        self.thickness_nm = thickness_nm
        self.attachment_angle = attachment_angle
        
        self.name = "Tent Support"
        self.description = f"{self.thickness_nm:.0f} nm membrane support"
        # Tent seeds l=6-20 (large scale)
        self.modes = [6, 8, 10, 12, 14, 16, 18, 20]
        self.azimuthal_modes = [0] * len(self.modes)
        self.amplitude_um = 0.5 * (self.thickness_nm / 50.0)  # Scales with thickness
        self.growth_rate = 1.5
        self.yield_degradation = 0.03


class SurfaceRoughness:
    """
    Surface roughness model.
    
    Both outer (ablator) and inner (DT ice) surfaces have roughness
    that seeds RT instabilities at all mode numbers.
    """
    def __init__(self, outer_rms_um: float = 1.0, inner_rms_um: float = 0.5,
                 spectrum_type: str = "NIF"):
        self.outer_rms_um = outer_rms_um
        self.inner_rms_um = inner_rms_um
        self.spectrum_type = spectrum_type
        
        self.name = "Surface Roughness"
        self.description = f"RMS: outer={self.outer_rms_um:.1f}μm, inner={self.inner_rms_um:.1f}μm"
        # Surface roughness seeds l=2 to 200
        self.modes = list(range(2, 201))
        self.azimuthal_modes = [0] * len(self.modes)  # Simplified to axisymmetric
        self.amplitude_um = np.sqrt(self.outer_rms_um**2 + self.inner_rms_um**2)
        self.growth_rate = 1.0  # Standard RT growth
        self.yield_degradation = 0.0  # Degradation from RT mix, not direct
        
    def mode_amplitudes(self) -> np.ndarray:
        """Generate amplitude spectrum for surface roughness."""
        amps = []
        for l in self.modes:
            if self.spectrum_type == "NIF":
                # NIF-like spectrum: peaks around l=10-20, falls as l^-1
                if l < 10:
                    amp = self.outer_rms_um * 0.1 / np.sqrt(len(self.modes))
                else:
                    amp = self.outer_rms_um * (10/l) / np.sqrt(len(self.modes))
            elif self.spectrum_type == "flat":
                amp = self.outer_rms_um / np.sqrt(len(self.modes))
            else:
                amp = self.outer_rms_um / l
            amps.append(amp)
        return np.array(amps)


class IceNonuniformity:
    """
    DT ice layer non-uniformity.
    
    The ice layer should be a perfect sphere, but manufacturing
    produces grooves, bubbles, and grain boundaries.
    """
    def __init__(self, groove_depth_um: float = 2.0, bubble_size_um: float = 5.0,
                 inner_surface_rms_um: float = 0.5):
        self.groove_depth_um = groove_depth_um
        self.bubble_size_um = bubble_size_um
        self.inner_surface_rms_um = inner_surface_rms_um
        
        self.name = "Ice Non-uniformity"
        self.description = f"Groove depth: {self.groove_depth_um:.1f}μm"
        # Grooves typically seed low modes
        self.modes = [2, 4, 6, 8, 10, 12]
        self.azimuthal_modes = [0] * len(self.modes)
        self.amplitude_um = self.groove_depth_um
        self.growth_rate = 1.5
        self.yield_degradation = 0.02


class AblatorDefects:
    """
    Ablator defects (voids, inclusions, dopant variations).
    
    HDC (high-density carbon) ablators can have:
    - Voids from manufacturing
    - High-Z inclusions
    - Dopant concentration variations
    """
    def __init__(self, void_fraction: float = 0.001, void_size_um: float = 5.0,
                 dopant_variation: float = 0.05):
        self.void_fraction = void_fraction
        self.void_size_um = void_size_um
        self.dopant_variation = dopant_variation
        
        self.name = "Ablator Defects"
        self.description = f"Void fraction: {self.void_fraction*100:.2f}%, dopant var: {self.dopant_variation*100:.1f}%"
        # Voids seed high modes; dopant variations seed low modes
        self.modes = list(range(4, 100, 2))
        self.azimuthal_modes = [0] * len(self.modes)
        self.amplitude_um = self.void_size_um * np.sqrt(self.void_fraction * 1000)
        self.growth_rate = 1.2
        self.yield_degradation = 0.01


class DefectModel:
    """
    Combined model for all manufacturing defects.
    
    Tracks each defect's contribution and computes total yield degradation.
    """
    
    def __init__(self, include_fill_tube: bool = True,
                 include_tent: bool = True,
                 include_roughness: bool = True,
                 include_ice: bool = True,
                 include_ablator: bool = True):
        """
        Args:
            include_*: Flags to enable/disable each defect type
        """
        self.defects: List[Defect] = []
        
        if include_fill_tube:
            self.defects.append(FillTube())
        if include_tent:
            self.defects.append(TentSupport())
        if include_roughness:
            self.defects.append(SurfaceRoughness())
        if include_ice:
            self.defects.append(IceNonuniformity())
        if include_ablator:
            self.defects.append(AblatorDefects())
            
        self.history = []
    
    def add_defect(self, defect: Defect):
        """Add a custom defect."""
        self.defects.append(defect)
    
    def total_direct_degradation(self) -> float:
        """
        Compute total direct yield degradation from all defects.
        
        Degradations combine multiplicatively.
        """
        degradation = 1.0
        for defect in self.defects:
            degradation *= (1 - defect.yield_degradation)
        return 1 - degradation
    
    def compute_perturbation_spectrum(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute combined perturbation amplitude spectrum.
        
        Returns:
            (l_values, amplitudes) arrays
        """
        # Collect all modes
        all_modes = set()
        for defect in self.defects:
            all_modes.update(defect.modes)
        l_values = np.array(sorted(all_modes))
        
        # Sum amplitudes in quadrature
        amplitudes = np.zeros_like(l_values, dtype=float)
        for defect in self.defects:
            for i, l in enumerate(l_values):
                if l in defect.modes:
                    idx = defect.modes.index(l)
                    if isinstance(defect, SurfaceRoughness):
                        amp = defect.mode_amplitudes()[idx]
                    else:
                        # Distribute amplitude across modes
                        amp = defect.amplitude_um / np.sqrt(len(defect.modes))
                    amplitudes[i] = np.sqrt(amplitudes[i]**2 + amp**2)
        
        return l_values, amplitudes
    
    def evolve_perturbations(self, CR: float, t_ns: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evolve perturbation amplitudes during implosion.
        
        Simple model: amplitudes grow as CR^α where α depends on mode.
        Bell-Plesset: α ≈ 1 for convergent compression
        
        Args:
            CR: Convergence ratio (R0/R)
            t_ns: Time [ns]
            
        Returns:
            (l_values, grown_amplitudes)
        """
        l_values, amp_0 = self.compute_perturbation_spectrum()
        
        # Growth model:
        # Low modes (l < 10): grow as CR^1.0
        # Mid modes (10-50): grow as CR^0.8
        # High modes (l > 50): stabilized, grow as CR^0.5
        grown_amps = np.zeros_like(amp_0)
        
        for i, l in enumerate(l_values):
            if l < 10:
                alpha = 1.0
            elif l < 50:
                alpha = 0.8
            else:
                alpha = 0.5
            
            grown_amps[i] = amp_0[i] * CR**alpha
        
        self.history.append({
            't': t_ns,
            'CR': CR,
            'l_values': l_values.copy(),
            'amplitudes': grown_amps.copy()
        })
        
        return l_values, grown_amps
    
    def mix_degradation(self, R_hs_um: float, shell_thickness_um: float) -> float:
        """
        Compute yield degradation from mix.
        
        Mix fraction based on perturbation amplitudes vs shell dimensions.
        """
        if not self.history:
            return 0.0
        
        last = self.history[-1]
        if len(last['amplitudes']) == 0:
            return 0.0
            
        amp_max = np.max(last['amplitudes'])
        
        # Mix fraction ~ amplitude / shell_thickness
        # Use softer scaling - not all perturbation penetrates
        effective_amp = amp_max * 0.2  # Only 20% penetrates as mix
        mix_frac = min(effective_amp / shell_thickness_um, 0.5)  # Cap at 50%
        
        # Yield degradation ~ (1 - mix_frac)²
        degradation = 1 - (1 - mix_frac)**2
        
        return degradation
    
    def total_degradation(self, R_hs_um: float = 30, 
                          shell_thickness_um: float = 50) -> float:
        """
        Compute total yield degradation from all sources.
        """
        direct = self.total_direct_degradation()
        mix = self.mix_degradation(R_hs_um, shell_thickness_um)
        
        # Combine (not exceeding 90%)
        total = 1 - (1 - direct) * (1 - mix)
        return min(total, 0.9)
    
    def report(self):
        """Print defect summary."""
        print("\nDefect Model Summary:")
        print("─" * 60)
        
        for defect in self.defects:
            print(f"  {defect.name:20s}: {defect.description}")
            print(f"      Amplitude: {defect.amplitude_um:.2f} μm")
            print(f"      Modes: l = {min(defect.modes)}-{max(defect.modes)}")
            print(f"      Direct degradation: {defect.yield_degradation*100:.1f}%")
            print()
        
        print(f"Total direct degradation: {self.total_direct_degradation()*100:.1f}%")
        print("─" * 60)
    
    def plot_spectrum(self, save_path: str = None):
        """Plot perturbation spectrum."""
        l_values, amplitudes = self.compute_perturbation_spectrum()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.semilogy(l_values, amplitudes, 'b-', linewidth=2)
        ax.fill_between(l_values, amplitudes, alpha=0.3)
        
        ax.set_xlabel('Mode Number l')
        ax.set_ylabel('Amplitude [μm]')
        ax.set_title('Combined Defect Perturbation Spectrum')
        ax.grid(True, alpha=0.3)
        
        # Mark key defect contributions
        colors = ['red', 'green', 'orange', 'purple', 'brown']
        for i, defect in enumerate(self.defects):
            l_center = np.mean(defect.modes)
            ax.axvline(x=l_center, color=colors[i % len(colors)], 
                      linestyle='--', alpha=0.5, label=defect.name)
        
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.close()
        return fig


def simulate_defect_impact():
    """
    Simulate implosion with manufacturing defects.
    """
    print()
    print("╔" + "═" * 60 + "╗")
    print("║  MANUFACTURING DEFECTS SIMULATION                           ║")
    print("║  Impact on NIF Implosion Performance                        ║")
    print("╚" + "═" * 60 + "╝")
    print()
    
    # Create defect model with all defects
    model = DefectModel(
        include_fill_tube=True,
        include_tent=True,
        include_roughness=True,
        include_ice=True,
        include_ablator=True
    )
    
    model.report()
    
    # Simulate implosion trajectory
    print("\nSimulating implosion with defect growth...")
    print("─" * 60)
    
    R0 = 1100  # μm
    CR_values = np.linspace(1, 35, 100)  # Convergence ratio
    
    results = []
    for CR in CR_values:
        R = R0 / CR
        t = (CR - 1) * 0.3  # Approximate time
        
        l_vals, amps = model.evolve_perturbations(CR, t)
        amp_max = np.max(amps)
        amp_rms = np.sqrt(np.mean(amps**2))
        
        results.append({
            'CR': CR,
            't': t,
            'R': R,
            'amp_max': amp_max,
            'amp_rms': amp_rms
        })
        
        if int(CR) in [1, 5, 10, 20, 30]:
            print(f"  CR = {CR:4.0f}×: R = {R:5.0f} μm, "
                  f"amp_max = {amp_max:5.1f} μm, amp_rms = {amp_rms:5.2f} μm")
    
    print("─" * 60)
    print()
    
    # Final state
    final_CR = 30
    shell_thickness = 50  # μm at stagnation
    
    total_deg = model.total_degradation(R_hs_um=30, shell_thickness_um=shell_thickness)
    
    print("  DEGRADATION ANALYSIS")
    print("  " + "─" * 40)
    print(f"  Direct defect degradation: {model.total_direct_degradation()*100:.1f}%")
    print(f"  Mix degradation:           {model.mix_degradation(30, shell_thickness)*100:.1f}%")
    print(f"  Total degradation:         {total_deg*100:.1f}%")
    print()
    
    ideal_yield = 2.88  # MJ
    actual_yield = ideal_yield * (1 - total_deg)
    Q_ideal = ideal_yield / 2.05
    Q_actual = actual_yield / 2.05
    
    print(f"  Ideal yield:    {ideal_yield:.2f} MJ  (Q = {Q_ideal:.2f})")
    print(f"  Actual yield:   {actual_yield:.2f} MJ  (Q = {Q_actual:.2f})")
    print()
    
    if Q_actual > 1:
        print("  ✓ Despite defects, ignition is achieved!")
    else:
        print("  ✗ Defects prevent ignition")
    
    return model, results


def compare_defect_scenarios():
    """
    Compare yield with different defect combinations.
    """
    print()
    print("╔" + "═" * 60 + "╗")
    print("║  DEFECT SCENARIO COMPARISON                                 ║")
    print("╚" + "═" * 60 + "╝")
    print()
    
    scenarios = [
        ("Perfect capsule", False, False, False, False, False),
        ("Fill tube only", True, False, False, False, False),
        ("Roughness only", False, False, True, False, False),
        ("Fill tube + tent", True, True, False, False, False),
        ("All defects", True, True, True, True, True),
    ]
    
    ideal_yield = 2.88
    
    print(f"{'Scenario':30s}  {'Degradation':>12}  {'Yield [MJ]':>12}  {'Q':>8}")
    print("─" * 70)
    
    results = []
    
    for name, ft, tent, rough, ice, abl in scenarios:
        model = DefectModel(
            include_fill_tube=ft,
            include_tent=tent,
            include_roughness=rough,
            include_ice=ice,
            include_ablator=abl
        )
        
        # Simulate to CR=30 (only if defects exist)
        if model.defects:
            for CR in np.linspace(1, 30, 50):
                model.evolve_perturbations(CR, (CR-1)*0.3)
        
        deg = model.total_degradation(30, 50)
        actual = ideal_yield * (1 - deg)
        Q = actual / 2.05
        
        results.append({
            'name': name,
            'degradation': deg,
            'yield': actual,
            'Q': Q
        })
        
        print(f"{name:30s}  {deg*100:>11.1f}%  {actual:>12.2f}  {Q:>8.2f}")
    
    print("─" * 70)
    print()
    
    # Plot comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    names = [r['name'] for r in results]
    yields = [r['yield'] for r in results]
    
    colors = ['green' if r['Q'] > 1 else 'red' for r in results]
    bars = ax.bar(names, yields, color=colors, alpha=0.7)
    
    ax.axhline(y=2.05, color='k', linestyle='--', linewidth=2, label='Q=1 threshold')
    ax.set_ylabel('Fusion Yield [MJ]')
    ax.set_title('Impact of Manufacturing Defects on Yield')
    ax.legend()
    
    # Add Q values on bars
    for bar, r in zip(bars, results):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'Q={r["Q"]:.2f}', ha='center', va='bottom', fontsize=10)
    
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig('defect_scenarios.png', dpi=150, bbox_inches='tight')
    print("Saved: defect_scenarios.png")
    plt.close()
    
    return results


def plot_defect_growth():
    """Plot perturbation growth during implosion."""
    
    model = DefectModel()
    
    CR_values = np.array([1, 5, 10, 20, 30])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Spectrum at different CRs
    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(CR_values)))
    
    for CR, color in zip(CR_values, colors):
        l_vals, amps = model.evolve_perturbations(CR, (CR-1)*0.3)
        ax.semilogy(l_vals, amps, '-', color=color, linewidth=2, label=f'CR={CR:.0f}')
    
    ax.set_xlabel('Mode Number l')
    ax.set_ylabel('Amplitude [μm]')
    ax.set_title('Perturbation Spectrum Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 120)
    
    # 2. Max amplitude vs CR
    ax = axes[1]
    
    model2 = DefectModel()
    CR_fine = np.linspace(1, 35, 100)
    amp_max = []
    amp_rms = []
    
    for CR in CR_fine:
        l_vals, amps = model2.evolve_perturbations(CR, (CR-1)*0.3)
        amp_max.append(np.max(amps))
        amp_rms.append(np.sqrt(np.mean(amps**2)))
    
    ax.semilogy(CR_fine, amp_max, 'b-', linewidth=2, label='Max amplitude')
    ax.semilogy(CR_fine, amp_rms, 'r--', linewidth=2, label='RMS amplitude')
    ax.axhline(y=50, color='k', linestyle=':', label='Shell thickness')
    
    ax.set_xlabel('Convergence Ratio')
    ax.set_ylabel('Perturbation Amplitude [μm]')
    ax.set_title('Perturbation Growth During Implosion')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('defect_growth.png', dpi=150, bbox_inches='tight')
    print("Saved: defect_growth.png")
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Run defect impact simulation
    model, results = simulate_defect_impact()
    
    # Compare scenarios
    print()
    scenario_results = compare_defect_scenarios()
    
    # Generate plots
    print("\nGenerating defect analysis plots...")
    model.plot_spectrum('defect_spectrum.png')
    plot_defect_growth()
    
    print()
    print("═" * 62)
    print()
    print("Key findings:")
    print("─" * 50)
    print("• Fill tube contributes ~5% direct degradation")
    print("• Surface roughness drives RT mix at high modes")
    print("• Combined defects can cause 10-20% yield loss")
    print("• NIF's success required mitigating all defect sources")
    print("• Future gains possible with improved manufacturing")
    print()
