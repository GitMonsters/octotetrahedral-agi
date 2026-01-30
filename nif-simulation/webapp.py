"""
Interactive ICF Simulation Web Application
==========================================

A browser-based visualization of the NIF ICF simulation with
adjustable parameters using Streamlit.

Features:
- Adjustable laser energy, power, and pulse shape
- Real-time simulation visualization
- Parameter space exploration
- Animated implosion sequence

To run:
    streamlit run webapp.py

Author: Evan Pieser
Date: 2026
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge
from matplotlib.collections import PatchCollection
import matplotlib.colors as mcolors

# Import our simulation modules
from icf_simulation import NIF, nif_pulse, State, sigma_v

# Page config
st.set_page_config(
    page_title="NIF ICF Simulation",
    page_icon="☢️",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stMetric {
        background-color: #1e1e1e;
        padding: 10px;
        border-radius: 5px;
    }
    .big-font {
        font-size: 24px !important;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


def create_pulse_function(energy_MJ: float, peak_power_TW: float, 
                          pulse_type: str) -> callable:
    """Create laser pulse function based on parameters."""
    
    # Scale factor to achieve desired energy
    # Baseline NIF pulse delivers ~2.05 MJ
    energy_scale = energy_MJ / 2.05
    power_scale = peak_power_TW / 500
    
    # Adjust duration to match energy at given power
    duration_scale = energy_scale / power_scale
    
    if pulse_type == "NIF Standard":
        def pulse(t_ns):
            t_scaled = t_ns / duration_scale
            if t_scaled < 0:
                return 0
            elif t_scaled < 2:
                return 30 * power_scale
            elif t_scaled < 5:
                return (30 + 470 * (t_scaled - 2) / 3) * power_scale
            elif t_scaled < 12:
                return peak_power_TW
            elif t_scaled < 13:
                return peak_power_TW * (13 - t_scaled)
            else:
                return 0
        return pulse, 14 * duration_scale
        
    elif pulse_type == "Square":
        duration = energy_MJ / (peak_power_TW * 1e-6)  # ns
        def pulse(t_ns):
            return peak_power_TW if 0 <= t_ns <= duration else 0
        return pulse, duration * 1.5
        
    elif pulse_type == "High Foot":
        def pulse(t_ns):
            t_scaled = t_ns / duration_scale
            if t_scaled < 0:
                return 0
            elif t_scaled < 4:
                return 100 * power_scale
            elif t_scaled < 6:
                return (100 + 400 * (t_scaled - 4) / 2) * power_scale
            elif t_scaled < 11:
                return peak_power_TW
            elif t_scaled < 12:
                return peak_power_TW * (12 - t_scaled)
            else:
                return 0
        return pulse, 14 * duration_scale
    
    else:  # Adiabat-shaped
        def pulse(t_ns):
            t_scaled = t_ns / duration_scale
            if t_scaled < 0 or t_scaled > 14:
                return 0
            elif t_scaled < 10:
                return peak_power_TW * (t_scaled / 10)**2
            elif t_scaled < 12:
                return peak_power_TW
            else:
                return peak_power_TW * (14 - t_scaled) / 2
        return pulse, 16 * duration_scale


def run_simulation(energy_MJ: float, peak_power_TW: float, 
                   pulse_type: str, target_scale: float):
    """Run simulation with given parameters."""
    
    sim = NIF()
    
    # Scale target
    sim.R0 *= target_scale
    sim.R_hs0 *= target_scale
    sim.M_shell *= target_scale**3
    sim.M_hs *= target_scale**3
    
    # Create pulse
    pulse, t_end = create_pulse_function(energy_MJ, peak_power_TW, pulse_type)
    
    # Run simulation
    sim.reset()
    states = []
    dt = 0.02
    
    while sim.t < t_end:
        P = pulse(sim.t)
        s = sim.step(dt, P)
        states.append(s)
    
    return states, sim


def plot_implosion_diagram(R_um: float, R_hs_um: float, T_keV: float,
                           rho_gcc: float, V_kms: float, t_ns: float):
    """Create a visual diagram of the capsule state."""
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal')
    
    # Normalize for display
    R_max = 1200
    
    # Color based on temperature
    T_norm = min(T_keV / 70, 1.0)
    hs_color = plt.cm.hot(0.3 + 0.7 * T_norm)
    
    # Shell color based on density
    rho_norm = min(rho_gcc / 200, 1.0)
    shell_color = plt.cm.Blues(0.3 + 0.7 * rho_norm)
    
    # Draw outer shell
    outer = Circle((0, 0), R_um / R_max, facecolor=shell_color, 
                   edgecolor='black', linewidth=2)
    ax.add_patch(outer)
    
    # Draw hot spot
    inner = Circle((0, 0), R_hs_um / R_max, facecolor=hs_color,
                   edgecolor='orange', linewidth=1)
    ax.add_patch(inner)
    
    # Velocity arrows (if moving)
    if abs(V_kms) > 10:
        arrow_scale = min(abs(V_kms) / 400, 1.0) * 0.3
        for angle in [0, 90, 180, 270]:
            rad = np.radians(angle)
            x = (R_um / R_max + 0.1) * np.cos(rad)
            y = (R_um / R_max + 0.1) * np.sin(rad)
            dx = -arrow_scale * np.cos(rad) * np.sign(V_kms)
            dy = -arrow_scale * np.sin(rad) * np.sign(V_kms)
            ax.arrow(x, y, dx, dy, head_width=0.05, head_length=0.02,
                    fc='red', ec='red')
    
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_title(f't = {t_ns:.2f} ns', fontsize=14)
    ax.axis('off')
    
    # Add legend
    ax.text(0, -1.3, f'R = {R_um:.0f} μm | V = {V_kms:.0f} km/s\n'
                     f'T = {T_keV:.1f} keV | ρ = {rho_gcc:.0f} g/cm³',
            ha='center', fontsize=10)
    
    return fig


def plot_time_evolution(states):
    """Plot time evolution of key parameters."""
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    t = [s.t for s in states]
    
    # Radius
    ax = axes[0, 0]
    ax.plot(t, [s.R for s in states], 'b-', linewidth=2)
    ax.set_xlabel('Time [ns]')
    ax.set_ylabel('Outer Radius [μm]')
    ax.set_title('Shell Trajectory')
    ax.grid(True, alpha=0.3)
    
    # Velocity
    ax = axes[0, 1]
    ax.plot(t, [s.V for s in states], 'r-', linewidth=2)
    ax.set_xlabel('Time [ns]')
    ax.set_ylabel('Velocity [km/s]')
    ax.set_title('Implosion Velocity')
    ax.grid(True, alpha=0.3)
    
    # Hot spot radius
    ax = axes[0, 2]
    ax.plot(t, [s.R_hs for s in states], 'g-', linewidth=2)
    ax.set_xlabel('Time [ns]')
    ax.set_ylabel('Hot Spot Radius [μm]')
    ax.set_title('Hot Spot Compression')
    ax.grid(True, alpha=0.3)
    
    # Temperature
    ax = axes[1, 0]
    ax.plot(t, [s.T_hs for s in states], 'orange', linewidth=2)
    ax.set_xlabel('Time [ns]')
    ax.set_ylabel('Temperature [keV]')
    ax.set_title('Hot Spot Temperature')
    ax.grid(True, alpha=0.3)
    
    # Density
    ax = axes[1, 1]
    ax.plot(t, [s.rho_hs for s in states], 'purple', linewidth=2)
    ax.set_xlabel('Time [ns]')
    ax.set_ylabel('Density [g/cm³]')
    ax.set_title('Hot Spot Density')
    ax.grid(True, alpha=0.3)
    
    # Yield
    ax = axes[1, 2]
    ax.plot(t, [s.yield_MJ for s in states], 'red', linewidth=2)
    ax.set_xlabel('Time [ns]')
    ax.set_ylabel('Yield [MJ]')
    ax.set_title('Fusion Yield')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# ============================================================================
# STREAMLIT APP
# ============================================================================

st.title("🔬 NIF Inertial Confinement Fusion Simulator")
st.markdown("""
Simulate the **National Ignition Facility's** historic December 2022 ignition shot.
Adjust parameters to explore the physics of fusion ignition.
""")

# Sidebar for parameters
st.sidebar.header("Simulation Parameters")

st.sidebar.subheader("Laser")
energy_MJ = st.sidebar.slider("Laser Energy [MJ]", 0.5, 3.5, 2.05, 0.05)
peak_power_TW = st.sidebar.slider("Peak Power [TW]", 200, 700, 500, 10)
pulse_type = st.sidebar.selectbox("Pulse Shape", 
    ["NIF Standard", "Square", "High Foot", "Adiabat-shaped"])

st.sidebar.subheader("Target")
target_scale = st.sidebar.slider("Target Scale", 0.7, 1.3, 1.0, 0.05)

st.sidebar.markdown("---")
run_button = st.sidebar.button("🚀 Run Simulation", type="primary")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Simulation Results")
    
    if run_button or 'states' not in st.session_state:
        with st.spinner("Running simulation..."):
            states, sim = run_simulation(energy_MJ, peak_power_TW, 
                                         pulse_type, target_scale)
            st.session_state.states = states
            st.session_state.sim = sim
    
    states = st.session_state.states
    
    # Find key metrics
    peak_rho = max(states, key=lambda s: s.rho_hs)
    peak_T = max(states, key=lambda s: s.T_hs)
    min_V = min(states, key=lambda s: s.V)
    final = states[-1]
    Q = final.yield_MJ / energy_MJ
    
    # Display metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Fusion Yield", f"{final.yield_MJ:.2f} MJ")
    with m2:
        st.metric("Gain Q", f"{Q:.2f}", delta="IGNITION!" if Q > 1 else None)
    with m3:
        st.metric("Peak Temperature", f"{peak_T.T_hs:.1f} keV")
    with m4:
        st.metric("Peak Density", f"{peak_rho.rho_hs:.0f} g/cm³")
    
    # Time evolution plot
    fig = plot_time_evolution(states)
    st.pyplot(fig)
    plt.close()

with col2:
    st.subheader("Capsule Visualization")
    
    # Time slider for visualization
    time_idx = st.slider("Time Step", 0, len(states)-1, 0)
    s = states[time_idx]
    
    # Create capsule diagram
    fig = plot_implosion_diagram(s.R, s.R_hs, s.T_hs, s.rho_hs, s.V, s.t)
    st.pyplot(fig)
    plt.close()
    
    # State details
    st.markdown(f"""
    **Time:** {s.t:.2f} ns  
    **Shell Radius:** {s.R:.0f} μm  
    **Hot Spot Radius:** {s.R_hs:.0f} μm  
    **Velocity:** {s.V:.0f} km/s  
    **Temperature:** {s.T_hs:.1f} keV  
    **Density:** {s.rho_hs:.0f} g/cm³  
    **Convergence:** {s.CR:.1f}×  
    **Yield (so far):** {s.yield_MJ:.3f} MJ
    """)

# Additional analysis
st.markdown("---")
st.subheader("📊 Analysis")

tab1, tab2, tab3 = st.tabs(["Ignition Physics", "Parameter Sensitivity", "About"])

with tab1:
    st.markdown("""
    ### Lawson Criterion for Ignition
    
    For fusion ignition, the hot spot must satisfy:
    
    **n·τ·T > 3×10²¹ keV·s/m³**
    
    where:
    - n = plasma density
    - τ = confinement time
    - T = temperature
    
    At NIF conditions:
    """)
    
    # Compute Lawson parameter
    peak_state = peak_rho
    n = peak_state.rho_hs * 1e6 / (2.5 * 1.673e-24)  # Convert to m⁻³
    tau = 1e-10  # ~100 ps confinement
    nTtau = n * peak_state.T_hs * tau
    
    c1, c2 = st.columns(2)
    with c1:
        st.metric("n·T·τ", f"{nTtau:.2e} keV·s/m³")
    with c2:
        st.metric("Required", "3×10²¹ keV·s/m³")
    
    if nTtau > 3e21:
        st.success("✅ Lawson criterion satisfied - Ignition possible!")
    else:
        st.warning("⚠️ Below Lawson criterion - May not ignite")

with tab2:
    st.markdown("### Quick Parameter Scan")
    
    if st.button("Run Energy Scan"):
        energies = np.linspace(1.0, 3.0, 11)
        gains = []
        
        progress = st.progress(0)
        for i, E in enumerate(energies):
            states_scan, _ = run_simulation(E, 500, "NIF Standard", 1.0)
            Q_scan = states_scan[-1].yield_MJ / E
            gains.append(Q_scan)
            progress.progress((i + 1) / len(energies))
        
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(energies, gains, 'b-o', linewidth=2, markersize=8)
        ax.axhline(y=1, color='r', linestyle='--', label='Ignition (Q=1)')
        ax.axvline(x=energy_MJ, color='g', linestyle=':', label=f'Current ({energy_MJ} MJ)')
        ax.set_xlabel('Laser Energy [MJ]')
        ax.set_ylabel('Gain Q')
        ax.set_title('Ignition Cliff')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()

with tab3:
    st.markdown("""
    ### About This Simulation
    
    This simulator models the **National Ignition Facility's** December 5, 2022 
    experiment - the first time in history that a fusion reaction produced more 
    energy than the laser energy used to initiate it.
    
    **Key features modeled:**
    - 192-beam laser implosion at up to 500 TW
    - Ablation-driven rocket acceleration
    - Adiabatic compression heating
    - DT thermonuclear burn with Bosch-Hale reactivity
    - Alpha particle bootstrap heating
    
    **NIF Actual Results (December 2022):**
    - Laser energy: 2.05 MJ
    - Fusion yield: 3.15 MJ
    - Gain Q: 1.54
    
    **Limitations:**
    - 0D (point model) rather than full hydrodynamics
    - Empirically calibrated parameters
    - No radiation transport
    - No instability effects
    
    ---
    *Created by Evan Pieser, 2026*
    """)

# Footer
st.markdown("---")
st.caption("NIF ICF Simulation | Based on December 2022 ignition experiment data")
