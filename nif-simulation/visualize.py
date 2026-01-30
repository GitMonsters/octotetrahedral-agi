"""
ICF Implosion Visualization
============================

Beautiful visualizations of inertial confinement fusion at NIF:
- Time evolution of implosion parameters
- Radial profiles of density and temperature
- 2D cross-section view
- Animation of the implosion sequence

Author: Evan Pieser
Date: 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle
import matplotlib.animation as animation
from typing import List

# Import simulation
from icf_simulation import NIF, nif_pulse, State


def plasma_cmap():
    """Hot plasma colormap: black -> purple -> red -> orange -> yellow -> white."""
    colors = [
        (0.0, 0.0, 0.1),   # Deep blue-black
        (0.2, 0.0, 0.4),   # Purple
        (0.5, 0.0, 0.3),   # Magenta
        (0.8, 0.1, 0.1),   # Red
        (1.0, 0.4, 0.0),   # Orange
        (1.0, 0.8, 0.2),   # Yellow
        (1.0, 1.0, 1.0),   # White
    ]
    return LinearSegmentedColormap.from_list('plasma_hot', colors, N=256)


def plot_time_evolution(states: List[State], save_path: str = None):
    """Plot key parameters vs time."""
    
    t = np.array([s.t for s in states])
    R = np.array([s.R for s in states])
    V = np.array([s.V for s in states])
    R_hs = np.array([s.R_hs for s in states])
    T = np.array([s.T_hs for s in states])
    rho = np.array([s.rho_hs for s in states])
    Y = np.array([s.yield_MJ for s in states])
    P_laser = np.array([nif_pulse(ti) for ti in t])
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.patch.set_facecolor('#0a0a0a')
    
    for ax in axes.flat:
        ax.set_facecolor('#0a0a0a')
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
    
    # Shell radius
    axes[0,0].plot(t, R, 'cyan', lw=2, label='Shell')
    axes[0,0].plot(t, R_hs, 'yellow', lw=2, label='Hot spot')
    axes[0,0].set_ylabel('Radius [μm]')
    axes[0,0].set_title('Implosion Trajectory')
    axes[0,0].legend(facecolor='#1a1a1a', edgecolor='white', labelcolor='white')
    axes[0,0].set_ylim(0, 1200)
    axes[0,0].grid(alpha=0.2, color='white')
    
    # Velocity
    axes[0,1].plot(t, V, 'lime', lw=2)
    axes[0,1].axhline(y=0, color='white', ls='--', alpha=0.3)
    axes[0,1].set_ylabel('Velocity [km/s]')
    axes[0,1].set_title('Implosion Velocity')
    axes[0,1].grid(alpha=0.2, color='white')
    
    # Temperature  
    axes[0,2].semilogy(t, np.maximum(T, 0.01), 'red', lw=2)
    axes[0,2].axhline(y=4, color='yellow', ls='--', alpha=0.5, label='Ignition threshold')
    axes[0,2].set_ylabel('Temperature [keV]')
    axes[0,2].set_title('Hot Spot Temperature')
    axes[0,2].legend(facecolor='#1a1a1a', edgecolor='white', labelcolor='white')
    axes[0,2].grid(alpha=0.2, color='white')
    
    # Density
    axes[1,0].semilogy(t, np.maximum(rho, 0.01), 'orange', lw=2)
    axes[1,0].set_xlabel('Time [ns]')
    axes[1,0].set_ylabel('Density [g/cm³]')
    axes[1,0].set_title('Hot Spot Density')
    axes[1,0].grid(alpha=0.2, color='white')
    
    # Fusion yield
    axes[1,1].plot(t, Y, 'magenta', lw=2)
    axes[1,1].axhline(y=2.05, color='cyan', ls='--', alpha=0.5, label='Laser energy')
    axes[1,1].fill_between(t, Y, alpha=0.3, color='magenta')
    axes[1,1].set_xlabel('Time [ns]')
    axes[1,1].set_ylabel('Yield [MJ]')
    axes[1,1].set_title('Fusion Yield')
    axes[1,1].legend(facecolor='#1a1a1a', edgecolor='white', labelcolor='white')
    axes[1,1].grid(alpha=0.2, color='white')
    
    # Laser power
    axes[1,2].fill_between(t, P_laser, alpha=0.5, color='cyan')
    axes[1,2].plot(t, P_laser, 'cyan', lw=2)
    axes[1,2].set_xlabel('Time [ns]')
    axes[1,2].set_ylabel('Power [TW]')
    axes[1,2].set_title('Laser Power')
    axes[1,2].grid(alpha=0.2, color='white')
    
    plt.suptitle('NIF Ignition Shot - Time Evolution', fontsize=16, color='white', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0a0a0a')
        print(f"  Saved: {save_path}")
    
    return fig


def plot_implosion_diagram(state: State, save_path: str = None):
    """Create a 2D cross-section diagram of the implosion."""
    
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor('#0a0a0a')
    ax.set_facecolor('#0a0a0a')
    
    # Create concentric circles representing the implosion
    R_shell = state.R
    R_hs = state.R_hs
    
    # Outer corona / ablated material
    corona = Circle((0, 0), R_shell * 1.3, fill=True, 
                    facecolor='#1a0a2a', edgecolor='purple', alpha=0.3)
    ax.add_patch(corona)
    
    # Shell
    shell = Circle((0, 0), R_shell, fill=True,
                   facecolor='#2a4a8a', edgecolor='cyan', linewidth=2)
    ax.add_patch(shell)
    
    # Compressed fuel layer
    fuel = Circle((0, 0), R_hs * 1.2, fill=True,
                  facecolor='#4a2a0a', edgecolor='orange', linewidth=1)
    ax.add_patch(fuel)
    
    # Hot spot
    T_norm = min(state.T_hs / 50, 1.0)  # Normalize temperature
    hot_color = plt.cm.hot(T_norm)
    hotspot = Circle((0, 0), R_hs, fill=True,
                     facecolor=hot_color, edgecolor='yellow', linewidth=2)
    ax.add_patch(hotspot)
    
    # Central bright spot
    if state.T_hs > 10:
        core = Circle((0, 0), R_hs * 0.3, fill=True,
                      facecolor='white', edgecolor='white', alpha=0.8)
        ax.add_patch(core)
    
    # Laser beams (simplified - show 8 beams)
    if nif_pulse(state.t) > 10:
        for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
            x_start = R_shell * 2 * np.cos(angle)
            y_start = R_shell * 2 * np.sin(angle)
            ax.plot([x_start, R_shell*1.1*np.cos(angle)], 
                   [y_start, R_shell*1.1*np.sin(angle)],
                   color='cyan', alpha=0.5, lw=3)
    
    # Scale and labels
    max_r = max(R_shell * 1.5, 1200)
    ax.set_xlim(-max_r, max_r)
    ax.set_ylim(-max_r, max_r)
    ax.set_aspect('equal')
    
    # Annotations
    ax.text(0, -max_r*0.85, f't = {state.t:.2f} ns', ha='center', 
            fontsize=14, color='white')
    ax.text(0, -max_r*0.95, f'T = {state.T_hs:.1f} keV  |  ρ = {state.rho_hs:.0f} g/cm³', 
            ha='center', fontsize=12, color='yellow')
    
    # Legend
    ax.text(-max_r*0.9, max_r*0.9, 'Shell', fontsize=10, color='cyan')
    ax.text(-max_r*0.9, max_r*0.8, 'Fuel', fontsize=10, color='orange')
    ax.text(-max_r*0.9, max_r*0.7, 'Hot Spot', fontsize=10, color='yellow')
    
    ax.set_title(f'NIF Implosion Cross-Section', fontsize=16, color='white')
    ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0a0a0a')
        print(f"  Saved: {save_path}")
    
    return fig


def plot_192_beams(save_path: str = None):
    """Visualize the 192 laser beam geometry."""
    
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor('#0a0a0a')
    ax.set_facecolor('#0a0a0a')
    
    # NIF beam geometry: beams at 23.5°, 30°, 44.5°, 50° from polar axis
    # 8 beams per cone, 4 cones per hemisphere = 64 beams per hemisphere
    # Total: 192 beams (but we'll approximate)
    
    angles = np.array([23.5, 30.0, 44.5, 50.0]) * np.pi / 180
    n_azimuth = 8
    
    beam_points = []
    
    for sign in [1, -1]:  # Both hemispheres
        for theta in angles:
            for i in range(n_azimuth):
                phi = 2 * np.pi * i / n_azimuth + (np.pi / n_azimuth if sign == -1 else 0)
                
                x = np.sin(theta) * np.cos(phi)
                y = np.sin(theta) * np.sin(phi)
                z = sign * np.cos(theta)
                
                # 3 beams per "quad"
                for offset in np.linspace(-0.02, 0.02, 3):
                    beam_points.append([x + offset, y + offset, z])
    
    beam_points = np.array(beam_points)
    
    # Target sphere
    u = np.linspace(0, 2*np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    xs = 0.1 * np.outer(np.cos(u), np.sin(v))
    ys = 0.1 * np.outer(np.sin(u), np.sin(v))
    zs = 0.1 * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(xs, ys, zs, color='yellow', alpha=0.8)
    
    # Draw laser beams
    for pt in beam_points:
        ax.plot([pt[0], 0], [pt[1], 0], [pt[2], 0], 
                'c-', alpha=0.15, lw=0.5)
        ax.scatter([pt[0]], [pt[1]], [pt[2]], c='cyan', s=2, alpha=0.5)
    
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_zlim(-1.2, 1.2)
    
    # Style
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('white')
    ax.yaxis.pane.set_edgecolor('white')
    ax.zaxis.pane.set_edgecolor('white')
    ax.tick_params(colors='white')
    
    ax.set_title('NIF 192-Beam Laser Configuration', fontsize=16, color='white')
    ax.set_xlabel('X', color='white')
    ax.set_ylabel('Y', color='white')
    ax.set_zlabel('Z', color='white')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0a0a0a')
        print(f"  Saved: {save_path}")
    
    return fig


def create_animation(states: List[State], save_path: str = 'implosion.gif'):
    """Create animated visualization of the implosion."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('#0a0a0a')
    
    for ax in [ax1, ax2]:
        ax.set_facecolor('#0a0a0a')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('white')
    
    # Sample frames
    n_frames = min(120, len(states))
    frame_idx = np.linspace(0, len(states)-1, n_frames, dtype=int)
    
    # Get ranges
    all_R = [states[i].R for i in frame_idx]
    max_R = max(all_R) * 1.3
    
    def animate(frame):
        ax1.clear()
        ax2.clear()
        
        ax1.set_facecolor('#0a0a0a')
        ax2.set_facecolor('#0a0a0a')
        
        s = states[frame_idx[frame]]
        
        # Left: 2D diagram
        shell = Circle((0, 0), s.R, fill=True,
                       facecolor='#2a4a8a', edgecolor='cyan', lw=2)
        ax1.add_patch(shell)
        
        T_norm = min(s.T_hs / 50, 1.0)
        hotspot = Circle((0, 0), s.R_hs, fill=True,
                        facecolor=plt.cm.hot(T_norm), edgecolor='yellow', lw=2)
        ax1.add_patch(hotspot)
        
        if s.T_hs > 20:
            core = Circle((0, 0), s.R_hs * 0.3, fill=True,
                         facecolor='white', alpha=0.8)
            ax1.add_patch(core)
        
        ax1.set_xlim(-max_R, max_R)
        ax1.set_ylim(-max_R, max_R)
        ax1.set_aspect('equal')
        ax1.set_title(f't = {s.t:.2f} ns', color='white', fontsize=14)
        ax1.axis('off')
        
        # Right: key parameters
        ax2.barh(['Yield [MJ]'], [s.yield_MJ], color='magenta', height=0.5)
        ax2.barh(['T [keV]'], [s.T_hs], color='red', height=0.5)
        ax2.barh(['ρ [g/cm³]'], [s.rho_hs], color='orange', height=0.5)
        ax2.barh(['V [km/s]'], [abs(s.V)/10], color='lime', height=0.5)
        
        ax2.set_xlim(0, 120)
        ax2.tick_params(colors='white')
        ax2.set_title('Parameters', color='white', fontsize=14)
        
        return []
    
    anim = animation.FuncAnimation(fig, animate, frames=n_frames, 
                                   interval=50, blit=True)
    
    print(f"  Creating animation ({n_frames} frames)...")
    anim.save(save_path, writer='pillow', fps=20)
    print(f"  Saved: {save_path}")
    plt.close()


def main():
    """Run simulation and create all visualizations."""
    
    print()
    print("═" * 60)
    print("  ICF VISUALIZATION SUITE")
    print("═" * 60)
    print()
    
    # Run simulation
    print("  Running simulation...")
    sim = NIF()
    states = sim.run(t_end_ns=14, laser_func=nif_pulse, dt=0.005)
    print()
    
    # Create visualizations
    print("  Creating visualizations...")
    print()
    
    # 1. Time evolution
    plot_time_evolution(states, 'time_evolution.png')
    plt.close()
    
    # 2. Peak compression diagram
    peak_state = max(states, key=lambda s: s.rho_hs)
    plot_implosion_diagram(peak_state, 'implosion_peak.png')
    plt.close()
    
    # 3. 192-beam geometry
    plot_192_beams('beam_geometry.png')
    plt.close()
    
    # 4. Animation
    create_animation(states, 'implosion.gif')
    
    print()
    print("  ✓ Visualizations complete!")
    print()
    print("  Output files:")
    print("    - time_evolution.png")
    print("    - implosion_peak.png")
    print("    - beam_geometry.png")
    print("    - implosion.gif")
    print()
    print("═" * 60)


if __name__ == "__main__":
    main()
