"""
Animated Sacred Geometry from Fusion Physics
=============================================

Creates evolving mandalas that transform during the implosion sequence.
The sacred geometry breathes with the physics - compressing, heating, igniting.

Animation Phases (mapped to implosion):
1. ACCELERATION (0-8 ns): Golden spiral winds inward, beams activate
2. COMPRESSION (8-12 ns): Flower of Life contracts, colors heat
3. STAGNATION (12-12.5 ns): Metatron's Cube crystallizes, pressure builds
4. BURN (12.5-14 ns): Vesica Piscis fusion portal opens, energy radiates
5. IGNITION (peak): Unified mandala explodes with light

The mathematics of stellar creation, animated.

Author: Evan Pieser
Date: 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Wedge, FancyArrowPatch
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D

# Physical constants
PHI = (1 + np.sqrt(5)) / 2  # Golden ratio
PI = np.pi

# ICF physics parameters
Q_IGNITION = 1.41
CR_OPTIMAL = 31
T_PEAK = 75          # keV
RHO_PEAK = 112       # g/cm³
N_BEAMS = 192
BANG_TIME = 8.33     # ns
R0 = 1100            # Initial radius [μm]
R_FINAL = 28         # Final hot spot [μm]


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


def get_implosion_state(t_ns):
    """
    Get implosion parameters at time t.
    Simplified model matching our full simulation.
    """
    # Shell radius
    if t_ns < 2:
        R = R0
        V = 0
    elif t_ns < 10:
        # Acceleration phase
        R = R0 - 50 * (t_ns - 2)**2
        V = -100 * (t_ns - 2)
    elif t_ns < 12.5:
        # Coast and compression
        t_coast = t_ns - 10
        R = max(R0 - 50 * 64 - 400 * t_coast, 35)
        V = -400
    else:
        # Stagnation and bounce
        R = 35 + 20 * (t_ns - 12.5)
        V = -400 + 200 * (t_ns - 12.5)
    
    # Ensure R is positive
    R = max(R, 35)
    
    # Hot spot
    R_hs = max(R0 * 0.77 * (R / R0)**0.95, 25)
    
    # Temperature (compression heating + shock)
    CR = max(R0 * 0.77 / R_hs, 1.0)
    T = min(0.03 * CR**2, T_PEAK)
    if t_ns > 11:
        T = min(T + 20 * (t_ns - 11), T_PEAK)
    
    # Density
    rho = min(0.25 * CR**3, RHO_PEAK)
    
    # Yield (accumulates during burn)
    if t_ns < 11:
        Y = 0
    elif t_ns < 14:
        Y = 2.88 * ((t_ns - 11) / 3)**2  # Quadratic ramp
    else:
        Y = 2.88
    
    return {
        'R': R,
        'R_hs': R_hs,
        'V': V,
        'T': T,
        'rho': rho,
        'CR': CR,
        'Y': Y,
        'P_laser': nif_pulse(t_ns)
    }


class AnimatedSacredGeometry:
    """
    Animated sacred geometry that evolves with the implosion.
    """
    
    def __init__(self, duration_ns=14, fps=30):
        self.duration = duration_ns
        self.fps = fps
        self.n_frames = int(duration_ns * fps)
        
        # Time array
        self.times = np.linspace(0, duration_ns, self.n_frames)
        
    def create_golden_spiral_animation(self):
        """
        Golden spiral that winds inward as the shell compresses.
        """
        print("Creating golden spiral animation...")
        
        fig, ax = plt.subplots(figsize=(10, 10), facecolor='black')
        ax.set_facecolor('black')
        ax.set_xlim(-1300, 1300)
        ax.set_ylim(-1300, 1300)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Pre-compute spiral
        theta_full = np.linspace(0, 7 * PI, 500)
        
        def animate(frame):
            ax.clear()
            ax.set_facecolor('black')
            ax.set_xlim(-1300, 1300)
            ax.set_ylim(-1300, 1300)
            ax.axis('off')
            
            t = self.times[frame]
            state = get_implosion_state(t)
            R_shell = state['R']
            T = state['T']
            P = state['P_laser']
            
            # Golden spiral winds inward with compression
            # At t=0, show outer turns; as t increases, reveal inner turns
            progress = 1 - R_shell / R0  # 0 to ~0.97
            n_points = int(50 + 450 * progress)
            
            theta = theta_full[:n_points]
            a = R_FINAL
            r = a * PHI ** (theta / (PI/2))
            
            # Only show points inside current shell
            mask = r < R_shell
            theta_vis = theta[mask]
            r_vis = r[mask]
            
            x = r_vis * np.cos(theta_vis)
            y = r_vis * np.sin(theta_vis)
            
            # Color by temperature
            if len(x) > 1:
                colors = plt.cm.plasma(np.linspace(0, min(T/T_PEAK, 1), len(x)))
                for i in range(len(x) - 1):
                    ax.plot(x[i:i+2], y[i:i+2], color=colors[i], 
                           linewidth=2 + 2*progress, alpha=0.8)
            
            # Shell circle
            shell = Circle((0, 0), R_shell, fill=False, color='cyan', 
                          linewidth=2, alpha=0.7)
            ax.add_patch(shell)
            
            # Hot spot
            hs = Circle((0, 0), state['R_hs'], fill=True, 
                       facecolor=plt.cm.hot(min(T/T_PEAK, 1)), 
                       alpha=0.5, edgecolor='white', linewidth=1)
            ax.add_patch(hs)
            
            # Laser beams (when active)
            if P > 10:
                beam_alpha = min(P / 500, 1) * 0.3
                for i in range(24):  # Show 24 representative beams
                    angle = i * PI / 12
                    x1, y1 = 1250 * np.cos(angle), 1250 * np.sin(angle)
                    x2, y2 = R_shell * np.cos(angle), R_shell * np.sin(angle)
                    ax.plot([x1, x2], [y1, y2], color='gold', 
                           linewidth=1, alpha=beam_alpha)
            
            # Golden ratio markers
            for i in range(min(7, int(progress * 7) + 1)):
                r_point = a * PHI ** i
                if r_point < R_shell:
                    ax.plot(r_point, 0, 'o', color='white', markersize=6, alpha=0.7)
                    ax.text(r_point * 1.15, 0, f'φ^{i}', color='white', 
                           fontsize=8, alpha=0.7)
            
            # Title with physics
            ax.set_title(f'Golden Spiral of Compression\n'
                        f't = {t:.1f} ns  |  R = {R_shell:.0f} μm  |  '
                        f'T = {T:.1f} keV  |  CR = {state["CR"]:.1f}×',
                        color='gold', fontsize=12, pad=10)
            
            return []
        
        anim = FuncAnimation(fig, animate, frames=self.n_frames, 
                            interval=1000/self.fps, blit=False)
        anim.save('anim_golden_spiral.gif', writer=PillowWriter(fps=self.fps),
                 dpi=100, savefig_kwargs={'facecolor': 'black'})
        plt.close()
        print("Saved: anim_golden_spiral.gif")
        
    def create_flower_of_life_animation(self):
        """
        Flower of Life that pulses and contracts with compression.
        """
        print("Creating Flower of Life animation...")
        
        fig, ax = plt.subplots(figsize=(10, 10), facecolor='black')
        
        def animate(frame):
            ax.clear()
            ax.set_facecolor('black')
            ax.set_xlim(-4, 4)
            ax.set_ylim(-4, 4)
            ax.axis('off')
            
            t = self.times[frame]
            state = get_implosion_state(t)
            CR = state['CR']
            T = state['T']
            
            # Base radius scales with compression
            R_base = 3.0 / (1 + 0.5 * (CR - 1) / 30)  # Contracts as CR increases
            
            # Pulsing effect from laser
            pulse = 1 + 0.05 * np.sin(t * 10) * (state['P_laser'] / 500)
            R = R_base * pulse
            
            # Flower of Life: 19 circles
            centers = [(0, 0)]
            for i in range(6):
                angle = i * PI / 3
                centers.append((R * np.cos(angle), R * np.sin(angle)))
            for i in range(6):
                angle = i * PI / 3
                centers.append((2 * R * np.cos(angle), 2 * R * np.sin(angle)))
                angle2 = (i + 0.5) * PI / 3
                centers.append((np.sqrt(3) * R * np.cos(angle2), 
                              np.sqrt(3) * R * np.sin(angle2)))
            
            # Color by temperature
            color_val = min(T / T_PEAK, 1)
            base_color = plt.cm.hot(0.3 + 0.6 * color_val)
            
            for cx, cy in centers:
                dist = np.sqrt(cx**2 + cy**2)
                alpha = max(0.1, min(0.9, 0.8 - 0.3 * dist / (3 * R + 0.01)))
                circle = Circle((cx, cy), R, fill=False, 
                               color=base_color, linewidth=1.5, alpha=alpha)
                ax.add_patch(circle)
            
            # Central seed of life (7 circles) - brighter
            seed_R = R * 0.5
            seed_color = 'cyan' if T < 10 else plt.cm.hot(min(T/T_PEAK + 0.3, 1))
            for i in range(6):
                angle = i * PI / 3
                cx, cy = seed_R * np.cos(angle), seed_R * np.sin(angle)
                circle = Circle((cx, cy), seed_R, fill=False, 
                               color=seed_color, linewidth=2, alpha=0.9)
                ax.add_patch(circle)
            circle = Circle((0, 0), seed_R, fill=False, 
                           color=seed_color, linewidth=2, alpha=0.9)
            ax.add_patch(circle)
            
            # Outer shell
            shell = Circle((0, 0), 3.5, fill=False, color='gold', 
                          linewidth=2, alpha=0.5)
            ax.add_patch(shell)
            
            # Mode labels
            ax.text(0, 0, 'l=0', color='white', fontsize=10, ha='center', va='center')
            
            # Title
            ax.set_title(f'Flower of Life - Spherical Harmonics\n'
                        f't = {t:.1f} ns  |  T = {T:.1f} keV  |  '
                        f'Modes l=0,6,12 active',
                        color='gold', fontsize=12, pad=10)
            
            return []
        
        anim = FuncAnimation(fig, animate, frames=self.n_frames,
                            interval=1000/self.fps, blit=False)
        anim.save('anim_flower_of_life.gif', writer=PillowWriter(fps=self.fps),
                 dpi=100, savefig_kwargs={'facecolor': 'black'})
        plt.close()
        print("Saved: anim_flower_of_life.gif")
        
    def create_metatrons_cube_animation(self):
        """
        Metatron's Cube that crystallizes at stagnation.
        """
        print("Creating Metatron's Cube animation...")
        
        fig, ax = plt.subplots(figsize=(10, 10), facecolor='black')
        
        # Pre-compute Metatron's cube structure
        R_meta = 1.0
        centers = [(0, 0)]
        for i in range(6):
            angle = i * PI / 3
            centers.append((R_meta * np.cos(angle), R_meta * np.sin(angle)))
        for i in range(6):
            angle = i * PI / 3 + PI / 6
            centers.append((2 * R_meta * np.cos(angle), 2 * R_meta * np.sin(angle)))
        
        # All connecting lines
        all_lines = []
        for i, (x1, y1) in enumerate(centers):
            for j, (x2, y2) in enumerate(centers):
                if i < j:
                    all_lines.append([(x1, y1), (x2, y2)])
        
        def animate(frame):
            ax.clear()
            ax.set_facecolor('black')
            ax.set_xlim(-4, 4)
            ax.set_ylim(-4.5, 4)
            ax.axis('off')
            
            t = self.times[frame]
            state = get_implosion_state(t)
            
            # Crystallization happens during stagnation (t > 11 ns)
            if t < 8:
                crystal_progress = 0
            elif t < 12:
                crystal_progress = (t - 8) / 4
            else:
                crystal_progress = 1.0
            
            # Energy radiance during burn
            burn_glow = state['Y'] / 2.88 if state['Y'] > 0 else 0
            
            # Draw lines that have crystallized
            n_lines = int(len(all_lines) * crystal_progress)
            if n_lines > 0:
                lines_to_draw = all_lines[:n_lines]
                
                # Color by burn progress
                line_colors = []
                for i, ((x1, y1), (x2, y2)) in enumerate(lines_to_draw):
                    dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    color = plt.cm.plasma(0.3 + 0.5 * burn_glow + 0.2 * i/len(lines_to_draw))
                    line_colors.append(color)
                
                lc = LineCollection(lines_to_draw, colors=line_colors, 
                                   linewidths=1, alpha=0.6)
                ax.add_collection(lc)
            
            # Draw vertices (spheres)
            n_vertices = int(len(centers) * crystal_progress)
            for i, (cx, cy) in enumerate(centers[:n_vertices]):
                glow = burn_glow if i == 0 else burn_glow * 0.5
                color = plt.cm.hot(0.3 + 0.6 * glow)
                size = 0.15 + 0.1 * burn_glow
                circle = Circle((cx, cy), size, fill=True, facecolor=color,
                               edgecolor='gold', linewidth=2, alpha=0.9)
                ax.add_patch(circle)
            
            # Platonic solid highlight (hexagon = cube projection)
            if crystal_progress > 0.5:
                hex_alpha = (crystal_progress - 0.5) * 2
                hex_points = [centers[i] for i in range(1, 7)]
                hexagon = Polygon(hex_points, fill=False, edgecolor='cyan',
                                 linewidth=2, alpha=hex_alpha * 0.7)
                ax.add_patch(hexagon)
            
            # Energy radiation during burn
            if burn_glow > 0:
                for i in range(8):
                    angle = i * PI / 4 + t * 0.5
                    for r in [2.5, 3.0, 3.5]:
                        x = r * np.cos(angle)
                        y = r * np.sin(angle)
                        ax.plot(x, y, '*', color='yellow', 
                               markersize=4 + 6 * burn_glow, alpha=burn_glow)
            
            # Title
            ax.set_title(f"Metatron's Cube - 192 Beam Geometry\n"
                        f't = {t:.1f} ns  |  Crystallization: {crystal_progress*100:.0f}%  |  '
                        f'Yield = {state["Y"]:.2f} MJ',
                        color='gold', fontsize=12, pad=10)
            
            return []
        
        anim = FuncAnimation(fig, animate, frames=self.n_frames,
                            interval=1000/self.fps, blit=False)
        anim.save('anim_metatrons_cube.gif', writer=PillowWriter(fps=self.fps),
                 dpi=100, savefig_kwargs={'facecolor': 'black'})
        plt.close()
        print("Saved: anim_metatrons_cube.gif")
        
    def create_vesica_piscis_animation(self):
        """
        Vesica Piscis fusion portal that opens during burn.
        """
        print("Creating Vesica Piscis animation...")
        
        fig, ax = plt.subplots(figsize=(12, 10), facecolor='black')
        
        def animate(frame):
            ax.clear()
            ax.set_facecolor('black')
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)
            ax.axis('off')
            
            t = self.times[frame]
            state = get_implosion_state(t)
            T = state['T']
            
            # Fusion begins above T > 4 keV
            fusion_progress = max(0, min(1, (T - 4) / 66))  # 0 at 4 keV, 1 at 70 keV
            
            R = 2.0
            
            # D and T circles approach each other
            separation = 1.0 - 0.5 * fusion_progress  # From 1.0 to 0.5
            d_center = (-R * separation, 0)
            t_center = (R * separation, 0)
            
            # Deuterium circle
            d_alpha = 0.9 - 0.4 * fusion_progress
            d_circle = Circle(d_center, R, fill=False, color='deepskyblue',
                            linewidth=3, alpha=d_alpha)
            ax.add_patch(d_circle)
            if d_alpha > 0.3:
                ax.text(d_center[0] - R/2, 0, 'D\n(²H)', color='deepskyblue',
                       fontsize=14, ha='center', va='center', fontweight='bold',
                       alpha=d_alpha)
            
            # Tritium circle
            t_circle = Circle(t_center, R, fill=False, color='lightgreen',
                            linewidth=3, alpha=d_alpha)
            ax.add_patch(t_circle)
            if d_alpha > 0.3:
                ax.text(t_center[0] + R/2, 0, 'T\n(³H)', color='lightgreen',
                       fontsize=14, ha='center', va='center', fontweight='bold',
                       alpha=d_alpha)
            
            # Vesica Piscis (intersection) - glows during fusion
            if fusion_progress > 0:
                theta = np.linspace(-PI/3, PI/3, 100)
                x_left = d_center[0] + R * np.cos(theta)
                y_left = R * np.sin(theta)
                x_right = t_center[0] + R * np.cos(PI - theta)
                y_right = R * np.sin(PI - theta)
                
                vesica_x = np.concatenate([x_left, x_right[::-1]])
                vesica_y = np.concatenate([y_left, y_right[::-1]])
                
                vesica_color = plt.cm.hot(0.3 + 0.6 * fusion_progress)
                ax.fill(vesica_x, vesica_y, color=vesica_color, 
                       alpha=0.3 + 0.5 * fusion_progress)
            
            # Fusion products emerge
            if fusion_progress > 0.3:
                product_progress = (fusion_progress - 0.3) / 0.7
                
                # Alpha particle (up)
                alpha_y = 0.5 + 2.5 * product_progress
                ax.annotate('', xy=(0, alpha_y), xytext=(0, 0.3),
                           arrowprops=dict(arrowstyle='->', color='red', lw=3,
                                          alpha=product_progress))
                ax.text(0, alpha_y + 0.5, '⁴He\n3.5 MeV', color='red',
                       fontsize=12, ha='center', fontweight='bold',
                       alpha=product_progress)
                
                # Neutron (down)
                n_y = -0.5 - 2.5 * product_progress
                ax.annotate('', xy=(0, n_y), xytext=(0, -0.3),
                           arrowprops=dict(arrowstyle='->', color='white', lw=3,
                                          alpha=product_progress))
                ax.text(0, n_y - 0.5, 'n\n14.1 MeV', color='white',
                       fontsize=12, ha='center', fontweight='bold',
                       alpha=product_progress)
            
            # Energy release glow
            if state['Y'] > 0:
                glow_size = 0.5 + 2 * (state['Y'] / 2.88)
                glow = Circle((0, 0), glow_size, fill=True, 
                             facecolor='yellow', alpha=0.2)
                ax.add_patch(glow)
            
            # Title
            ax.set_title(f'Vesica Piscis - Fusion Portal\n'
                        f't = {t:.1f} ns  |  T = {T:.1f} keV  |  '
                        f'Yield = {state["Y"]:.2f} MJ',
                        color='gold', fontsize=12, pad=10)
            
            # Equation
            if fusion_progress > 0.5:
                ax.text(0, -4.5, 'D + T → ⁴He + n + 17.6 MeV', color='gold',
                       fontsize=14, ha='center', fontweight='bold',
                       alpha=fusion_progress)
            
            return []
        
        anim = FuncAnimation(fig, animate, frames=self.n_frames,
                            interval=1000/self.fps, blit=False)
        anim.save('anim_vesica_piscis.gif', writer=PillowWriter(fps=self.fps),
                 dpi=100, savefig_kwargs={'facecolor': 'black'})
        plt.close()
        print("Saved: anim_vesica_piscis.gif")
        
    def create_unified_mandala_animation(self):
        """
        Unified mandala combining all sacred geometry elements.
        The ultimate animation showing the complete transformation.
        """
        print("Creating unified mandala animation...")
        
        fig, ax = plt.subplots(figsize=(12, 12), facecolor='black')
        
        def animate(frame):
            ax.clear()
            ax.set_facecolor('black')
            ax.set_xlim(-10, 10)
            ax.set_ylim(-10, 10)
            ax.axis('off')
            
            t = self.times[frame]
            state = get_implosion_state(t)
            R_shell = state['R']
            T = state['T']
            CR = state['CR']
            Y = state['Y']
            P = state['P_laser']
            
            # Normalize values for visualization
            compression = 1 - R_shell / R0  # 0 to ~0.97
            heat = min(T / T_PEAK, 1)
            burn = Y / 2.88
            
            # === LAYER 1: Outer ring (hohlraum) ===
            for r in [9.0, 9.2, 9.4]:
                circle = Circle((0, 0), r, fill=False, color='gold',
                               linewidth=1, alpha=0.4)
                ax.add_patch(circle)
            
            # === LAYER 2: 192 beam positions (activate with laser) ===
            if P > 10:
                beam_alpha = min(P / 500, 1) * 0.6
                for i in range(192):
                    angle = i * 2 * PI / 192
                    x, y = 9.1 * np.cos(angle), 9.1 * np.sin(angle)
                    ax.plot(x, y, '.', color='orange', markersize=2, alpha=beam_alpha)
                    
                    # Beam lines to shell
                    if i % 8 == 0:
                        r_shell = 8 * (R_shell / R0)
                        x2, y2 = r_shell * np.cos(angle), r_shell * np.sin(angle)
                        ax.plot([x, x2], [y, y2], color='gold', 
                               linewidth=0.5, alpha=beam_alpha * 0.5)
            
            # === LAYER 3: Flower of Life (scales with compression) ===
            R_flower = 2.0 + 4.0 * (1 - compression)
            pulse = 1 + 0.03 * np.sin(t * 8)
            R_f = R_flower * pulse
            
            flower_color = plt.cm.plasma(0.3 + 0.5 * heat)
            
            # Central 7 circles (seed of life)
            for i in range(6):
                angle = i * PI / 3
                cx, cy = R_f * 0.5 * np.cos(angle), R_f * 0.5 * np.sin(angle)
                circle = Circle((cx, cy), R_f * 0.5, fill=False,
                               color=flower_color, linewidth=1, alpha=0.6)
                ax.add_patch(circle)
            circle = Circle((0, 0), R_f * 0.5, fill=False,
                           color=flower_color, linewidth=1, alpha=0.6)
            ax.add_patch(circle)
            
            # Outer rings
            for ring in [1, 2]:
                for i in range(6):
                    angle = i * PI / 3 + (ring % 2) * PI / 6
                    cx = R_f * ring * np.cos(angle)
                    cy = R_f * ring * np.sin(angle)
                    circle = Circle((cx, cy), R_f * 0.5, fill=False,
                                   color=flower_color, linewidth=0.5, 
                                   alpha=0.4 - 0.1 * ring)
                    ax.add_patch(circle)
            
            # === LAYER 4: Sri Yantra triangles (emerge during compression) ===
            if compression > 0.3:
                sri_alpha = (compression - 0.3) / 0.7
                scales = [2.0, 1.5, 1.0, 0.7, 0.4]
                
                for i, scale in enumerate(scales):
                    s = scale * (1 + 0.5 * compression)
                    if i % 2 == 0:  # Up triangles
                        points = [(0, s), (-s*0.866, -s*0.5), (s*0.866, -s*0.5)]
                        color = plt.cm.hot(0.3 + 0.5 * heat)
                    else:  # Down triangles
                        points = [(0, -s), (-s*0.866, s*0.5), (s*0.866, s*0.5)]
                        color = plt.cm.cool(0.3 + 0.5 * heat)
                    
                    triangle = Polygon(points, fill=False, edgecolor=color,
                                       linewidth=1.5, alpha=sri_alpha * 0.7)
                    ax.add_patch(triangle)
            
            # === LAYER 5: Golden spiral (winds with compression) ===
            theta_spiral = np.linspace(0, 6 * PI * compression, int(500 * compression) + 10)
            r_spiral = 0.3 * PHI ** (theta_spiral / (PI/2))
            x_spiral = r_spiral * np.cos(theta_spiral)
            y_spiral = r_spiral * np.sin(theta_spiral)
            ax.plot(x_spiral, y_spiral, color='gold', linewidth=1.5, alpha=0.6)
            
            # === LAYER 6: Metatron's Cube connections (crystallize at stagnation) ===
            if compression > 0.6:
                meta_alpha = (compression - 0.6) / 0.4
                R_m = 1.5
                
                meta_centers = [(0, 0)]
                for i in range(6):
                    angle = i * PI / 3
                    meta_centers.append((R_m * np.cos(angle), R_m * np.sin(angle)))
                
                for i, (x1, y1) in enumerate(meta_centers):
                    for j, (x2, y2) in enumerate(meta_centers):
                        if i < j:
                            ax.plot([x1, x2], [y1, y2], color='cyan',
                                   linewidth=0.5, alpha=meta_alpha * 0.4)
            
            # === LAYER 7: Central bindu (ignition point) ===
            bindu_size = 0.1 + 0.3 * burn
            bindu_color = 'white' if burn > 0.5 else plt.cm.hot(heat)
            bindu = Circle((0, 0), bindu_size, fill=True, facecolor=bindu_color,
                          edgecolor='gold', linewidth=2, alpha=0.9)
            ax.add_patch(bindu)
            
            # === LAYER 8: Energy radiation (during burn) ===
            if burn > 0:
                # Radial rays
                for i in range(12):
                    angle = i * PI / 6 + t * 0.3
                    r1 = bindu_size
                    r2 = bindu_size + 3 * burn
                    x1, y1 = r1 * np.cos(angle), r1 * np.sin(angle)
                    x2, y2 = r2 * np.cos(angle), r2 * np.sin(angle)
                    ax.plot([x1, x2], [y1, y2], color='yellow',
                           linewidth=1 + 2 * burn, alpha=burn * 0.7)
                
                # Expanding rings
                for ring_r in [1, 2, 3]:
                    ring = Circle((0, 0), ring_r * burn * 2, fill=False,
                                 color='yellow', linewidth=1, alpha=burn * 0.3)
                    ax.add_patch(ring)
            
            # === Physics constants around border ===
            constants = [
                f'Q = {Q_IGNITION:.2f}',
                f'CR = {CR:.1f}',
                f'T = {T:.1f} keV',
                f'φ = {PHI:.3f}',
            ]
            for i, const in enumerate(constants):
                angle = i * PI / 2 - PI / 4
                x = 8.5 * np.cos(angle)
                y = 8.5 * np.sin(angle)
                ax.text(x, y, const, color='gold', fontsize=9, ha='center',
                       va='center', rotation=np.degrees(angle) + 90, alpha=0.7)
            
            # === Title ===
            ax.set_title(f'UNIFIED MANDALA OF STELLAR IGNITION\n'
                        f't = {t:.1f} ns  |  R = {R_shell:.0f} μm  |  '
                        f'T = {T:.1f} keV  |  Y = {Y:.2f} MJ',
                        color='gold', fontsize=12, pad=10, fontweight='bold')
            
            # Phase indicator
            if t < 2:
                phase = "FOOT PULSE"
            elif t < 10:
                phase = "ACCELERATION"
            elif t < 12:
                phase = "COMPRESSION"
            elif t < 12.5:
                phase = "STAGNATION"
            elif Y < 2.05:
                phase = "BURN"
            else:
                phase = "★ IGNITION ★"
            
            ax.text(0, -9.5, phase, color='cyan' if 'IGNITION' not in phase else 'yellow',
                   fontsize=14, ha='center', fontweight='bold')
            
            return []
        
        anim = FuncAnimation(fig, animate, frames=self.n_frames,
                            interval=1000/self.fps, blit=False)
        anim.save('anim_unified_mandala.gif', writer=PillowWriter(fps=self.fps),
                 dpi=100, savefig_kwargs={'facecolor': 'black'})
        plt.close()
        print("Saved: anim_unified_mandala.gif")
        
    def run_all(self):
        """Generate all animations."""
        print()
        print("╔" + "═" * 62 + "╗")
        print("║  ANIMATED SACRED GEOMETRY FROM FUSION PHYSICS               ║")
        print("║  The Mathematics of Stellar Creation, in Motion             ║")
        print("╚" + "═" * 62 + "╝")
        print()
        print(f"Duration: {self.duration} ns  |  FPS: {self.fps}  |  "
              f"Frames: {self.n_frames}")
        print()
        
        self.create_golden_spiral_animation()
        self.create_flower_of_life_animation()
        self.create_metatrons_cube_animation()
        self.create_vesica_piscis_animation()
        self.create_unified_mandala_animation()
        
        print()
        print("═" * 64)
        print()
        print("Generated animations:")
        print("  1. anim_golden_spiral.gif    - Compression as golden spiral")
        print("  2. anim_flower_of_life.gif   - Pulsing spherical harmonics")
        print("  3. anim_metatrons_cube.gif   - Crystallizing beam geometry")
        print("  4. anim_vesica_piscis.gif    - D-T fusion portal opening")
        print("  5. anim_unified_mandala.gif  - All elements combined")
        print()
        print("'The geometry of the cosmos breathes with the physics of creation.'")
        print()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Create animations at 20 fps for reasonable file size
    animator = AnimatedSacredGeometry(duration_ns=14, fps=20)
    animator.run_all()
