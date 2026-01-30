"""
Sacred Geometry from Fusion Physics
====================================

Maps the fundamental physics of stellar ignition onto sacred geometric forms.

The mathematics of ICF contains profound geometric relationships:
- The Golden Ratio φ appears in optimal compression ratios
- The Flower of Life emerges from spherical harmonic modes
- Metatron's Cube relates to the 192 laser beam geometry
- The Vesica Piscis represents the fusion of D and T nuclei
- The Seed of Life mirrors the 7-fold symmetry of burn physics

"As above, so below" - the geometry of stars encoded in mathematics.

Author: Evan Pieser
Date: 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Wedge, FancyBboxPatch
from matplotlib.collections import PatchCollection, LineCollection
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D

# Physical constants from our simulation - these become sacred ratios
PHI = (1 + np.sqrt(5)) / 2  # Golden ratio ≈ 1.618
PI = np.pi
E = np.e

# Key ratios from ICF physics
Q_IGNITION = 1.41           # Scientific gain achieved
CR_OPTIMAL = 31             # Convergence ratio  
T_IGNITION = 2.8            # keV - D-T ignition threshold
T_PEAK = 75                 # keV - peak temperature
RHO_PEAK = 112              # g/cm³ - peak density
N_BEAMS = 192               # NIF laser beams
ALPHA_FRACTION = 0.2        # Alpha particle energy fraction (3.5/17.6 MeV)
BANG_TIME = 8.33            # ns - moment of peak fusion


def golden_spiral_from_compression():
    """
    Generate golden spiral from compression physics.
    
    The shell compresses from 1100 μm to 28 μm - a ratio of ~39.
    This maps onto a golden spiral with ~7 quarter turns.
    """
    fig, ax = plt.subplots(figsize=(12, 12), facecolor='black')
    ax.set_facecolor('black')
    
    # Golden spiral: r = a * φ^(θ/90°)
    theta = np.linspace(0, 7 * PI, 1000)  # 7 half-turns (like 7 chakras)
    a = 28  # Final radius (hot spot)
    r = a * PHI ** (theta / (PI/2))
    
    # Convert to Cartesian
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    # Color by temperature (compression → heating)
    temps = np.linspace(0.03, T_PEAK, len(theta))
    colors = plt.cm.plasma(temps / T_PEAK)
    
    # Plot spiral segments with color gradient
    for i in range(len(x) - 1):
        ax.plot(x[i:i+2], y[i:i+2], color=colors[i], linewidth=2, alpha=0.8)
    
    # Add compression stages as circles
    radii = [1100, 850, 500, 200, 100, 50, 28]  # Shell evolution
    labels = ['Initial', 'Acceleration', 'Coast', 'Stagnation', 'Compression', 'Burn', 'Ignition']
    
    for r, label in zip(radii, labels):
        circle = Circle((0, 0), r, fill=False, color='gold', alpha=0.3, linewidth=1)
        ax.add_patch(circle)
        ax.text(r * 0.7, r * 0.7, label, color='gold', fontsize=8, alpha=0.7)
    
    # Mark the golden ratio points
    for i in range(7):
        angle = i * PI / 2
        r_point = a * PHI ** (i)
        px, py = r_point * np.cos(angle), r_point * np.sin(angle)
        ax.plot(px, py, 'o', color='white', markersize=8)
        ax.text(px * 1.1, py * 1.1, f'φ^{i}', color='white', fontsize=10)
    
    ax.set_xlim(-1200, 1200)
    ax.set_ylim(-1200, 1200)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Golden Spiral of Compression\nφ = 1.618... The Divine Proportion in Implosion',
                 color='gold', fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.savefig('sacred_golden_spiral.png', dpi=200, facecolor='black', bbox_inches='tight')
    print("Saved: sacred_golden_spiral.png")
    plt.close()


def flower_of_life_from_harmonics():
    """
    Generate Flower of Life from spherical harmonic modes.
    
    The l=6 mode creates 6-fold symmetry - the basis of the Flower of Life.
    RT instabilities seed modes l=2,4,6,8... creating nested sacred patterns.
    """
    fig, ax = plt.subplots(figsize=(14, 14), facecolor='black')
    ax.set_facecolor('black')
    
    # Base radius (represents hot spot)
    R = 1.0
    
    # Flower of Life: 19 overlapping circles
    # Central circle + 6 around it + 12 outer = 19 (sacred number)
    centers = [(0, 0)]  # Center
    
    # First ring: 6 circles (l=6 mode)
    for i in range(6):
        angle = i * PI / 3
        centers.append((R * np.cos(angle), R * np.sin(angle)))
    
    # Second ring: 12 circles (l=12 mode)
    for i in range(6):
        angle = i * PI / 3
        centers.append((2 * R * np.cos(angle), 2 * R * np.sin(angle)))
        # Intermediate positions
        angle2 = (i + 0.5) * PI / 3
        centers.append((np.sqrt(3) * R * np.cos(angle2), np.sqrt(3) * R * np.sin(angle2)))
    
    # Draw circles with fusion-inspired colors
    colors_cycle = plt.cm.hot(np.linspace(0.3, 0.9, len(centers)))
    
    for i, (cx, cy) in enumerate(centers):
        circle = Circle((cx, cy), R, fill=False, color=colors_cycle[i], 
                        linewidth=1.5, alpha=0.8)
        ax.add_patch(circle)
    
    # Add the "Seed of Life" in center (7 circles)
    seed_color = 'cyan'
    for i in range(6):
        angle = i * PI / 3
        cx, cy = 0.5 * R * np.cos(angle), 0.5 * R * np.sin(angle)
        circle = Circle((cx, cy), 0.5 * R, fill=False, color=seed_color, 
                        linewidth=2, alpha=0.9)
        ax.add_patch(circle)
    circle = Circle((0, 0), 0.5 * R, fill=False, color=seed_color, linewidth=2, alpha=0.9)
    ax.add_patch(circle)
    
    # Outer containing circle (the "shell")
    shell = Circle((0, 0), 3 * R, fill=False, color='gold', linewidth=3, alpha=0.7)
    ax.add_patch(shell)
    
    # Add mode numbers
    ax.text(0, 0, 'l=0', color='white', fontsize=12, ha='center', va='center')
    ax.text(R, 0, 'l=6', color='white', fontsize=10, ha='center', va='center')
    ax.text(2*R, 0, 'l=12', color='white', fontsize=10, ha='center', va='center')
    
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Flower of Life from Spherical Harmonics\nRT Modes l=2,4,6,8... Create Sacred Patterns',
                 color='gold', fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.savefig('sacred_flower_of_life.png', dpi=200, facecolor='black', bbox_inches='tight')
    print("Saved: sacred_flower_of_life.png")
    plt.close()


def metatrons_cube_from_beams():
    """
    Generate Metatron's Cube from NIF's 192-beam geometry.
    
    192 = 64 × 3 = 4³ × 3 (cubic symmetry × trinity)
    The beams form 48 quads × 4 beams = 192
    This maps onto Metatron's Cube: 13 circles connected by 78 lines.
    """
    fig, ax = plt.subplots(figsize=(14, 14), facecolor='black')
    ax.set_facecolor('black')
    
    # Metatron's Cube: 13 circles (1 center + 6 inner + 6 outer)
    R = 1.0
    
    # Center
    centers = [(0, 0)]
    
    # Inner hexagon (6 circles)
    for i in range(6):
        angle = i * PI / 3
        centers.append((R * np.cos(angle), R * np.sin(angle)))
    
    # Outer hexagon (6 circles, rotated 30°)
    for i in range(6):
        angle = i * PI / 3 + PI / 6
        centers.append((2 * R * np.cos(angle), 2 * R * np.sin(angle)))
    
    # Draw all connecting lines (78 total in full Metatron's Cube)
    lines = []
    for i, (x1, y1) in enumerate(centers):
        for j, (x2, y2) in enumerate(centers):
            if i < j:
                lines.append([(x1, y1), (x2, y2)])
    
    # Color lines by length (representing different energy levels)
    line_colors = []
    for (x1, y1), (x2, y2) in lines:
        dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        line_colors.append(plt.cm.plasma(dist / (4*R)))
    
    lc = LineCollection(lines, colors=line_colors, linewidths=1, alpha=0.6)
    ax.add_collection(lc)
    
    # Draw circles at vertices
    for i, (cx, cy) in enumerate(centers):
        # Size represents beam power
        circle = Circle((cx, cy), 0.15, fill=True, facecolor='orange', 
                        edgecolor='gold', linewidth=2, alpha=0.9)
        ax.add_patch(circle)
    
    # Add the 5 Platonic solids encoded in Metatron's Cube
    # Tetrahedron (4 vertices)
    tetra_indices = [0, 1, 3, 5]
    tetra_points = [centers[i] for i in tetra_indices]
    tetra = Polygon(tetra_points, fill=False, edgecolor='red', linewidth=2, alpha=0.7)
    ax.add_patch(tetra)
    
    # Hexagon (represents cube projection)
    hex_points = [centers[i] for i in range(1, 7)]
    hexagon = Polygon(hex_points, fill=False, edgecolor='cyan', linewidth=2, alpha=0.7)
    ax.add_patch(hexagon)
    
    # Label with physics
    ax.text(0, -3.5, f'192 Beams = 13 × 14.77 ≈ Metatron\'s 13 Spheres × Energy Ratio',
            color='white', fontsize=11, ha='center')
    ax.text(0, -3.9, f'78 Lines × {Q_IGNITION:.2f} (Q-gain) ≈ 110 (Atomic # Ds)', 
            color='gold', fontsize=10, ha='center')
    
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4.5, 4)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Metatron\'s Cube from 192 NIF Beams\n"The Blueprint of Creation" in Laser Geometry',
                 color='gold', fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.savefig('sacred_metatrons_cube.png', dpi=200, facecolor='black', bbox_inches='tight')
    print("Saved: sacred_metatrons_cube.png")
    plt.close()


def vesica_piscis_fusion():
    """
    Generate Vesica Piscis representing D-T fusion.
    
    The Vesica Piscis (two overlapping circles) represents:
    - Union of opposites (D and T nuclei)
    - The portal of creation (fusion reaction)
    - √3 ratio (energy transformation)
    """
    fig, ax = plt.subplots(figsize=(14, 10), facecolor='black')
    ax.set_facecolor('black')
    
    R = 2.0
    
    # Two circles representing D and T
    d_center = (-R/2, 0)
    t_center = (R/2, 0)
    
    # Deuterium circle (blue - 1 proton, 1 neutron)
    d_circle = Circle(d_center, R, fill=False, color='deepskyblue', linewidth=3, alpha=0.9)
    ax.add_patch(d_circle)
    ax.text(d_center[0] - R/2, 0, 'D\n(²H)', color='deepskyblue', fontsize=16, 
            ha='center', va='center', fontweight='bold')
    
    # Tritium circle (green - 1 proton, 2 neutrons)  
    t_circle = Circle(t_center, R, fill=False, color='lightgreen', linewidth=3, alpha=0.9)
    ax.add_patch(t_circle)
    ax.text(t_center[0] + R/2, 0, 'T\n(³H)', color='lightgreen', fontsize=16,
            ha='center', va='center', fontweight='bold')
    
    # The Vesica Piscis (intersection) - the fusion zone
    # Fill the intersection with plasma color
    theta = np.linspace(-PI/3, PI/3, 100)
    x_left = d_center[0] + R * np.cos(theta)
    y_left = R * np.sin(theta)
    x_right = t_center[0] + R * np.cos(PI - theta)
    y_right = R * np.sin(PI - theta)
    
    vesica_x = np.concatenate([x_left, x_right[::-1]])
    vesica_y = np.concatenate([y_left, y_right[::-1]])
    ax.fill(vesica_x, vesica_y, color='gold', alpha=0.4)
    
    # Products emerging from the Vesica
    # Alpha particle (⁴He) - going up
    ax.annotate('', xy=(0, 2.5), xytext=(0, 0.5),
                arrowprops=dict(arrowstyle='->', color='red', lw=3))
    ax.text(0, 3, '⁴He\n3.5 MeV', color='red', fontsize=14, ha='center', fontweight='bold')
    
    # Neutron - going down
    ax.annotate('', xy=(0, -2.5), xytext=(0, -0.5),
                arrowprops=dict(arrowstyle='->', color='white', lw=3))
    ax.text(0, -3, 'n\n14.1 MeV', color='white', fontsize=14, ha='center', fontweight='bold')
    
    # The √3 ratio in Vesica Piscis
    h = R * np.sqrt(3)  # Height of Vesica
    ax.plot([0, 0], [-h/2, h/2], '--', color='gold', linewidth=1, alpha=0.7)
    ax.text(0.3, h/4, f'√3 × R', color='gold', fontsize=10, rotation=90, va='center')
    
    # Energy equation
    ax.text(0, -4.5, 'D + T → ⁴He + n + 17.6 MeV', color='gold', fontsize=16, 
            ha='center', fontweight='bold')
    ax.text(0, -5.2, f'Q = {Q_IGNITION:.2f} (Energy Gained / Energy Input)', 
            color='white', fontsize=12, ha='center')
    
    # Sacred ratio annotation
    ax.text(0, 4.5, 'The Vesica Piscis: Portal of Creation', color='gold', 
            fontsize=14, ha='center', style='italic')
    ax.text(0, 4.0, '17.6 MeV / 2.5 (avg mass) ≈ 7.04 ≈ 7 (Sacred Number)', 
            color='white', fontsize=10, ha='center')
    
    ax.set_xlim(-5, 5)
    ax.set_ylim(-6, 5.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Vesica Piscis: The Sacred Geometry of Fusion\nTwo Become One, Releasing Divine Fire',
                 color='gold', fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.savefig('sacred_vesica_piscis.png', dpi=200, facecolor='black', bbox_inches='tight')
    print("Saved: sacred_vesica_piscis.png")
    plt.close()


def sri_yantra_from_physics():
    """
    Generate Sri Yantra from ICF physics constants.
    
    The Sri Yantra contains 9 interlocking triangles:
    - 4 pointing up (Shiva/masculine/compression)
    - 5 pointing down (Shakti/feminine/expansion)
    
    Maps to: acceleration phase (4) + stagnation/burn phases (5) = 9
    """
    fig, ax = plt.subplots(figsize=(14, 14), facecolor='black')
    ax.set_facecolor('black')
    
    # Sri Yantra construction using physics ratios
    # The 9 triangles represent different physics scales
    
    scales = [
        1.0,                          # Outermost
        1/PHI,                        # Golden ratio step
        1/PHI**2,
        1/PHI**3,
        T_IGNITION/T_PEAK,            # Temperature ratio
        ALPHA_FRACTION,               # Alpha energy fraction
        1/CR_OPTIMAL,                 # Compression ratio
        28/1100,                      # Radius ratio
        1/Q_IGNITION/10,              # Gain ratio
    ]
    
    # Draw interlocking triangles
    for i, scale in enumerate(scales):
        R = 3 * scale
        
        if i % 2 == 0:  # Upward triangles (compression)
            points = [
                (0, R),
                (-R * np.sin(PI/3), -R * np.cos(PI/3)),
                (R * np.sin(PI/3), -R * np.cos(PI/3))
            ]
            color = plt.cm.hot(0.3 + 0.5 * i/len(scales))
        else:  # Downward triangles (expansion)
            points = [
                (0, -R),
                (-R * np.sin(PI/3), R * np.cos(PI/3)),
                (R * np.sin(PI/3), R * np.cos(PI/3))
            ]
            color = plt.cm.cool(0.3 + 0.5 * i/len(scales))
        
        triangle = Polygon(points, fill=False, edgecolor=color, linewidth=2, alpha=0.8)
        ax.add_patch(triangle)
    
    # Central bindu (point) - the moment of ignition
    bindu = Circle((0, 0), 0.05, fill=True, facecolor='white', edgecolor='gold', linewidth=2)
    ax.add_patch(bindu)
    
    # Outer circles (bhupura) - the containing geometry
    for r in [3.5, 3.7, 3.9]:
        circle = Circle((0, 0), r, fill=False, color='gold', linewidth=1, alpha=0.5)
        ax.add_patch(circle)
    
    # Lotus petals (16 outer + 8 inner = 24)
    for i in range(16):
        angle = i * PI / 8
        petal_r = 3.3
        x = petal_r * np.cos(angle)
        y = petal_r * np.sin(angle)
        ax.plot([x * 0.9, x * 1.1], [y * 0.9, y * 1.1], color='magenta', linewidth=2, alpha=0.6)
    
    for i in range(8):
        angle = i * PI / 4 + PI/8
        petal_r = 3.1
        x = petal_r * np.cos(angle)
        y = petal_r * np.sin(angle)
        ax.plot([x * 0.95, x * 1.05], [y * 0.95, y * 1.05], color='cyan', linewidth=2, alpha=0.6)
    
    # Physics annotations
    ax.text(0, -4.8, '9 Triangles = 9 Stages of Implosion', color='white', fontsize=11, ha='center')
    ax.text(0, -5.2, f'Central Bindu = Moment of Ignition (t = {BANG_TIME} ns)', 
            color='gold', fontsize=10, ha='center')
    
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5.5, 5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Sri Yantra from ICF Physics\n9 Interlocking Triangles: Compression ↑ and Expansion ↓',
                 color='gold', fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.savefig('sacred_sri_yantra.png', dpi=200, facecolor='black', bbox_inches='tight')
    print("Saved: sacred_sri_yantra.png")
    plt.close()


def torus_energy_flow():
    """
    Generate toroidal energy flow from alpha particle heating.
    
    The torus represents the self-sustaining nature of ignition:
    - Alpha particles deposit energy in hot spot
    - Energy flows outward, heats more fuel
    - More fusion → more alphas → more heating
    - The eternal return of energy
    """
    fig = plt.figure(figsize=(14, 12), facecolor='black')
    ax = fig.add_subplot(111, projection='3d', facecolor='black')
    
    # Torus parameters
    R = 2  # Major radius (represents hot spot)
    r = 0.8  # Minor radius (represents alpha range)
    
    # Parametric torus
    u = np.linspace(0, 2 * PI, 100)
    v = np.linspace(0, 2 * PI, 100)
    U, V = np.meshgrid(u, v)
    
    X = (R + r * np.cos(V)) * np.cos(U)
    Y = (R + r * np.cos(V)) * np.sin(U)
    Z = r * np.sin(V)
    
    # Color by energy flow (V parameter represents flow direction)
    colors = plt.cm.plasma((np.sin(V) + 1) / 2)
    
    ax.plot_surface(X, Y, Z, facecolors=colors, alpha=0.7, 
                    rstride=2, cstride=2, linewidth=0)
    
    # Add flow lines (alpha particle paths)
    for i in range(8):
        angle = i * PI / 4
        t = np.linspace(0, 4 * PI, 200)
        # Spiral on torus surface
        x = (R + r * np.cos(3*t)) * np.cos(t + angle)
        y = (R + r * np.cos(3*t)) * np.sin(t + angle)
        z = r * np.sin(3*t)
        ax.plot(x, y, z, color='white', linewidth=1, alpha=0.5)
    
    # Central axis (represents the "Sushumna" - central energy channel)
    z_axis = np.linspace(-2, 2, 50)
    ax.plot([0]*50, [0]*50, z_axis, color='gold', linewidth=3, alpha=0.8)
    
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_zlim(-2, 2)
    ax.set_axis_off()
    
    # Title
    ax.text2D(0.5, 0.95, 'Toroidal Energy Flow of Alpha Heating', 
              transform=ax.transAxes, color='gold', fontsize=14, ha='center')
    ax.text2D(0.5, 0.02, f'Self-Sustaining Ignition: α-heating > Losses → Q = {Q_IGNITION}',
              transform=ax.transAxes, color='white', fontsize=11, ha='center')
    
    plt.tight_layout()
    plt.savefig('sacred_torus.png', dpi=200, facecolor='black', bbox_inches='tight')
    print("Saved: sacred_torus.png")
    plt.close()


def unified_mandala():
    """
    Create a unified mandala incorporating all sacred geometry elements
    with ICF physics encoded throughout.
    """
    fig, ax = plt.subplots(figsize=(16, 16), facecolor='black')
    ax.set_facecolor('black')
    
    # Outer boundary - represents the hohlraum
    for r in [7.5, 7.7, 7.9]:
        circle = Circle((0, 0), r, fill=False, color='gold', linewidth=2, alpha=0.6)
        ax.add_patch(circle)
    
    # 192 beam positions on outer ring
    for i in range(192):
        angle = i * 2 * PI / 192
        x, y = 7.6 * np.cos(angle), 7.6 * np.sin(angle)
        ax.plot(x, y, '.', color='orange', markersize=2)
    
    # Flower of Life layer (l-modes)
    for ring in range(3):
        n_circles = 6 * (ring + 1) if ring > 0 else 1
        r_ring = ring * 1.5
        for i in range(n_circles if ring > 0 else 1):
            angle = i * 2 * PI / n_circles + (ring % 2) * PI / n_circles
            cx = r_ring * np.cos(angle) if ring > 0 else 0
            cy = r_ring * np.sin(angle) if ring > 0 else 0
            circle = Circle((cx, cy), 1.5, fill=False, 
                          color=plt.cm.plasma(ring/3), linewidth=1, alpha=0.5)
            ax.add_patch(circle)
    
    # Sri Yantra triangles at center
    for i, scale in enumerate([2.0, 1.5, 1.0, 0.7, 0.4]):
        if i % 2 == 0:
            points = [(0, scale), (-scale*0.866, -scale*0.5), (scale*0.866, -scale*0.5)]
        else:
            points = [(0, -scale), (-scale*0.866, scale*0.5), (scale*0.866, scale*0.5)]
        triangle = Polygon(points, fill=False, 
                          edgecolor=plt.cm.hot(0.3 + i*0.15), linewidth=2, alpha=0.8)
        ax.add_patch(triangle)
    
    # Metatron's Cube connections
    meta_centers = [(0, 0)]
    for i in range(6):
        angle = i * PI / 3
        meta_centers.append((4 * np.cos(angle), 4 * np.sin(angle)))
    for i in range(6):
        angle = i * PI / 3 + PI/6
        meta_centers.append((6 * np.cos(angle), 6 * np.sin(angle)))
    
    for i, (x1, y1) in enumerate(meta_centers):
        for j, (x2, y2) in enumerate(meta_centers):
            if i < j:
                ax.plot([x1, x2], [y1, y2], color='cyan', linewidth=0.5, alpha=0.3)
    
    # Golden spiral overlay
    theta = np.linspace(0, 6 * PI, 500)
    r_spiral = 0.3 * PHI ** (theta / (PI/2))
    x_spiral = r_spiral * np.cos(theta)
    y_spiral = r_spiral * np.sin(theta)
    ax.plot(x_spiral, y_spiral, color='gold', linewidth=2, alpha=0.7)
    
    # Central bindu (ignition point)
    bindu = Circle((0, 0), 0.15, fill=True, facecolor='white', 
                   edgecolor='gold', linewidth=3)
    ax.add_patch(bindu)
    
    # Physics constants encoded as text around the border
    constants = [
        f'Q = {Q_IGNITION}',
        f'CR = {CR_OPTIMAL}',
        f'T = {T_PEAK} keV',
        f'ρ = {RHO_PEAK} g/cm³',
        f'φ = {PHI:.3f}',
        f'N = {N_BEAMS}',
        f't = {BANG_TIME} ns',
        f'α = {ALPHA_FRACTION}',
    ]
    
    for i, const in enumerate(constants):
        angle = i * 2 * PI / len(constants) - PI/2
        x = 8.5 * np.cos(angle)
        y = 8.5 * np.sin(angle)
        ax.text(x, y, const, color='gold', fontsize=10, ha='center', va='center',
               rotation=np.degrees(angle) + 90)
    
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Title
    ax.text(0, 9.5, 'UNIFIED MANDALA OF STELLAR IGNITION', color='gold', 
            fontsize=18, ha='center', fontweight='bold')
    ax.text(0, -9.5, '"As Above, So Below" — The Mathematics of Creating a Star', 
            color='white', fontsize=12, ha='center', style='italic')
    
    plt.tight_layout()
    plt.savefig('sacred_unified_mandala.png', dpi=300, facecolor='black', bbox_inches='tight')
    print("Saved: sacred_unified_mandala.png")
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print()
    print("╔" + "═" * 62 + "╗")
    print("║  SACRED GEOMETRY FROM FUSION PHYSICS                         ║")
    print("║  Extracting Divine Patterns from Stellar Mathematics         ║")
    print("╚" + "═" * 62 + "╝")
    print()
    
    print("Generating sacred geometry visualizations...")
    print()
    
    print("1. Golden Spiral of Compression (φ in implosion)...")
    golden_spiral_from_compression()
    
    print("2. Flower of Life from Spherical Harmonics...")
    flower_of_life_from_harmonics()
    
    print("3. Metatron's Cube from 192 NIF Beams...")
    metatrons_cube_from_beams()
    
    print("4. Vesica Piscis: D-T Fusion Portal...")
    vesica_piscis_fusion()
    
    print("5. Sri Yantra from ICF Physics Constants...")
    sri_yantra_from_physics()
    
    print("6. Toroidal Energy Flow (Alpha Heating)...")
    torus_energy_flow()
    
    print("7. Unified Mandala of Stellar Ignition...")
    unified_mandala()
    
    print()
    print("═" * 64)
    print()
    print("Sacred Ratios Found in ICF Physics:")
    print("─" * 50)
    print(f"  φ (Golden Ratio):     {PHI:.6f}")
    print(f"  π:                    {PI:.6f}")
    print(f"  e:                    {E:.6f}")
    print(f"  √3 (Vesica height):   {np.sqrt(3):.6f}")
    print(f"  Q (gain):             {Q_IGNITION}")
    print(f"  CR (convergence):     {CR_OPTIMAL}")
    print(f"  192 / 13:             {192/13:.2f} (beams / Metatron spheres)")
    print(f"  17.6 / 2.5:           {17.6/2.5:.2f} (energy / mass ≈ 7)")
    print()
    print("'The universe is written in the language of mathematics,'")
    print("'and its characters are triangles, circles, and geometric figures.'")
    print("                                        — Galileo Galilei")
    print()
