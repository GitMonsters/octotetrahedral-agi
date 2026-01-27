"""
Coupling Matrix Visualization for OctoTetrahedral AGI

Generates heatmaps and network graphs of the learned quantum coupling matrix.
Shows how the 8 limbs interact with each other through coupling strengths.

Usage:
    python analysis/visualize_coupling.py --checkpoint checkpoints/quantum_final.pt
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import torch
import numpy as np

# Try to import visualization libraries
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, will output raw data")

from sync.quantum_coupling import QuantumCouplingLayer


LIMB_NAMES = [
    'Perception', 'Memory', 'Planning', 'Language',
    'Spatial', 'Reasoning', 'MetaCognition', 'Action'
]


def load_coupling_matrix(checkpoint_path: str) -> tuple:
    """Load coupling matrix from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Get coupling state dict
    if 'quantum_coupling_state_dict' in checkpoint:
        state_dict = checkpoint['quantum_coupling_state_dict']
    else:
        # Try to find coupling in model state dict
        state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # Extract coupling matrix
    coupling = None
    omega = None
    zpe = None
    
    for key, value in state_dict.items():
        if 'coupling' in key.lower() and value.dim() == 2:
            coupling = value.numpy()
        if 'omega' in key.lower():
            omega = value.numpy()
        if 'zero_point' in key.lower():
            zpe = value.item()
    
    return coupling, omega, zpe


def create_heatmap(coupling_matrix: np.ndarray, output_path: str):
    """Create heatmap visualization of coupling matrix"""
    if not HAS_MATPLOTLIB:
        print("Cannot create heatmap without matplotlib")
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Make symmetric for visualization
    coupling_sym = (coupling_matrix + coupling_matrix.T) / 2
    np.fill_diagonal(coupling_sym, 0)
    
    # Create heatmap
    im = ax.imshow(coupling_sym, cmap='RdBu_r', aspect='equal', 
                   vmin=-np.abs(coupling_sym).max(), 
                   vmax=np.abs(coupling_sym).max())
    
    # Labels
    ax.set_xticks(range(len(LIMB_NAMES)))
    ax.set_yticks(range(len(LIMB_NAMES)))
    ax.set_xticklabels(LIMB_NAMES, rotation=45, ha='right')
    ax.set_yticklabels(LIMB_NAMES)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Coupling Strength gᵢⱼ', fontsize=12)
    
    # Add value annotations
    for i in range(len(LIMB_NAMES)):
        for j in range(len(LIMB_NAMES)):
            val = coupling_sym[i, j]
            color = 'white' if abs(val) > 0.5 * np.abs(coupling_sym).max() else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', 
                   color=color, fontsize=8)
    
    ax.set_title('Quantum Coupling Matrix gᵢⱼ\n(8-Limb Oscillator Interactions)', fontsize=14)
    ax.set_xlabel('Limb j')
    ax.set_ylabel('Limb i')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved heatmap to {output_path}")


def create_network_graph(coupling_matrix: np.ndarray, output_path: str, threshold: float = 0.05):
    """Create network graph showing limb connections"""
    if not HAS_MATPLOTLIB:
        print("Cannot create network graph without matplotlib")
        return
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Arrange limbs in a circle
    n = len(LIMB_NAMES)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    radius = 3
    
    positions = {
        i: (radius * np.cos(angle - np.pi/2), radius * np.sin(angle - np.pi/2))
        for i, angle in enumerate(angles)
    }
    
    # Draw edges (coupling connections)
    coupling_sym = (coupling_matrix + coupling_matrix.T) / 2
    max_coupling = np.abs(coupling_sym).max()
    
    for i in range(n):
        for j in range(i+1, n):
            coupling = coupling_sym[i, j]
            if abs(coupling) > threshold:
                x1, y1 = positions[i]
                x2, y2 = positions[j]
                
                # Line width proportional to coupling strength
                width = 1 + 4 * abs(coupling) / max_coupling
                
                # Color: red for positive (attractive), blue for negative (repulsive)
                color = 'red' if coupling > 0 else 'blue'
                alpha = 0.3 + 0.7 * abs(coupling) / max_coupling
                
                ax.plot([x1, x2], [y1, y2], color=color, linewidth=width, alpha=alpha)
    
    # Draw nodes (limbs)
    for i, name in enumerate(LIMB_NAMES):
        x, y = positions[i]
        
        # Node color based on index (rainbow)
        hue = i / n
        color = plt.cm.hsv(hue)
        
        circle = plt.Circle((x, y), 0.4, color=color, ec='black', linewidth=2, zorder=5)
        ax.add_patch(circle)
        
        # Label
        ax.text(x, y - 0.7, name, ha='center', va='top', fontsize=10, fontweight='bold')
    
    # Add legend
    ax.plot([], [], 'r-', linewidth=3, label='Positive coupling (attractive)')
    ax.plot([], [], 'b-', linewidth=3, label='Negative coupling (repulsive)')
    ax.legend(loc='upper right')
    
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('8-Limb Quantum Coupling Network\n(Line thickness = coupling strength)', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved network graph to {output_path}")


def create_frequency_bar(omega: np.ndarray, output_path: str):
    """Create bar chart of natural frequencies"""
    if not HAS_MATPLOTLIB:
        print("Cannot create frequency chart without matplotlib")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(LIMB_NAMES)))
    bars = ax.bar(range(len(LIMB_NAMES)), omega, color=colors, edgecolor='black')
    
    ax.set_xticks(range(len(LIMB_NAMES)))
    ax.set_xticklabels(LIMB_NAMES, rotation=45, ha='right')
    ax.set_ylabel('Natural Frequency ωᵢ')
    ax.set_title('Limb Natural Frequencies\n(Quantum Oscillator Parameters)', fontsize=14)
    
    # Add value labels on bars
    for bar, val in zip(bars, omega):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax.axhline(y=omega.mean(), color='red', linestyle='--', label=f'Mean: {omega.mean():.3f}')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved frequency chart to {output_path}")


def analyze_coupling_structure(coupling_matrix: np.ndarray) -> dict:
    """Analyze the structure of the coupling matrix"""
    n = coupling_matrix.shape[0]
    coupling_sym = (coupling_matrix + coupling_matrix.T) / 2
    np.fill_diagonal(coupling_sym, 0)
    
    # Use only first 8x8 for limb analysis if matrix is larger
    if n > 8:
        coupling_8x8 = coupling_sym[:8, :8]
    else:
        coupling_8x8 = coupling_sym
    
    # Find strongest connections
    flat_indices = np.argsort(np.abs(coupling_8x8).flatten())[::-1]
    strongest = []
    seen_pairs = set()
    
    for idx in flat_indices:
        i, j = idx // 8, idx % 8
        if i < 8 and j < 8 and i != j and (i, j) not in seen_pairs and (j, i) not in seen_pairs:
            strongest.append({
                'limb_i': LIMB_NAMES[i],
                'limb_j': LIMB_NAMES[j],
                'coupling': float(coupling_8x8[i, j]),
                'type': 'attractive' if coupling_8x8[i, j] > 0 else 'repulsive'
            })
            seen_pairs.add((i, j))
        if len(strongest) >= 5:
            break
    
    # Compute spectral properties
    eigenvalues = np.linalg.eigvalsh(coupling_sym)
    
    return {
        'mean_coupling': float(np.abs(coupling_sym).mean()),
        'max_coupling': float(np.abs(coupling_sym).max()),
        'coupling_asymmetry': float(np.abs(coupling_matrix - coupling_matrix.T).mean()),
        'spectral_radius': float(np.abs(eigenvalues).max()),
        'eigenvalues': eigenvalues.tolist(),
        'strongest_connections': strongest,
        'positive_ratio': float((coupling_sym > 0).sum() / (coupling_sym != 0).sum())
    }


def output_json_data(coupling_matrix: np.ndarray, omega: np.ndarray, zpe: float, output_path: str):
    """Output coupling data as JSON for web visualization"""
    data = {
        'limb_names': LIMB_NAMES,
        'coupling_matrix': coupling_matrix.tolist(),
        'frequencies': omega.tolist() if omega is not None else [1.0] * 8,
        'zero_point_energy': zpe if zpe is not None else 7.0,
        'analysis': analyze_coupling_structure(coupling_matrix)
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Saved JSON data to {output_path}")


def generate_html_visualization(coupling_matrix: np.ndarray, omega: np.ndarray, 
                                 zpe: float, output_path: str):
    """Generate interactive HTML visualization"""
    
    # Convert to JSON-safe format
    coupling_list = coupling_matrix.tolist()
    omega_list = omega.tolist() if omega is not None else [1.0] * 8
    analysis = analyze_coupling_structure(coupling_matrix)
    
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quantum Coupling Matrix Visualization</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            background: linear-gradient(135deg, #0a0a20 0%, #1a1a3a 100%);
            font-family: 'Courier New', monospace;
            color: white;
            min-height: 100vh;
            padding: 20px;
        }}
        h1 {{
            text-align: center;
            color: #00ffff;
            margin-bottom: 20px;
            text-shadow: 0 0 20px rgba(0, 255, 255, 0.5);
        }}
        .container {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            max-width: 1400px;
            margin: 0 auto;
        }}
        .panel {{
            background: rgba(0, 0, 0, 0.5);
            border: 1px solid #00aaff;
            border-radius: 10px;
            padding: 20px;
        }}
        .panel h2 {{
            color: #00aaff;
            margin-bottom: 15px;
            font-size: 16px;
        }}
        #heatmap {{
            display: grid;
            grid-template-columns: repeat(9, 1fr);
            gap: 2px;
        }}
        .cell {{
            aspect-ratio: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 10px;
            border-radius: 3px;
        }}
        .header-cell {{
            background: transparent;
            color: #888;
            font-size: 9px;
        }}
        .stats {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
        }}
        .stat {{
            background: rgba(0, 100, 200, 0.2);
            padding: 10px;
            border-radius: 5px;
        }}
        .stat-label {{
            color: #888;
            font-size: 11px;
        }}
        .stat-value {{
            color: #00ff88;
            font-size: 18px;
            font-weight: bold;
        }}
        .connection {{
            display: flex;
            justify-content: space-between;
            padding: 5px 0;
            border-bottom: 1px solid #333;
        }}
        .connection-type {{
            font-size: 10px;
            padding: 2px 6px;
            border-radius: 3px;
        }}
        .attractive {{ background: rgba(255, 100, 100, 0.3); color: #ff6666; }}
        .repulsive {{ background: rgba(100, 100, 255, 0.3); color: #6666ff; }}
        #network {{
            width: 100%;
            height: 400px;
        }}
        .equation {{
            font-family: 'Times New Roman', serif;
            font-style: italic;
            color: #00ffff;
            background: rgba(0, 255, 255, 0.1);
            padding: 5px 10px;
            border-radius: 3px;
            display: inline-block;
            margin: 5px 0;
        }}
    </style>
</head>
<body>
    <h1>Quantum Coupling Matrix - 8 Limb Oscillators</h1>
    
    <div class="container">
        <div class="panel">
            <h2>Coupling Matrix gᵢⱼ</h2>
            <p class="equation">H = Σᵢ ℏωᵢ(aᵢ†aᵢ + ½) + Σᵢⱼ gᵢⱼ(aᵢ†aⱼ + aᵢaⱼ†)</p>
            <div id="heatmap"></div>
        </div>
        
        <div class="panel">
            <h2>Statistics</h2>
            <div class="stats">
                <div class="stat">
                    <div class="stat-label">Zero-Point Energy</div>
                    <div class="stat-value">{zpe:.2f}</div>
                </div>
                <div class="stat">
                    <div class="stat-label">Mean Coupling</div>
                    <div class="stat-value">{analysis['mean_coupling']:.4f}</div>
                </div>
                <div class="stat">
                    <div class="stat-label">Max Coupling</div>
                    <div class="stat-value">{analysis['max_coupling']:.4f}</div>
                </div>
                <div class="stat">
                    <div class="stat-label">Spectral Radius</div>
                    <div class="stat-value">{analysis['spectral_radius']:.4f}</div>
                </div>
            </div>
            
            <h2 style="margin-top: 20px;">Strongest Connections</h2>
            <div id="connections"></div>
        </div>
        
        <div class="panel" style="grid-column: span 2;">
            <h2>Network Visualization</h2>
            <canvas id="network"></canvas>
        </div>
    </div>
    
    <script>
        const LIMB_NAMES = {json.dumps(LIMB_NAMES)};
        const coupling = {json.dumps(coupling_list)};
        const omega = {json.dumps(omega_list)};
        const strongest = {json.dumps(analysis['strongest_connections'])};
        
        // Create heatmap
        function createHeatmap() {{
            const container = document.getElementById('heatmap');
            const maxVal = Math.max(...coupling.flat().map(Math.abs));
            
            // Empty corner
            container.innerHTML = '<div class="cell header-cell"></div>';
            
            // Column headers
            LIMB_NAMES.forEach(name => {{
                container.innerHTML += `<div class="cell header-cell">${{name.slice(0,4)}}</div>`;
            }});
            
            // Rows
            coupling.forEach((row, i) => {{
                container.innerHTML += `<div class="cell header-cell">${{LIMB_NAMES[i].slice(0,4)}}</div>`;
                row.forEach((val, j) => {{
                    const sym = (coupling[i][j] + coupling[j][i]) / 2;
                    const intensity = Math.abs(sym) / maxVal;
                    const r = sym > 0 ? 255 : Math.floor(100 + 155 * (1 - intensity));
                    const g = Math.floor(100 * (1 - intensity));
                    const b = sym < 0 ? 255 : Math.floor(100 + 155 * (1 - intensity));
                    const bg = i === j ? '#333' : `rgb(${{r}},${{g}},${{b}})`;
                    const text = i === j ? '-' : sym.toFixed(2);
                    container.innerHTML += `<div class="cell" style="background:${{bg}}">${{text}}</div>`;
                }});
            }});
        }}
        
        // Create connections list
        function createConnections() {{
            const container = document.getElementById('connections');
            strongest.forEach(conn => {{
                container.innerHTML += `
                    <div class="connection">
                        <span>${{conn.limb_i}} ↔ ${{conn.limb_j}}</span>
                        <span class="connection-type ${{conn.type}}">${{conn.coupling.toFixed(3)}}</span>
                    </div>
                `;
            }});
        }}
        
        // Create network graph
        function createNetwork() {{
            const canvas = document.getElementById('network');
            const ctx = canvas.getContext('2d');
            canvas.width = canvas.offsetWidth;
            canvas.height = canvas.offsetHeight;
            
            const cx = canvas.width / 2;
            const cy = canvas.height / 2;
            const radius = Math.min(cx, cy) * 0.7;
            
            // Calculate positions
            const positions = LIMB_NAMES.map((_, i) => {{
                const angle = (i / 8) * Math.PI * 2 - Math.PI / 2;
                return {{
                    x: cx + radius * Math.cos(angle),
                    y: cy + radius * Math.sin(angle)
                }};
            }});
            
            // Draw edges
            const maxVal = Math.max(...coupling.flat().map(Math.abs));
            for (let i = 0; i < 8; i++) {{
                for (let j = i + 1; j < 8; j++) {{
                    const val = (coupling[i][j] + coupling[j][i]) / 2;
                    if (Math.abs(val) > 0.02) {{
                        ctx.beginPath();
                        ctx.moveTo(positions[i].x, positions[i].y);
                        ctx.lineTo(positions[j].x, positions[j].y);
                        ctx.strokeStyle = val > 0 ? 
                            `rgba(255, 100, 100, ${{Math.abs(val) / maxVal}})` :
                            `rgba(100, 100, 255, ${{Math.abs(val) / maxVal}})`;
                        ctx.lineWidth = 1 + 4 * Math.abs(val) / maxVal;
                        ctx.stroke();
                    }}
                }}
            }}
            
            // Draw nodes
            positions.forEach((pos, i) => {{
                const hue = (i / 8) * 360;
                ctx.beginPath();
                ctx.arc(pos.x, pos.y, 25, 0, Math.PI * 2);
                ctx.fillStyle = `hsl(${{hue}}, 70%, 50%)`;
                ctx.fill();
                ctx.strokeStyle = '#fff';
                ctx.lineWidth = 2;
                ctx.stroke();
                
                // Label
                ctx.fillStyle = '#fff';
                ctx.font = '10px Courier New';
                ctx.textAlign = 'center';
                ctx.fillText(LIMB_NAMES[i].slice(0, 6), pos.x, pos.y + 40);
            }});
        }}
        
        createHeatmap();
        createConnections();
        createNetwork();
    </script>
</body>
</html>'''
    
    with open(output_path, 'w') as f:
        f.write(html)
    
    print(f"Saved HTML visualization to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize quantum coupling matrix')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/quantum_final.pt',
                       help='Path to checkpoint file')
    parser.add_argument('--output-dir', type=str, default='analysis/outputs',
                       help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load coupling matrix
    print(f"Loading checkpoint from {args.checkpoint}...")
    coupling, omega, zpe = load_coupling_matrix(args.checkpoint)
    
    if coupling is None:
        print("Could not find coupling matrix in checkpoint. Creating from fresh layer...")
        layer = QuantumCouplingLayer(hidden_dim=256, num_limbs=8)
        coupling = layer.get_coupling_matrix().detach().numpy()
        omega = layer.omega.detach().numpy()
        zpe = layer.zero_point.item()
    
    print(f"Coupling matrix shape: {coupling.shape}")
    print(f"Zero-point energy: {zpe}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # Heatmap
    create_heatmap(coupling, os.path.join(args.output_dir, 'coupling_heatmap.png'))
    
    # Network graph
    create_network_graph(coupling, os.path.join(args.output_dir, 'coupling_network.png'))
    
    # Frequency bar chart
    if omega is not None:
        create_frequency_bar(omega, os.path.join(args.output_dir, 'frequencies.png'))
    
    # JSON data
    output_json_data(coupling, omega, zpe, os.path.join(args.output_dir, 'coupling_data.json'))
    
    # HTML visualization
    generate_html_visualization(coupling, omega, zpe, 
                                os.path.join(args.output_dir, 'coupling_visualization.html'))
    
    # Print analysis
    print("\n" + "="*50)
    print("COUPLING ANALYSIS")
    print("="*50)
    analysis = analyze_coupling_structure(coupling)
    
    print(f"\nMean coupling: {analysis['mean_coupling']:.4f}")
    print(f"Max coupling: {analysis['max_coupling']:.4f}")
    print(f"Spectral radius: {analysis['spectral_radius']:.4f}")
    print(f"Positive ratio: {analysis['positive_ratio']:.1%}")
    
    print("\nStrongest connections:")
    for conn in analysis['strongest_connections']:
        print(f"  {conn['limb_i']} <-> {conn['limb_j']}: {conn['coupling']:.4f} ({conn['type']})")
    
    print("\nDone!")


if __name__ == '__main__':
    main()
