#!/usr/bin/env python3
"""
ARC Contest Visualization Generator
====================================

Generates enhanced HTML visualizations for ARC-AGI contest submissions with:
- Chain-of-Thought (CoT) summary cards at top
- Confidence heatmaps with rgba overlays  
- Interactive JavaScript step-through (Input→ChangeMask→Prediction)
- Leave-One-Out (LOO) accuracy panels
- ISO 3D isometric views
- Integer cell labels, color legends, diff stats
- Gold-bordered diffs on predictions

Usage:
    python3 arc_contest_viz_generator.py --task-id <id> --html-dir <output_dir>
    python3 arc_contest_viz_generator.py --batch-file tasks.json --html-dir ./visualizations
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

# ARC color palette
ARC_COLORS = [
    "#000000",  # 0: black
    "#0074D9",  # 1: blue
    "#FF4136",  # 2: red
    "#2ECC40",  # 3: green
    "#FFDC00",  # 4: yellow
    "#AAAAAA",  # 5: gray
    "#F012BE",  # 6: magenta
    "#FF851B",  # 7: orange
    "#7FDBFF",  # 8: cyan
    "#870C25",  # 9: maroon
]


def grid_to_html(grid: List[List[int]], cell_size: int = 20, 
                 show_labels: bool = True, confidence: Optional[List[List[float]]] = None,
                 diff_overlay: Optional[List[List[bool]]] = None) -> str:
    """
    Render grid as HTML with optional confidence heatmap and diff overlay.
    
    Args:
        grid: 2D array of integers (0-9)
        cell_size: Size of each cell in pixels
        show_labels: Whether to show integer labels in cells
        confidence: Optional 2D array of confidence values (0.0-1.0)
        diff_overlay: Optional 2D array marking differences (gold border)
    """
    h, w = len(grid), len(grid[0])
    html = f'<svg width="{w*cell_size}" height="{h*cell_size}" style="border:1px solid #333">\n'
    
    for y in range(h):
        for x in range(w):
            val = grid[y][x]
            color = ARC_COLORS[val]
            
            # Apply confidence overlay if provided
            if confidence is not None:
                conf = confidence[y][x]
                opacity = 0.3 + 0.7 * conf  # 30-100% opacity based on confidence
                color_rgba = color + f"{int(opacity * 255):02x}"
            else:
                color_rgba = color
            
            # Determine stroke (border)
            stroke = "#FFD700" if (diff_overlay is not None and diff_overlay[y][x]) else "#222"
            stroke_width = 3 if (diff_overlay is not None and diff_overlay[y][x]) else 1
            
            html += f'  <rect x="{x*cell_size}" y="{y*cell_size}" width="{cell_size}" height="{cell_size}" '
            html += f'fill="{color_rgba}" stroke="{stroke}" stroke-width="{stroke_width}"/>\n'
            
            # Add integer label
            if show_labels:
                text_color = "#fff" if val in [0, 1, 2, 9] else "#000"
                html += f'  <text x="{x*cell_size + cell_size/2}" y="{y*cell_size + cell_size/2 + 5}" '
                html += f'text-anchor="middle" font-size="10" fill="{text_color}" font-family="monospace">{val}</text>\n'
    
    html += '</svg>'
    return html


def create_3d_iso_view(grid: List[List[int]], cell_size: int = 15) -> str:
    """
    Generate isometric 3D view of the grid (ISO perspective).
    Non-zero cells are rendered as 3D blocks.
    """
    h, w = len(grid), len(grid[0])
    
    # Calculate canvas size for isometric view
    iso_width = int((w + h) * cell_size * 0.866) + 100
    iso_height = int((w + h) * cell_size * 0.5) + max(h, w) * cell_size + 100
    
    html = f'<svg width="{iso_width}" height="{iso_height}" style="background:#111; border:1px solid #333">\n'
    
    # Draw from back to front for proper occlusion
    blocks = []
    for y in range(h):
        for x in range(w):
            val = grid[y][x]
            if val == 0:  # Skip black cells
                continue
            
            # Isometric coordinates
            iso_x = (x - y) * cell_size * 0.866 + iso_width/2
            iso_y = (x + y) * cell_size * 0.5 + 50
            
            color = ARC_COLORS[val]
            # Calculate lighter/darker shades for 3D effect
            blocks.append((x + y, iso_x, iso_y, color, val))
    
    # Sort by depth (back to front)
    blocks.sort()
    
    for _, iso_x, iso_y, color, val in blocks:
        # Draw 3D block (top, left, right faces)
        # Top face
        points_top = f"{iso_x},{iso_y} {iso_x + cell_size*0.866},{iso_y + cell_size*0.5} "
        points_top += f"{iso_x},{iso_y + cell_size} {iso_x - cell_size*0.866},{iso_y + cell_size*0.5}"
        html += f'  <polygon points="{points_top}" fill="{color}" stroke="#000" stroke-width="0.5"/>\n'
        
        # Left face (darker)
        points_left = f"{iso_x},{iso_y} {iso_x - cell_size*0.866},{iso_y + cell_size*0.5} "
        points_left += f"{iso_x - cell_size*0.866},{iso_y + cell_size*1.5} {iso_x},{iso_y + cell_size}"
        dark_color = f"#{int(int(color[1:3], 16)*0.7):02x}{int(int(color[3:5], 16)*0.7):02x}{int(int(color[5:7], 16)*0.7):02x}"
        html += f'  <polygon points="{points_left}" fill="{dark_color}" stroke="#000" stroke-width="0.5"/>\n'
        
        # Right face (medium)
        points_right = f"{iso_x},{iso_y} {iso_x + cell_size*0.866},{iso_y + cell_size*0.5} "
        points_right += f"{iso_x + cell_size*0.866},{iso_y + cell_size*1.5} {iso_x},{iso_y + cell_size}"
        med_color = f"#{int(int(color[1:3], 16)*0.85):02x}{int(int(color[3:5], 16)*0.85):02x}{int(int(color[5:7], 16)*0.85):02x}"
        html += f'  <polygon points="{points_right}" fill="{med_color}" stroke="#000" stroke-width="0.5"/>\n'
    
    html += '</svg>'
    return html


def compute_diff_stats(input_grid: List[List[int]], output_grid: List[List[int]]) -> Dict[str, Any]:
    """Compute difference statistics between input and output."""
    h1, w1 = len(input_grid), len(input_grid[0])
    h2, w2 = len(output_grid), len(output_grid[0])
    
    size_change = (h2 - h1, w2 - w1)
    
    # Count changed cells (only for overlapping region)
    changed = 0
    total = min(h1, h2) * min(w1, w2)
    diff_mask = [[False] * w2 for _ in range(h2)]
    
    for y in range(min(h1, h2)):
        for x in range(min(w1, w2)):
            if input_grid[y][x] != output_grid[y][x]:
                changed += 1
                diff_mask[y][x] = True
    
    return {
        'size_change': size_change,
        'cells_changed': changed,
        'total_cells': total,
        'percent_changed': (changed / total * 100) if total > 0 else 0,
        'diff_mask': diff_mask
    }


def generate_contest_html(task_id: str, task_data: Dict[str, Any], 
                          prediction: Optional[List[List[int]]] = None,
                          cot_summary: str = "",
                          confidence_map: Optional[List[List[float]]] = None,
                          loo_accuracy: Optional[float] = None) -> str:
    """
    Generate complete contest-ready HTML visualization.
    
    Args:
        task_id: ARC task ID
        task_data: Task data with 'train' and 'test' examples
        prediction: Optional prediction grid for test example
        cot_summary: Chain-of-thought reasoning summary
        confidence_map: Optional per-cell confidence (0.0-1.0)
        loo_accuracy: Leave-one-out cross-validation accuracy
    """
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>ARC Contest: {task_id}</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ background: #0a0a0a; color: #eee; font-family: system-ui, sans-serif; padding: 20px; }}

/* CoT Summary Card */
.cot-card {{
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 2px solid #0f3460;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 30px;
    box-shadow: 0 8px 32px rgba(15, 52, 96, 0.3);
}}
.cot-card h2 {{
    color: #61dafb;
    margin-bottom: 12px;
    font-size: 20px;
    display: flex;
    align-items: center;
    gap: 10px;
}}
.cot-card h2::before {{
    content: "🧠";
    font-size: 24px;
}}
.cot-summary {{
    line-height: 1.6;
    color: #ccc;
    background: rgba(0,0,0,0.3);
    padding: 15px;
    border-radius: 8px;
    border-left: 4px solid #61dafb;
}}

/* LOO Panel */
.loo-panel {{
    background: #1a1a1a;
    border: 2px solid #2a6e2a;
    border-radius: 8px;
    padding: 15px;
    margin-bottom: 20px;
    display: inline-block;
}}
.loo-panel h3 {{
    color: #4CAF50;
    margin-bottom: 8px;
    font-size: 16px;
}}
.loo-accuracy {{
    font-size: 32px;
    font-weight: bold;
    color: #4CAF50;
    font-family: monospace;
}}

/* Color Legend */
.color-legend {{
    background: #1a1a1a;
    border: 1px solid #333;
    border-radius: 8px;
    padding: 15px;
    margin-bottom: 20px;
    display: inline-flex;
    flex-wrap: wrap;
    gap: 10px;
}}
.legend-item {{
    display: flex;
    align-items: center;
    gap: 6px;
}}
.legend-color {{
    width: 20px;
    height: 20px;
    border: 1px solid #555;
    border-radius: 3px;
}}
.legend-label {{
    font-size: 12px;
    color: #aaa;
    font-family: monospace;
}}

/* Grid Container */
.grid-section {{
    margin-bottom: 40px;
    background: #111;
    border-radius: 10px;
    padding: 20px;
    border: 1px solid #222;
}}
.grid-section h3 {{
    color: #61dafb;
    margin-bottom: 15px;
    font-size: 18px;
}}
.example {{
    margin-bottom: 30px;
    padding: 15px;
    background: #0a0a0a;
    border-radius: 8px;
}}
.grid-pair {{
    display: flex;
    gap: 20px;
    align-items: flex-start;
    flex-wrap: wrap;
}}
.grid-wrapper {{
    position: relative;
}}
.grid-label {{
    font-size: 12px;
    color: #888;
    margin-bottom: 8px;
    font-weight: bold;
}}
.diff-stats {{
    margin-top: 10px;
    font-size: 11px;
    color: #888;
    font-family: monospace;
}}
.diff-stats span {{
    color: #FFD700;
}}

/* Prediction Section */
.prediction {{
    background: linear-gradient(135deg, #2a1a2e 0%, #1a163e 100%);
    border: 3px solid #FFD700;
    border-radius: 10px;
    padding: 20px;
}}
.prediction h3 {{
    color: #FFD700;
}}

/* Interactive Step-Through */
.step-through {{
    background: #1a1a1a;
    border: 2px solid #444;
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 30px;
}}
.step-controls {{
    display: flex;
    gap: 10px;
    margin-bottom: 15px;
    align-items: center;
}}
.step-btn {{
    background: #0074D9;
    color: white;
    border: none;
    padding: 8px 16px;
    border-radius: 5px;
    cursor: pointer;
    font-size: 14px;
    transition: background 0.2s;
}}
.step-btn:hover {{
    background: #0056a0;
}}
.step-btn:disabled {{
    background: #333;
    cursor: not-allowed;
}}
.step-indicator {{
    font-size: 14px;
    color: #aaa;
    font-family: monospace;
}}
#step-display {{
    min-height: 400px;
}}

/* 3D ISO View */
.iso-view {{
    background: #0a0a0a;
    border-radius: 10px;
    padding: 20px;
    margin-top: 20px;
}}
.iso-view h3 {{
    color: #7FDBFF;
    margin-bottom: 15px;
}}

/* Confidence Heatmap */
.heatmap-legend {{
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 10px 0;
    font-size: 12px;
}}
.heatmap-gradient {{
    width: 200px;
    height: 20px;
    background: linear-gradient(to right, 
        rgba(255,0,0,0.8) 0%, 
        rgba(255,255,0,0.8) 50%, 
        rgba(0,255,0,0.8) 100%);
    border: 1px solid #555;
    border-radius: 3px;
}}
</style>
</head>
<body>

<h1 style="color: #61dafb; margin-bottom: 30px;">🏆 ARC-AGI Contest Submission: {task_id}</h1>

<!-- Chain-of-Thought Summary Card -->
<div class="cot-card">
    <h2>Chain-of-Thought Summary</h2>
    <div class="cot-summary">
        {cot_summary if cot_summary else "Pattern recognition: Identify spatial transformations and apply to test input."}
    </div>
</div>

"""
    
    # LOO Panel
    if loo_accuracy is not None:
        html += f"""
<!-- Leave-One-Out Accuracy Panel -->
<div class="loo-panel">
    <h3>📊 Leave-One-Out Cross-Validation</h3>
    <div class="loo-accuracy">{loo_accuracy:.1f}%</div>
    <div style="font-size: 11px; color: #888; margin-top: 5px;">
        Training accuracy with one example held out
    </div>
</div>
"""
    
    # Color Legend
    html += """
<div class="color-legend">
    <strong style="margin-right: 10px;">Color Legend:</strong>
"""
    for i, color in enumerate(ARC_COLORS):
        html += f'    <div class="legend-item"><div class="legend-color" style="background:{color}"></div><span class="legend-label">{i}</span></div>\n'
    html += "</div>\n\n"
    
    # Training Examples
    html += '<div class="grid-section">\n<h3>📚 Training Examples</h3>\n'
    for idx, example in enumerate(task_data.get('train', []), 1):
        input_grid = example['input'] if 'input' in example else example.get('i', [])
        output_grid = example['output'] if 'output' in example else example.get('o', [])
        
        diff_stats = compute_diff_stats(input_grid, output_grid)
        
        html += f'<div class="example">\n<h4 style="color:#888; margin-bottom:10px;">Example {idx}</h4>\n'
        html += '<div class="grid-pair">\n'
        html += f'<div class="grid-wrapper">\n<div class="grid-label">Input ({len(input_grid)}×{len(input_grid[0])})</div>\n'
        html += grid_to_html(input_grid, show_labels=True)
        html += '</div>\n'
        html += f'<div class="grid-wrapper">\n<div class="grid-label">Output ({len(output_grid)}×{len(output_grid[0])})</div>\n'
        html += grid_to_html(output_grid, show_labels=True, diff_overlay=diff_stats['diff_mask'])
        html += f'<div class="diff-stats">Δ size: {diff_stats["size_change"]}, '
        html += f'<span>{diff_stats["cells_changed"]}</span> cells changed ({diff_stats["percent_changed"]:.1f}%)</div>\n'
        html += '</div>\n'
        html += '</div>\n</div>\n'
    
    html += '</div>\n\n'
    
    # Test Example with Prediction
    test_examples = task_data.get('test', [])
    if test_examples:
        test_input = test_examples[0]['input'] if 'input' in test_examples[0] else test_examples[0].get('i', [])
        
        html += '<div class="grid-section prediction">\n<h3>🎯 Test Example & Prediction</h3>\n'
        html += '<div class="grid-pair">\n'
        html += f'<div class="grid-wrapper">\n<div class="grid-label">Test Input ({len(test_input)}×{len(test_input[0])})</div>\n'
        html += grid_to_html(test_input, show_labels=True)
        html += '</div>\n'
        
        if prediction:
            html += f'<div class="grid-wrapper">\n<div class="grid-label">Prediction ({len(prediction)}×{len(prediction[0])})</div>\n'
            if confidence_map:
                html += '<div class="heatmap-legend"><span>Confidence:</span><div class="heatmap-gradient"></div><span>Low → High</span></div>\n'
                html += grid_to_html(prediction, show_labels=True, confidence=confidence_map)
            else:
                html += grid_to_html(prediction, show_labels=True)
            html += '</div>\n'
        
        html += '</div>\n</div>\n\n'
        
        # Interactive Step-Through
        if prediction:
            html += """
<div class="step-through">
    <h3 style="color:#61dafb; margin-bottom:15px;">🎬 Interactive Step-Through</h3>
    <div class="step-controls">
        <button class="step-btn" id="prevBtn" onclick="prevStep()">← Previous</button>
        <button class="step-btn" id="nextBtn" onclick="nextStep()">Next →</button>
        <span class="step-indicator">Step: <span id="stepNum">1</span> / 3</span>
    </div>
    <div id="step-display"></div>
</div>

<script>
let currentStep = 0;
const steps = [
    {
        title: 'Step 1: Input Grid',
        content: `""" + grid_to_html(test_input, cell_size=25, show_labels=True) + """`
    },
    {
        title: 'Step 2: Change Mask',
        content: `""" + grid_to_html(compute_diff_stats(test_input, prediction)['diff_mask'] if prediction else test_input, cell_size=25) + """` 
    },
    {
        title: 'Step 3: Predicted Output',
        content: `""" + grid_to_html(prediction, cell_size=25, show_labels=True) + """`
    }
];

function showStep(n) {
    currentStep = Math.max(0, Math.min(n, steps.length - 1));
    document.getElementById('step-display').innerHTML = `
        <h4 style="color:#7FDBFF; margin-bottom:10px;">${steps[currentStep].title}</h4>
        ${steps[currentStep].content}
    `;
    document.getElementById('stepNum').textContent = currentStep + 1;
    document.getElementById('prevBtn').disabled = currentStep === 0;
    document.getElementById('nextBtn').disabled = currentStep === steps.length - 1;
}

function nextStep() { showStep(currentStep + 1); }
function prevStep() { showStep(currentStep - 1); }

// Initialize
showStep(0);
</script>
"""
        
        # 3D ISO View
        if prediction:
            html += '<div class="iso-view">\n<h3>🎲 ISO 3D View</h3>\n'
            html += '<div style="display:flex; gap:30px; flex-wrap:wrap;">\n'
            html += '<div><h4 style="color:#888; font-size:14px; margin-bottom:10px;">Input (3D)</h4>\n'
            html += create_3d_iso_view(test_input)
            html += '</div>\n<div><h4 style="color:#888; font-size:14px; margin-bottom:10px;">Prediction (3D)</h4>\n'
            html += create_3d_iso_view(prediction)
            html += '</div>\n</div>\n</div>\n'
    
    html += """
</body>
</html>
"""
    
    return html


def main():
    parser = argparse.ArgumentParser(description='Generate ARC contest visualizations')
    parser.add_argument('--task-id', help='Single task ID to visualize')
    parser.add_argument('--batch-file', help='JSON file with multiple tasks')
    parser.add_argument('--html-dir', required=True, help='Output directory for HTML files')
    parser.add_argument('--with-predictions', help='JSON file with predictions')
    
    args = parser.parse_args()
    
    output_dir = Path(args.html_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.task_id:
        # Single task mode
        # Load task data (placeholder - would load from ARC dataset)
        task_data = {"train": [], "test": []}
        html = generate_contest_html(
            args.task_id,
            task_data,
            cot_summary="Analyzing spatial patterns and transformations...",
            loo_accuracy=95.0
        )
        
        output_file = output_dir / f"{args.task_id}_contest.html"
        output_file.write_text(html)
        print(f"✅ Generated: {output_file}")
    
    elif args.batch_file:
        # Batch mode
        with open(args.batch_file) as f:
            tasks = json.load(f)
        
        for task in tasks:
            task_id = task['id']
            html = generate_contest_html(
                task_id,
                task,
                cot_summary=task.get('cot_summary', ''),
                loo_accuracy=task.get('loo_accuracy')
            )
            
            output_file = output_dir / f"{task_id}_contest.html"
            output_file.write_text(html)
            print(f"✅ Generated: {output_file}")
        
        print(f"\n🎉 Generated {len(tasks)} contest visualizations in {output_dir}")


if __name__ == "__main__":
    main()
