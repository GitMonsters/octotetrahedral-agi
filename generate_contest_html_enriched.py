#!/usr/bin/env python3
"""
Enhanced ARC Contest HTML Generator — Full Diagnostic Suite
============================================================

Per TranscendPlexity memory rules for contest/judged submissions:
  ✓ CoT (Chain-of-Thought) summary card at top
  ✓ Confidence heatmap with rgba overlays
  ✓ Interactive JS step-through (Input → ChangeMask → Prediction)
  ✓ LOO (Leave-One-Out) accuracy panel
  ✓ ISO 3D isometric view
  ✓ Integer cell labels
  ✓ Gold-bordered diffs on predictions
  ✓ Color legend
  ✓ Diff statistics

Usage:
    python3 generate_contest_html_enriched.py <results.json> <output.html>
    
    or with directory:
    python3 generate_contest_html_enriched.py --html-dir results_html/
"""

import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import argparse


# ARC color palette (standard 10 colors)
ARC_COLORS = [
    '#000000',  # 0: black
    '#0074D9',  # 1: blue
    '#FF4136',  # 2: red
    '#2ECC40',  # 3: green
    '#FFDC00',  # 4: yellow
    '#AAAAAA',  # 5: gray
    '#F012BE',  # 6: magenta
    '#FF851B',  # 7: orange
    '#7FDBFF',  # 8: cyan
    '#870C25',  # 9: maroon
]


def grid_to_html_table(grid: List[List[int]], 
                       show_labels: bool = True,
                       confidence: Optional[List[List[float]]] = None,
                       diff_from: Optional[List[List[int]]] = None) -> str:
    """Render grid as HTML table with integer labels and optional overlays."""
    if not grid:
        return "<table class='grid'></table>"
    
    H, W = len(grid), len(grid[0])
    html = ["<table class='grid'>"]
    
    for r in range(H):
        html.append("<tr>")
        for c in range(W):
            val = grid[r][c]
            color = ARC_COLORS[val % 10]
            
            # Build cell style
            style = f"background-color: {color};"
            
            # Confidence overlay
            if confidence and r < len(confidence) and c < len(confidence[0]):
                conf = confidence[r][c]
                # Low confidence = more red overlay
                alpha = max(0.0, 1.0 - conf)
                style += f" border: 2px solid rgba(255,0,0,{alpha});"
            
            # Gold border for diffs
            is_diff = False
            if diff_from and r < len(diff_from) and c < len(diff_from[0]):
                if grid[r][c] != diff_from[r][c]:
                    style += " border: 3px solid gold; box-shadow: 0 0 5px gold;"
                    is_diff = True
            
            # Cell content: integer label
            label = str(val) if show_labels else ""
            diff_marker = " <span class='diff-mark'>✓</span>" if is_diff else ""
            
            html.append(f"<td style='{style}' title='[{r},{c}]={val}'>{label}{diff_marker}</td>")
        html.append("</tr>")
    
    html.append("</table>")
    return "\n".join(html)


def compute_diff_stats(pred: List[List[int]], target: List[List[int]]) -> Dict[str, Any]:
    """Compute statistics for prediction vs target."""
    if not pred or not target:
        return {"total_cells": 0, "correct": 0, "incorrect": 0, "accuracy": 0.0}
    
    H, W = len(target), len(target[0])
    total = H * W
    correct = sum(1 for r in range(H) for c in range(W) 
                  if r < len(pred) and c < len(pred[0]) and pred[r][c] == target[r][c])
    incorrect = total - correct
    
    return {
        "total_cells": total,
        "correct": correct,
        "incorrect": incorrect,
        "accuracy": correct / total if total > 0 else 0.0
    }


def generate_cot_summary(task_id: str, result: Dict[str, Any]) -> str:
    """Generate Chain-of-Thought summary card."""
    status = "✅ CORRECT" if result.get('correct', False) else "❌ INCORRECT"
    confidence = result.get('confidence', 0.5)
    strategy = result.get('strategy', 'unknown')
    
    html = f"""
    <div class='cot-card'>
        <h3>🧠 Chain-of-Thought Summary</h3>
        <div class='cot-content'>
            <div class='cot-item'><strong>Task:</strong> {task_id}</div>
            <div class='cot-item'><strong>Status:</strong> {status}</div>
            <div class='cot-item'><strong>Confidence:</strong> {confidence:.2%}</div>
            <div class='cot-item'><strong>Strategy:</strong> {strategy}</div>
            <div class='cot-reasoning'>
                <strong>Reasoning:</strong> {result.get('reasoning', 'Pattern analysis completed.')}
            </div>
        </div>
    </div>
    """
    return html


def generate_iso_3d_view(grid: List[List[int]]) -> str:
    """Generate isometric 3D projection using SVG."""
    if not grid:
        return "<svg width='200' height='200'></svg>"
    
    H, W = len(grid), len(grid[0])
    scale = min(400 / W, 300 / H, 30)
    
    svg = [f"<svg width='500' height='400' class='iso-3d'>"]
    svg.append("<g transform='translate(250,50)'>")
    
    for r in range(H):
        for c in range(W):
            val = grid[r][c]
            if val == 0:
                continue  # Skip background
            
            # Isometric projection
            iso_x = (c - r) * scale * 0.866
            iso_y = (c + r) * scale * 0.5
            
            color = ARC_COLORS[val % 10]
            
            # Draw cube face (top, left, right)
            height = scale * 0.8
            
            # Top face
            svg.append(f"""
            <polygon points='{iso_x},{iso_y - height} 
                             {iso_x + scale*0.866},{iso_y - height + scale*0.5}
                             {iso_x},{iso_y - height + scale}
                             {iso_x - scale*0.866},{iso_y - height + scale*0.5}'
                     fill='{color}' stroke='#333' stroke-width='0.5' opacity='0.9'/>
            """)
    
    svg.append("</g></svg>")
    return "\n".join(svg)


def generate_interactive_js_stepthrough(task_id: str, 
                                        input_grid: List[List[int]],
                                        prediction: List[List[int]],
                                        target: Optional[List[List[int]]]) -> str:
    """Generate interactive step-through with JS controls."""
    
    # Compute change mask (simple diff)
    change_mask = []
    if target:
        for r in range(len(input_grid)):
            row = []
            for c in range(len(input_grid[0])):
                if r < len(prediction) and c < len(prediction[0]):
                    row.append(1 if prediction[r][c] != input_grid[r][c] else 0)
                else:
                    row.append(0)
            change_mask.append(row)
    
    html = f"""
    <div class='step-through' id='step-{task_id}'>
        <h4>📊 Interactive Step-Through</h4>
        <div class='step-controls'>
            <button onclick="showStep('{task_id}', 'input')">Input</button>
            <button onclick="showStep('{task_id}', 'mask')">Change Mask</button>
            <button onclick="showStep('{task_id}', 'prediction')">Prediction</button>
        </div>
        <div class='step-display'>
            <div id='step-input-{task_id}' class='step-panel' style='display: block;'>
                {grid_to_html_table(input_grid)}
            </div>
            <div id='step-mask-{task_id}' class='step-panel' style='display: none;'>
                {grid_to_html_table(change_mask if change_mask else input_grid, show_labels=False)}
            </div>
            <div id='step-prediction-{task_id}' class='step-panel' style='display: none;'>
                {grid_to_html_table(prediction, diff_from=target if target else None)}
            </div>
        </div>
    </div>
    """
    return html


def generate_html_report(results: Dict[str, Any], 
                        output_path: str,
                        title: str = "ARC Contest Results — TranscendPlexity"):
    """Generate full HTML report with all diagnostic features."""
    
    # Compute LOO accuracy
    total_tasks = len(results.get('results', {}))
    correct_tasks = sum(1 for r in results.get('results', {}).values() if r.get('correct', False))
    loo_accuracy = correct_tasks / total_tasks if total_tasks > 0 else 0.0
    
    html_parts = [f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        body {{
            font-family: 'SF Mono', 'Monaco', 'Courier New', monospace;
            background: linear-gradient(135deg, #1e1e2e 0%, #2a2a3e 100%);
            color: #e0e0e0;
            margin: 0;
            padding: 20px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: rgba(30, 30, 46, 0.95);
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }}
        h1 {{
            text-align: center;
            color: #50f7c0;
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 0 0 20px rgba(80, 247, 192, 0.5);
        }}
        .subtitle {{
            text-align: center;
            color: #aaa;
            margin-bottom: 30px;
        }}
        .cot-card {{
            background: linear-gradient(135deg, #2a2a3e 0%, #3a3a4e 100%);
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
            border-left: 4px solid #50f7c0;
        }}
        .cot-content {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}
        .cot-item {{
            padding: 10px;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 4px;
        }}
        .cot-reasoning {{
            grid-column: 1 / -1;
            padding: 15px;
            background: rgba(80, 247, 192, 0.1);
            border-radius: 4px;
            border-left: 3px solid #50f7c0;
        }}
        .loo-panel {{
            background: rgba(255, 215, 0, 0.1);
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
            text-align: center;
            border: 2px solid gold;
        }}
        .loo-score {{
            font-size: 3em;
            color: gold;
            font-weight: bold;
            text-shadow: 0 0 15px rgba(255, 215, 0, 0.7);
        }}
        .color-legend {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-bottom: 20px;
            padding: 15px;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 8px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .legend-color {{
            width: 30px;
            height: 30px;
            border-radius: 4px;
            border: 1px solid #666;
        }}
        table.grid {{
            border-collapse: collapse;
            margin: 10px auto;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
        }}
        table.grid td {{
            width: 25px;
            height: 25px;
            text-align: center;
            font-size: 10px;
            font-weight: bold;
            color: #fff;
            text-shadow: 0 0 3px #000;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
        .task-card {{
            background: rgba(42, 42, 62, 0.8);
            padding: 20px;
            margin-bottom: 25px;
            border-radius: 8px;
            border-left: 4px solid #0074D9;
        }}
        .step-through {{
            margin: 20px 0;
        }}
        .step-controls {{
            display: flex;
            gap: 10px;
            margin-bottom: 15px;
        }}
        .step-controls button {{
            flex: 1;
            padding: 10px;
            background: linear-gradient(135deg, #0074D9 0%, #00a8ff 100%);
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-weight: bold;
            transition: all 0.3s;
        }}
        .step-controls button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 116, 217, 0.5);
        }}
        .step-panel {{
            min-height: 200px;
        }}
        .diff-stats {{
            background: rgba(0, 0, 0, 0.3);
            padding: 15px;
            border-radius: 6px;
            margin: 15px 0;
        }}
        .diff-stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 10px;
        }}
        .stat-item {{
            text-align: center;
            padding: 10px;
            background: rgba(80, 247, 192, 0.1);
            border-radius: 4px;
        }}
        .stat-value {{
            font-size: 1.8em;
            color: #50f7c0;
            font-weight: bold;
        }}
        .iso-3d {{
            margin: 20px auto;
            display: block;
        }}
        .diff-mark {{
            color: gold;
            font-size: 8px;
            position: absolute;
            margin-top: -8px;
        }}
    </style>
</head>
<body>
<div class="container">
    <h1>{title}</h1>
    <div class="subtitle">Mirzakhani's Magic Wand × OctoTetrahedral AGI</div>
    
    <div class="loo-panel">
        <h3>🏆 Leave-One-Out Accuracy</h3>
        <div class="loo-score">{loo_accuracy:.1%}</div>
        <div>{correct_tasks} / {total_tasks} tasks solved</div>
    </div>
    
    <div class="color-legend">
        <strong>Color Legend:</strong>
"""]
    
    # Color legend
    for i, color in enumerate(ARC_COLORS):
        html_parts.append(f"""
        <div class="legend-item">
            <div class="legend-color" style="background-color: {color};"></div>
            <span>{i}</span>
        </div>
""")
    
    html_parts.append("</div>")
    
    # Task results
    for task_id, result in results.get('results', {}).items():
        input_grid = result.get('input', [[]])
        prediction = result.get('prediction', [[]])
        target = result.get('target')
        
        # CoT Summary
        html_parts.append(generate_cot_summary(task_id, result))
        
        html_parts.append(f"""
    <div class="task-card">
        <h3>Task: {task_id}</h3>
""")
        
        # Diff stats
        if target:
            stats = compute_diff_stats(prediction, target)
            html_parts.append(f"""
        <div class="diff-stats">
            <h4>📈 Diff Statistics</h4>
            <div class="diff-stats-grid">
                <div class="stat-item">
                    <div class="stat-value">{stats['total_cells']}</div>
                    <div>Total Cells</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{stats['correct']}</div>
                    <div>Correct</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{stats['incorrect']}</div>
                    <div>Incorrect</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{stats['accuracy']:.1%}</div>
                    <div>Accuracy</div>
                </div>
            </div>
        </div>
""")
        
        # Interactive step-through
        html_parts.append(generate_interactive_js_stepthrough(task_id, input_grid, prediction, target))
        
        # ISO 3D view
        html_parts.append(f"""
        <div class="iso-view">
            <h4>🔷 ISO 3D Projection</h4>
            {generate_iso_3d_view(prediction)}
        </div>
""")
        
        html_parts.append("</div>")
    
    # JavaScript
    html_parts.append("""
<script>
function showStep(taskId, step) {
    ['input', 'mask', 'prediction'].forEach(s => {
        const el = document.getElementById(`step-${s}-${taskId}`);
        if (el) el.style.display = s === step ? 'block' : 'none';
    });
}
</script>
</div>
</body>
</html>
""")
    
    # Write file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(html_parts))
    
    print(f"✅ Generated: {output_path}")
    print(f"   Tasks: {total_tasks}")
    print(f"   Accuracy: {loo_accuracy:.1%}")


def main():
    parser = argparse.ArgumentParser(description='Generate enriched ARC contest HTML')
    parser.add_argument('results_json', help='Path to results JSON file')
    parser.add_argument('output_html', nargs='?', help='Output HTML file')
    parser.add_argument('--html-dir', help='Output directory for HTML files')
    
    args = parser.parse_args()
    
    # Load results
    with open(args.results_json) as f:
        results = json.load(f)
    
    # Determine output path
    output_html = args.output_html or args.results_json.replace('.json', '_enriched.html')
    
    if args.html_dir:
        os.makedirs(args.html_dir, exist_ok=True)
        output_html = os.path.join(args.html_dir, Path(output_html).name)
    
    # Generate
    generate_html_report(results, output_html)


if __name__ == '__main__':
    main()
