"""
ARC HTML 3D Reasoning Visualizer — TranscendPlexity Edition
===========================================================
Generates rich dark-theme HTML pages with:
  • Standard 2D ARC grid visualisation (input → output, train + test)
  • Hybrid Prismatic / Tetrahedral 3D grid rendering via inline SVG
    – each cell is drawn as an isometric prism whose top face shows the
      ARC colour and whose depth face shows the tetrahedral sub-cell pattern
    – colour intensity encodes value magnitude; edge glow highlights diffs
  • Canvas-based WebGL-lite animation (pure JS, no external deps)
    – rotating tetrahedral cage inscribed around the grid volume
    – prismatic extrusion of each non-zero cell in 3D
  • Per-task reasoning panel: rule text, colour-frequency histogram,
    pattern entropy score, pass/fail badge

Usage
-----
    from arc_html_3d import generate_html_3d
    html = generate_html_3d(task_id, task_data, rule, predicted_output)
    open(f"solves/{task_id}/reasoning.html", "w").write(html)
"""

from __future__ import annotations
import math, json
from typing import List, Optional, Dict, Tuple

# ─── ARC colour palette ─────────────────────────────────────────────────────
ARC_HEX = {
    0: "#111111", 1: "#0074D9", 2: "#FF4136", 3: "#2ECC40",
    4: "#FFDC00", 5: "#AAAAAA", 6: "#F012BE", 7: "#FF851B",
    8: "#7FDBFF", 9: "#870C25",
}
ARC_NAMES = {
    0:"black",1:"blue",2:"red",3:"green",4:"yellow",
    5:"grey",6:"magenta",7:"orange",8:"cyan",9:"maroon",
}

def _hex_to_rgb(h: str) -> Tuple[int,int,int]:
    h = h.lstrip('#')
    return int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)

def _darken(h: str, f: float = 0.55) -> str:
    r,g,b = _hex_to_rgb(h)
    return f"#{int(r*f):02x}{int(g*f):02x}{int(b*f):02x}"

def _lighten(h: str, f: float = 1.4) -> str:
    r,g,b = _hex_to_rgb(h)
    return f"#{min(255,int(r*f)):02x}{min(255,int(g*f)):02x}{min(255,int(b*f)):02x}"

# ─── Isometric prism SVG cell ────────────────────────────────────────────────
ISO_W  = 20   # isometric cell width (projected)
ISO_H  = 10   # isometric cell height (projected top face)
ISO_D  = 8    # depth of prism side face

def _iso_cell_svg(col: int, row: int, value: int,
                  highlight: bool = False, size: int = 20) -> str:
    """Return an <g> SVG group for one isometric prism cell at grid (col,row)."""
    s = size
    half_s = s // 2
    depth  = max(4, s * 2 // 5)
    top_col = ARC_HEX.get(value, "#333")
    right_col = _darken(top_col, 0.55)
    left_col  = _darken(top_col, 0.70)

    # Isometric projection: (col,row) → screen (x,y)
    x = col * s
    y = row * half_s + col * half_s // 2   # slight diagonal stagger

    # Top face (parallelogram)
    # Points: TL, TR, BR, BL (isometric top)
    pts_top = [
        (x,         y + half_s),
        (x + s,     y),
        (x + s,     y + half_s),
        (x,         y + s),
    ]
    # Right face
    pts_right = [
        (x + s,     y),
        (x + s + half_s // 2, y + depth),
        (x + s + half_s // 2, y + depth + half_s),
        (x + s,     y + half_s),
    ]
    # Left face  (floor of cell)
    pts_left = [
        (x,         y + s),
        (x + s,     y + half_s),
        (x + s + half_s//2, y + depth + half_s),
        (x + half_s//2,     y + depth + s),
    ]

    def pts(pairs): return " ".join(f"{px},{py}" for px,py in pairs)

    glow = f'filter="url(#glow)"' if highlight else ''
    # Tetrahedral pattern on top face: draw 4 triangles
    cx = x + s//2; cy = y + s//2
    tet = ""
    if value != 0:
        tc = _lighten(top_col, 1.25)
        tet = (
            f'<line x1="{x}" y1="{y+half_s}" x2="{cx}" y2="{cy}" '
            f'stroke="{tc}" stroke-width="0.5" opacity="0.5"/>'
            f'<line x1="{x+s}" y1="{y}" x2="{cx}" y2="{cy}" '
            f'stroke="{tc}" stroke-width="0.5" opacity="0.5"/>'
            f'<line x1="{x+s}" y1="{y+half_s}" x2="{cx}" y2="{cy}" '
            f'stroke="{tc}" stroke-width="0.5" opacity="0.5"/>'
            f'<line x1="{x}" y1="{y+s}" x2="{cx}" y2="{cy}" '
            f'stroke="{tc}" stroke-width="0.5" opacity="0.5"/>'
        )

    return (
        f'<g {glow}>'
        f'<polygon points="{pts(pts_left)}" fill="{left_col}" stroke="#222" stroke-width="0.5"/>'
        f'<polygon points="{pts(pts_right)}" fill="{right_col}" stroke="#222" stroke-width="0.5"/>'
        f'<polygon points="{pts(pts_top)}" fill="{top_col}" stroke="#333" stroke-width="0.5"/>'
        f'{tet}'
        f'</g>'
    )

def _grid_to_iso_svg(grid: List[List[int]],
                     diff: Optional[List[List[int]]] = None,
                     cell_size: int = 20) -> str:
    """Render a full grid as isometric SVG."""
    rows = len(grid); cols = len(grid[0])
    s = cell_size; half_s = s // 2
    depth = max(4, s * 2 // 5)
    # Compute canvas size
    W = cols * s + rows * half_s + half_s // 2 + 30
    H = rows * (half_s + s//4) + cols * half_s//2 + depth + s + 20

    defs = """<defs>
      <filter id="glow" x="-30%" y="-30%" width="160%" height="160%">
        <feGaussianBlur stdDeviation="3" result="blur"/>
        <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
      </filter>
    </defs>"""

    cells = []
    # Draw back-to-front for correct z-ordering
    for r in range(rows - 1, -1, -1):
        for c in range(cols):
            v = grid[r][c]
            hi = diff is not None and diff[r][c] != v
            cells.append(_iso_cell_svg(c, r, v, highlight=hi, size=s))

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'style="display:block;overflow:visible">'
        f'{defs}{"".join(cells)}</svg>'
    )

# ─── Flat 2-D HTML table cell (compact, for side-by-side overview) ───────────
def _flat_cell(v: int, hi: bool = False) -> str:
    bg = ARC_HEX.get(v, "#333")
    border = "2px solid #FFD700" if hi else "1px solid #333"
    return (f'<td style="width:18px;height:18px;background:{bg};'
            f'border:{border};padding:0"></td>')

def _flat_grid(grid: List[List[int]],
               diff: Optional[List[List[int]]] = None) -> str:
    rows = []
    for r, row in enumerate(grid):
        cells = "".join(
            _flat_cell(v, diff is not None and diff[r][c] != v)
            for c, v in enumerate(row)
        )
        rows.append(f"<tr>{cells}</tr>")
    return '<table style="border-collapse:collapse;margin:0">' + "".join(rows) + "</table>"

# ─── Colour histogram ────────────────────────────────────────────────────────
def _colour_hist(grid: List[List[int]]) -> str:
    from collections import Counter
    flat = [v for row in grid for v in row]
    total = len(flat) or 1
    counts = Counter(flat)
    bars = []
    for v in sorted(counts):
        pct = counts[v] / total * 100
        col = ARC_HEX.get(v, "#333")
        bars.append(
            f'<div title="{ARC_NAMES.get(v,v)}: {counts[v]}" style="display:inline-block;'
            f'width:20px;background:{col};height:{max(4,int(pct*1.5))}px;'
            f'margin:1px;vertical-align:bottom;border:1px solid #555"></div>'
        )
    return '<div style="height:50px;display:flex;align-items:flex-end">' + "".join(bars) + "</div>"

# ─── Entropy score ───────────────────────────────────────────────────────────
def _entropy(grid: List[List[int]]) -> float:
    from collections import Counter
    flat = [v for row in grid for v in row]
    total = len(flat) or 1
    return -sum((c/total)*math.log2(c/total) for c in Counter(flat).values() if c > 0)

# ─── WebGL-lite 3D canvas animation ─────────────────────────────────────────
def _webgl_canvas(task_id: str, grid: List[List[int]]) -> str:
    """Pure-JS Canvas 2D isometric spinning tetrahedral cage + extruded cells."""
    rows = len(grid); cols = len(grid[0]) if rows else 0
    # Build cell data as JS array
    cells_js = json.dumps([
        {"r": r, "c": c, "v": grid[r][c]}
        for r in range(rows) for c in range(cols) if grid[r][c] != 0
    ])
    colors_js = json.dumps(ARC_HEX)
    cid = f"canvas3d_{task_id.replace('-','_')}"

    return f"""
<canvas id="{cid}" width="320" height="240"
  style="background:#0a0a14;border:1px solid #333;border-radius:4px;display:block"></canvas>
<script>
(function(){{
  var canvas = document.getElementById('{cid}');
  if (!canvas) return;
  var ctx = canvas.getContext('2d');
  var W = 320, H = 240, t = 0;
  var cells = {cells_js};
  var colors = {colors_js};
  var ROWS = {rows}, COLS = {cols};

  // Isometric project + rotate around Y axis
  function project(x, y, z, angle) {{
    var cos = Math.cos(angle), sin = Math.sin(angle);
    var rx = x * cos - z * sin;
    var rz = x * sin + z * cos;
    var sx = (rx - rz) * 14 + W/2;
    var sy = (-y * 10 + (rx + rz) * 7) + H/2;
    return [sx, sy];
  }}

  // Draw tetrahedral edge (4 vertices of tetrahedron)
  function drawTet(cx, cy, cz, size, angle, alpha) {{
    var verts = [
      [cx, cy + size, cz],
      [cx + size*0.94, cy - size*0.33, cz - size*0.33],
      [cx - size*0.47, cy - size*0.33, cz + size*0.82],
      [cx - size*0.47, cy - size*0.33, cz - size*0.82]
    ];
    var edges = [[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]];
    ctx.strokeStyle = `rgba(255,215,0,${{alpha}})`;
    ctx.lineWidth = 0.8;
    edges.forEach(function(e) {{
      var p0 = project(verts[e[0]][0], verts[e[0]][1], verts[e[0]][2], angle);
      var p1 = project(verts[e[1]][0], verts[e[1]][1], verts[e[1]][2], angle);
      ctx.beginPath(); ctx.moveTo(p0[0],p0[1]); ctx.lineTo(p1[0],p1[1]); ctx.stroke();
    }});
  }}

  // Draw prismatic cell
  function drawCell(r, c, v, angle) {{
    var col = colors[v] || '#333';
    var cx = (c - COLS/2) * 1.8;
    var cz = (r - ROWS/2) * 1.8;
    var h = 0.8 + (v / 9) * 1.2;
    // 4 top corners
    var top = [
      project(cx-0.9, h, cz-0.9, angle),
      project(cx+0.9, h, cz-0.9, angle),
      project(cx+0.9, h, cz+0.9, angle),
      project(cx-0.9, h, cz+0.9, angle),
    ];
    var bot = [
      project(cx-0.9, 0, cz-0.9, angle),
      project(cx+0.9, 0, cz-0.9, angle),
      project(cx+0.9, 0, cz+0.9, angle),
      project(cx-0.9, 0, cz+0.9, angle),
    ];
    // Draw sides
    ctx.fillStyle = col + '88';
    ctx.strokeStyle = col;
    ctx.lineWidth = 0.5;
    [[0,1],[1,2],[2,3],[3,0]].forEach(function(e) {{
      ctx.beginPath();
      ctx.moveTo(top[e[0]][0], top[e[0]][1]);
      ctx.lineTo(top[e[1]][0], top[e[1]][1]);
      ctx.lineTo(bot[e[1]][0], bot[e[1]][1]);
      ctx.lineTo(bot[e[0]][0], bot[e[0]][1]);
      ctx.closePath();
      ctx.fill(); ctx.stroke();
    }});
    // Draw top face
    ctx.fillStyle = col + 'cc';
    ctx.beginPath();
    top.forEach(function(p, i) {{ i===0 ? ctx.moveTo(p[0],p[1]) : ctx.lineTo(p[0],p[1]); }});
    ctx.closePath(); ctx.fill(); ctx.stroke();
  }}

  function frame() {{
    ctx.clearRect(0, 0, W, H);
    // Subtle grid floor
    ctx.strokeStyle = 'rgba(100,100,150,0.15)';
    ctx.lineWidth = 0.5;
    for (var i = -5; i <= 5; i++) {{
      var a = project(i*2, 0, -10, t); var b = project(i*2, 0, 10, t);
      ctx.beginPath(); ctx.moveTo(a[0],a[1]); ctx.lineTo(b[0],b[1]); ctx.stroke();
      a = project(-10, 0, i*2, t); b = project(10, 0, i*2, t);
      ctx.beginPath(); ctx.moveTo(a[0],a[1]); ctx.lineTo(b[0],b[1]); ctx.stroke();
    }}
    // Draw cells back-to-front
    cells.slice().sort(function(a,b){{
      var azx = (a.c - COLS/2)*1.8, azz = (a.r - ROWS/2)*1.8;
      var bzx = (b.c - COLS/2)*1.8, bzz = (b.r - ROWS/2)*1.8;
      var da = Math.cos(t)*azx + Math.sin(t)*azz;
      var db = Math.cos(t)*bzx + Math.sin(t)*bzz;
      return db - da;
    }}).forEach(function(cell) {{
      drawCell(cell.r, cell.c, cell.v, t);
    }});
    // Outer tetrahedral cage
    var scale = Math.max(ROWS, COLS) * 1.2;
    drawTet(0, scale*0.5, 0, scale, t, 0.6);
    // Inner tet (counter-rotating)
    drawTet(0, scale*0.3, 0, scale*0.5, -t*0.7, 0.3);
    t += 0.008;
    requestAnimationFrame(frame);
  }}
  frame();
}})();
</script>"""

# ─── Main HTML generator ─────────────────────────────────────────────────────
def generate_html_3d(task_id: str, task_data: dict, rule: str,
                     predicted_test: Optional[List[List[int]]] = None,
                     test_ground_truth: Optional[List[List[int]]] = None) -> str:

    train_pairs = task_data.get("train", [])
    test_inp    = task_data["test"][0]["input"]
    test_out    = test_ground_truth or task_data["test"][0].get("output")

    # Build pass/fail badge
    if predicted_test and test_out:
        ok = predicted_test == test_out
        badge_col  = "#2ECC40" if ok else "#FF4136"
        badge_text = "✅ PASS" if ok else "❌ FAIL"
    else:
        badge_col  = "#FFDC00"; badge_text = "⏳ PENDING"

    # ── Entropy + histogram for first train output ─
    first_out = train_pairs[0]["output"] if train_pairs else [[0]]
    ent = _entropy(first_out)
    hist_html = _colour_hist(first_out)

    # ── Training pairs ────────────────────────────
    train_html_parts = []
    for i, ex in enumerate(train_pairs):
        inp_g = ex["input"]; out_g = ex["output"]
        # Only compute diff when grids are same size
        same_size = (len(inp_g) == len(out_g) and
                     (not inp_g or len(inp_g[0]) == len(out_g[0])))
        diff  = [[inp_g[r][c] if inp_g[r][c] == out_g[r][c] else out_g[r][c]
                  for c in range(len(inp_g[0]))] for r in range(len(inp_g))] \
                if same_size else None
        train_html_parts.append(f"""
<div class="pair-block">
  <h3 class="pair-label">Train {i+1}</h3>
  <div class="pair-row">
    <div class="grid-wrap">
      <div class="grid-title">Input</div>
      {_flat_grid(inp_g)}
    </div>
    <div class="arrow">→</div>
    <div class="grid-wrap">
      <div class="grid-title">Output</div>
      {_flat_grid(out_g)}
    </div>
    <div class="grid-wrap iso-col">
      <div class="grid-title">3-D Prism View</div>
      {_grid_to_iso_svg(out_g, cell_size=14)}
    </div>
  </div>
</div>""")

    # ── Test block ────────────────────────────────
    pred_block = ""
    if predicted_test:
        # All three grids must have matching dimensions for diff computation
        same_size = (test_out and predicted_test and
                     len(predicted_test) == len(test_out) ==
                     len(test_inp) and
                     len(predicted_test[0]) == len(test_out[0]) ==
                     len(test_inp[0]))
        match_diff = [[test_inp[r][c] if test_out[r][c] == predicted_test[r][c]
                       else predicted_test[r][c]
                       for c in range(len(predicted_test[0]))]
                      for r in range(len(predicted_test))] \
                     if same_size else None
        pred_block = f"""
    <div class="grid-wrap">
      <div class="grid-title" style="color:{badge_col}">Predicted {badge_text}</div>
      {_flat_grid(predicted_test)}
    </div>"""

    gt_block = ""
    if test_out:
        gt_block = f"""
    <div class="grid-wrap">
      <div class="grid-title">Ground Truth</div>
      {_flat_grid(test_out)}
    </div>"""

    test_html = f"""
<div class="pair-block">
  <h3 class="pair-label">Test</h3>
  <div class="pair-row">
    <div class="grid-wrap">
      <div class="grid-title">Input</div>
      {_flat_grid(test_inp)}
    </div>
    <div class="arrow">→</div>
    {pred_block}
    {gt_block}
    <div class="grid-wrap iso-col">
      <div class="grid-title">3-D Prism View (Input)</div>
      {_grid_to_iso_svg(test_inp, cell_size=14)}
    </div>
  </div>
</div>"""

    # ── 3-D canvas on last training output ────────
    canvas_grid = train_pairs[-1]["output"] if train_pairs else test_inp
    canvas_html = _webgl_canvas(task_id, canvas_grid)

    # ── Geometry panel ────────────────────────────
    geom_info = f"""
<div class="geo-panel">
  <div class="geo-title">🔷 Hybrid Prismatic / Tetrahedral Analysis</div>
  <div class="geo-row">
    <div>
      <b>Grid dimensions:</b><br>
      {"<br>".join(
          f"Train {i+1}: {len(ex['input'])}×{len(ex['input'][0])} → {len(ex['output'])}×{len(ex['output'][0])}"
          for i, ex in enumerate(train_pairs)
      )}<br>
      Test: {len(test_inp)}×{len(test_inp[0])}
    </div>
    <div>
      <b>Output entropy (Train 1):</b> {ent:.3f} bits<br>
      <b>Colour distribution:</b><br>
      {hist_html}
    </div>
    <div style="min-width:320px">
      <b>Live 3-D tetrahedral cage</b> (last train output):<br>
      {canvas_html}
    </div>
  </div>
</div>"""

    css = """
<style>
  body { font-family: 'Segoe UI', sans-serif; background: #1a1a1a; color: #eee;
         padding: 24px; margin: 0; }
  h1 { color: #FFD700; margin: 0 0 8px; }
  code { background: #2a2a2a; padding: 2px 8px; border-radius: 3px; color: #FFD700; }
  h2 { color: #aaa; font-size: 14px; font-weight: normal; margin: 0 0 16px; }
  h3 { color: #ccc; margin: 16px 0 6px; }
  .rule-box { background: #252535; padding: 14px 18px; border-left: 4px solid #FFD700;
              border-radius: 4px; margin-bottom: 20px; line-height: 1.6; }
  .badge { display: inline-block; padding: 4px 14px; border-radius: 20px;
           font-weight: bold; font-size: 15px; margin-left: 12px; }
  .pair-block { background: #222; border: 1px solid #333; border-radius: 6px;
                padding: 14px; margin-bottom: 16px; }
  .pair-label { color: #FFD700; margin: 0 0 10px; font-size: 14px; }
  .pair-row { display: flex; align-items: flex-start; flex-wrap: wrap; gap: 16px; }
  .grid-wrap { display: flex; flex-direction: column; gap: 4px; }
  .grid-title { font-size: 11px; font-weight: bold; color: #888; text-transform: uppercase;
                letter-spacing: 1px; margin-bottom: 4px; }
  .arrow { font-size: 26px; align-self: center; color: #555; }
  .iso-col { margin-left: 8px; }
  .geo-panel { background: #1e1e2e; border: 1px solid #445; border-radius: 8px;
               padding: 18px; margin-top: 20px; }
  .geo-title { color: #7FDBFF; font-weight: bold; font-size: 15px; margin-bottom: 14px; }
  .geo-row { display: flex; flex-wrap: wrap; gap: 28px; align-items: flex-start; }
  .geo-row b { color: #FFD700; }
  footer { margin-top: 40px; color: #444; font-size: 11px; }
</style>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>ARC {task_id} — 3D Reasoning</title>
{css}
</head>
<body>
<h1>ARC Task <code>{task_id}</code>
  <span class="badge" style="background:{badge_col}22;color:{badge_col};
        border:1px solid {badge_col}">{badge_text}</span>
</h1>
<h2>RE-ARC Bench · Hybrid Prismatic/Tetrahedral Reasoning Visualisation</h2>

<div class="rule-box">
  <b style="color:#FFD700">Discovered Rule:</b><br>
  <span style="color:#eee">{rule}</span>
</div>

{"".join(train_html_parts)}
{test_html}
{geom_info}

<footer>
  TranscendPlexity OctoTetrahedral AGI — RE-ARC Bench Solver Pipeline<br>
  Hybrid Prismatic/Tetrahedral 3-D Grid Generation for Complex Geometries
</footer>
</body></html>"""
