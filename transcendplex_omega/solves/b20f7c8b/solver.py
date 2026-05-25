"""
Solver for ARC-AGI task b20f7c8b

Rule:
- Grid has a "key panel" (bg=8) with colored shapes and a "data panel" (bg=0) with boxes.
- Key panel contains 4 shapes, each a unique color, drawn on bg 8.
- Data panel contains 4 boxes (5x5 rectangles). Each box is either:
  - PATTERNED: border of 2s, interior has 1s (foreground) and 2s (background)
  - SOLID: entirely one color matching a key shape
- Transform:
  - Patterned box -> match inner 1-pattern to a key shape (any dihedral symmetry) -> fill entire box with that shape's color
  - Solid box -> find key shape with that color -> convert to 2-bordered box with shape's pattern as 1s inside (no rotation)
"""
import json
from typing import List

Grid = List[List[int]]


def rot90(p: list[list[int]]) -> list[list[int]]:
    """Rotate pattern 90 degrees clockwise."""
    h, w = len(p), len(p[0])
    return [[p[h - 1 - c][r] for c in range(h)] for r in range(w)]


def hflip(p: list[list[int]]) -> list[list[int]]:
    """Horizontal flip."""
    return [row[::-1] for row in p]


def matches_dihedral(inner: list[list[int]], pat: list[list[int]]) -> bool:
    """Check if inner matches pat under any of 8 dihedral transforms."""
    p = [row[:] for row in pat]
    for _ in range(4):
        if p == inner:
            return True
        if hflip(p) == inner:
            return True
        p = rot90(p)
    return False


def solve(grid: Grid) -> Grid:
    grid = [row[:] for row in grid]
    rows, cols = len(grid), len(grid[0])

    # Determine panel membership (0=data, 8=key) for each cell
    panel = [[None] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 0:
                panel[r][c] = 0
            elif grid[r][c] == 8:
                panel[r][c] = 8

    changed = True
    while changed:
        changed = False
        for r in range(rows):
            for c in range(cols):
                if panel[r][c] is None:
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and panel[nr][nc] is not None:
                            panel[r][c] = panel[nr][nc]
                            changed = True
                            break

    # Extract shapes from key panel (non-8 cells grouped by color)
    shape_cells: dict[int, list[tuple[int, int]]] = {}
    for r in range(rows):
        for c in range(cols):
            if panel[r][c] == 8 and grid[r][c] != 8:
                shape_cells.setdefault(grid[r][c], []).append((r, c))

    shape_patterns: dict[int, list[list[int]]] = {}
    for color, cells in shape_cells.items():
        min_r = min(r for r, _ in cells)
        min_c = min(c for _, c in cells)
        max_r = max(r for r, _ in cells)
        max_c = max(c for _, c in cells)
        pat = [[0] * (max_c - min_c + 1) for _ in range(max_r - min_r + 1)]
        for r, c in cells:
            pat[r - min_r][c - min_c] = 1
        shape_patterns[color] = pat

    # Find rectangular boxes in data panel (contiguous non-0 blocks)
    visited: set[tuple[int, int]] = set()
    boxes: list[tuple[int, int, int, int]] = []
    for r in range(rows):
        for c in range(cols):
            if panel[r][c] == 0 and grid[r][c] != 0 and (r, c) not in visited:
                r2 = r
                while r2 + 1 < rows and grid[r2 + 1][c] != 0 and panel[r2 + 1][c] == 0:
                    r2 += 1
                c2 = c
                while c2 + 1 < cols and grid[r][c2 + 1] != 0 and panel[r][c2 + 1] == 0:
                    c2 += 1
                for rr in range(r, r2 + 1):
                    for cc in range(c, c2 + 1):
                        visited.add((rr, cc))
                boxes.append((r, c, r2, c2))

    # Process each box
    for r1, c1, r2, c2 in boxes:
        vals = set()
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                vals.add(grid[r][c])

        if len(vals) == 1:
            # SOLID box -> convert to patterned (2-border + shape as 1s)
            solid_color = next(iter(vals))
            if solid_color in shape_patterns:
                pat = shape_patterns[solid_color]
                for r in range(r1, r2 + 1):
                    for c in range(c1, c2 + 1):
                        if r == r1 or r == r2 or c == c1 or c == c2:
                            grid[r][c] = 2
                        else:
                            ir, ic = r - r1 - 1, c - c1 - 1
                            grid[r][c] = 1 if (ir < len(pat) and ic < len(pat[0]) and pat[ir][ic]) else 2
        else:
            # PATTERNED box -> match to shape -> fill solid
            inner_h = r2 - r1 - 1
            inner_w = c2 - c1 - 1
            inner = []
            for ir in range(inner_h):
                row = []
                for ic in range(inner_w):
                    row.append(1 if grid[r1 + 1 + ir][c1 + 1 + ic] == 1 else 0)
                inner.append(row)

            matched_color = None
            for color, pat in shape_patterns.items():
                if matches_dihedral(inner, pat):
                    matched_color = color
                    break

            if matched_color is not None:
                for r in range(r1, r2 + 1):
                    for c in range(c1, c2 + 1):
                        grid[r][c] = matched_color

    return grid


if __name__ == "__main__":
    task = json.load(open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/b20f7c8b.json"))

    colors = ['#000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00', '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25']

    all_pass = True
    for i, pair in enumerate(task["train"]):
        result = solve(pair["input"])
        ok = result == pair["output"]
        print(f"Train {i}: {'PASS' if ok else 'FAIL'}")
        if not ok:
            all_pass = False
            for r in range(len(result)):
                if result[r] != pair["output"][r]:
                    print(f"  Row {r}: got    {result[r]}")
                    print(f"         expect {pair['output'][r]}")

    for i, pair in enumerate(task["test"]):
        result = solve(pair["input"])
        print(f"Test {i}: solved ({len(result)}x{len(result[0])})")
        if "output" in pair:
            ok = result == pair["output"]
            print(f"  -> {'PASS' if ok else 'FAIL'}")
            if not ok:
                all_pass = False

    if all_pass:
        print("\nALL TRAIN PASSED!")

    # Save test output
    test_result = solve(task["test"][0]["input"])
    with open("/Users/evanpieser/arc-puzzle-catalog/solves/b20f7c8b/solution.json", "w") as f:
        json.dump(test_result, f)

    # Generate HTML visualization
    html = ['<!DOCTYPE html><html><head><style>',
            'body{font-family:monospace;background:#1a1a2e;color:#fff;padding:20px}',
            '.pair{display:flex;gap:20px;margin:20px 0;align-items:start}',
            '.grid{display:inline-block;border:1px solid #333}',
            '.grid td{width:18px;height:18px;padding:0;border:1px solid #222}',
            'h2,h3{margin:10px 0}',
            '</style></head><body>',
            f'<h2>ARC Task b20f7c8b</h2>']

    def grid_to_html(g, label):
        s = f'<div><h3>{label}</h3><table class="grid">'
        for row in g:
            s += '<tr>'
            for v in row:
                s += f'<td style="background:{colors[v]}"></td>'
            s += '</tr>'
        s += '</table></div>'
        return s

    for i, pair in enumerate(task["train"]):
        result = solve(pair["input"])
        html.append(f'<h3>Train {i}</h3><div class="pair">')
        html.append(grid_to_html(pair["input"], "Input"))
        html.append(grid_to_html(pair["output"], "Expected"))
        html.append(grid_to_html(result, "Got"))
        html.append('</div>')

    html.append(f'<h3>Test 0</h3><div class="pair">')
    html.append(grid_to_html(task["test"][0]["input"], "Input"))
    html.append(grid_to_html(test_result, "Output"))
    html.append('</div>')
    html.append('</body></html>')

    with open("/Users/evanpieser/arc-puzzle-catalog/solves/b20f7c8b/visualize.html", "w") as f:
        f.write('\n'.join(html))
    print("Saved solution.json and visualize.html")
