import json, sys
from collections import Counter

def solve(grid):
    H = len(grid)
    W = len(grid[0])
    bg = Counter(grid[r][c] for r in range(H) for c in range(W)).most_common(1)[0][0]

    cells = {}
    for r in range(H):
        for c in range(W):
            if grid[r][c] != bg:
                cells[(r, c)] = grid[r][c]

    # BFS connected components
    visited = set()
    components = []
    for (r, c) in cells:
        if (r, c) in visited:
            continue
        comp = []
        queue = [(r, c)]
        visited.add((r, c))
        while queue:
            cr, cc = queue.pop(0)
            comp.append((cr, cc))
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = cr + dr, cc + dc
                if (nr, nc) not in visited and (nr, nc) in cells:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        components.append(comp)

    multi = [c for c in components if len(c) > 1]
    singles = [c[0] for c in components if len(c) == 1]
    edge_set = {(r, c) for (r, c) in singles
                if r == 0 or r == H - 1 or c == 0 or c == W - 1}

    # Map each color to its edge dot(s)
    edge_dot_by_color: dict[int, list[tuple[int, int]]] = {}
    for (r, c) in edge_set:
        edge_dot_by_color.setdefault(cells[(r, c)], []).append((r, c))

    def _alignment(dot_r, dot_c):
        """Return (axis, target_value, edge_side)."""
        if dot_r == 0 or dot_r == H - 1:
            return 'col', dot_c, ('top' if dot_r == 0 else 'bottom')
        return 'row', dot_r, ('left' if dot_c == 0 else 'right')

    out = [[bg] * W for _ in range(H)]

    # 1. Place edge dots at their original positions
    for (r, c) in edge_set:
        out[r][c] = cells[(r, c)]

    # 2. Move each multi-cell shape toward its matching edge dot
    for comp in multi:
        comp_colors = Counter(cells[(r, c)] for r, c in comp)

        # Marker = least-common color that has a matching edge dot
        marker_color = None
        marker_dot = None
        for color, _ in sorted(comp_colors.items(), key=lambda x: x[1]):
            if color in edge_dot_by_color:
                marker_color = color
                marker_dot = edge_dot_by_color[color][0]
                break

        if marker_color is None:
            for (r, c) in comp:
                out[r][c] = cells[(r, c)]
            continue

        marker_cell = None
        for (r, c) in comp:
            if cells[(r, c)] == marker_color:
                marker_cell = (r, c)
                break

        axis, target, edge_side = _alignment(*marker_dot)
        mr, mc = marker_cell

        if axis == 'row':
            dr = target - mr
            dc = 0
            # Minimal diagonal nudge when the shape barely crosses the target
            if dr == 1:
                dc = -1 if edge_side == 'left' else (1 if edge_side == 'right' else 0)
        else:
            dr = 0
            dc = target - mc

        for (r, c) in comp:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                out[nr][nc] = cells[(r, c)]

    # 3. Move interior (non-edge) single dots to their matching edge-dot row/col
    for (r, c) in singles:
        if (r, c) in edge_set:
            continue
        color = cells[(r, c)]
        if color in edge_dot_by_color:
            dot = edge_dot_by_color[color][0]
            axis, target, _ = _alignment(*dot)
            if axis == 'row':
                out[target][c] = color
            else:
                out[r][target] = color
        else:
            out[r][c] = color

    return out

if __name__ == '__main__':
    with open(sys.argv[1]) as f:
        task = json.load(f)
    for split in ['train', 'test']:
        for i, ex in enumerate(task[split]):
            result = solve(ex['input'])
            match = result == ex['output']
            print(f"{split} {i}: {'PASS' if match else 'FAIL'}")
            if not match:
                H, W = len(ex['output']), len(ex['output'][0])
                diffs = [(r, c, result[r][c], ex['output'][r][c])
                         for r in range(H) for c in range(W)
                         if result[r][c] != ex['output'][r][c]]
                print(f"  {len(diffs)} diffs")
                for r, c, got, exp in diffs:
                    print(f"    ({r},{c}): got={got} expected={exp}")
