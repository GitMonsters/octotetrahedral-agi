"""
Solver for ARC-AGI task 67e490f4.

Pattern: A template rectangle (bordered by a non-background color) contains
"holes" (connected components of background-colored cells). Various small
colored shapes are scattered outside the template. Each hole's shape
(rotation/reflection invariant) matches one of the scattered shape types.
The color with the most instances of a given shape fills the corresponding
holes. The output is the template with all holes filled.
"""

import json
from collections import Counter, defaultdict


def solve(grid: list[list[int]]) -> list[list[int]]:
    rows = len(grid)
    cols = len(grid[0])

    # 1. Background = most common color
    color_count: Counter = Counter()
    for r in range(rows):
        for c in range(cols):
            color_count[grid[r][c]] += 1
    bg = color_count.most_common(1)[0][0]

    # 2. Find template: non-bg color whose bounding box has all 4 edges
    #    completely filled with that color (rectangular frame). Pick largest.
    best_template = None
    best_area = 0

    for color in color_count:
        if color == bg:
            continue
        min_r = min_c = float("inf")
        max_r = max_c = float("-inf")
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == color:
                    min_r, max_r = min(min_r, r), max(max_r, r)
                    min_c, max_c = min(min_c, c), max(max_c, c)

        top = all(grid[min_r][c] == color for c in range(min_c, max_c + 1))
        bot = all(grid[max_r][c] == color for c in range(min_c, max_c + 1))
        lft = all(grid[r][min_c] == color for r in range(min_r, max_r + 1))
        rgt = all(grid[r][max_c] == color for r in range(min_r, max_r + 1))

        if top and bot and lft and rgt:
            area = (max_r - min_r + 1) * (max_c - min_c + 1)
            if area > best_area:
                best_area = area
                best_template = (color, min_r, min_c, max_r, max_c)

    tc, tr1, tc1, tr2, tc2 = best_template
    th, tw = tr2 - tr1 + 1, tc2 - tc1 + 1

    # 3. Holes = bg-colored cells inside the template (relative coords)
    hole_cells: set[tuple[int, int]] = set()
    for r in range(th):
        for c in range(tw):
            if grid[tr1 + r][tc1 + c] == bg:
                hole_cells.add((r, c))

    # 4. Connected components (4-connected)
    def connected_components(cells):
        cells_set = set(cells)
        visited: set[tuple[int, int]] = set()
        comps = []
        for cell in sorted(cells_set):
            if cell in visited:
                continue
            comp = []
            stack = [cell]
            visited.add(cell)
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    nb = (cr + dr, cc + dc)
                    if nb in cells_set and nb not in visited:
                        visited.add(nb)
                        stack.append(nb)
            comps.append(comp)
        return comps

    hole_comps = connected_components(hole_cells)

    # 5. Canonical shape (invariant under 4 rotations × 2 reflections)
    def _normalize(coords):
        mr = min(r for r, _ in coords)
        mc = min(c for _, c in coords)
        return tuple(sorted((r - mr, c - mc) for r, c in coords))

    def canonical(cells):
        coords = list(cells)
        best = None
        for _ in range(4):
            n = _normalize(coords)
            if best is None or n < best:
                best = n
            ref = [(r, -c) for r, c in coords]
            n = _normalize(ref)
            if n < best:
                best = n
            coords = [(c, -r) for r, c in coords]  # rotate 90°
        return best

    # Map each hole cell → its canonical shape
    cell_shape: dict[tuple[int, int], tuple] = {}
    unique_shapes: set[tuple] = set()
    for comp in hole_comps:
        s = canonical(comp)
        unique_shapes.add(s)
        for cell in comp:
            cell_shape[cell] = s

    # 6. Gather all non-bg shapes outside the template, count per color
    outside: dict[int, set[tuple[int, int]]] = defaultdict(set)
    for r in range(rows):
        for c in range(cols):
            if tr1 <= r <= tr2 and tc1 <= c <= tc2:
                continue
            v = grid[r][c]
            if v != bg:
                outside[v].add((r, c))

    shape_votes: dict[tuple, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for color, cells in outside.items():
        for comp in connected_components(cells):
            s = canonical(comp)
            shape_votes[s][color] += 1

    # 7. Winning color per shape = most instances
    shape_color: dict[tuple, int] = {}
    for s in unique_shapes:
        if s in shape_votes:
            counts = shape_votes[s]
            shape_color[s] = max(counts, key=counts.get)

    # 8. Build output: template with holes filled
    output = []
    for r in range(th):
        row = []
        for c in range(tw):
            if (r, c) in cell_shape:
                row.append(shape_color.get(cell_shape[(r, c)], bg))
            else:
                row.append(grid[tr1 + r][tc1 + c])
        output.append(row)
    return output


if __name__ == "__main__":
    with open("/Users/evanpieser/arc-puzzle-catalog/dataset/tasks/67e490f4.json") as f:
        task = json.load(f)

    pairs = [(i, "train", p) for i, p in enumerate(task["train"])]
    pairs += [(i, "test", p) for i, p in enumerate(task.get("test", []))]

    for idx, kind, pair in pairs:
        result = solve(pair["input"])
        expected = pair["output"]
        status = "PASS" if result == expected else "FAIL"
        print(f"{kind} {idx}: {status}")
        if status == "FAIL":
            rr, rc = len(result), len(result[0]) if result else 0
            er, ec = len(expected), len(expected[0]) if expected else 0
            print(f"  dims: got {rr}x{rc}, expected {er}x{ec}")
            for r in range(min(rr, er)):
                for c in range(min(rc, ec)):
                    if result[r][c] != expected[r][c]:
                        print(f"  mismatch ({r},{c}): got {result[r][c]}, expected {expected[r][c]}")
