"""
Solver for ARC-AGI-2 task 88bcf3b4
Rule: Wall-Anchor Perpendicular Distance Profile Reflection

Each grid contains one or more independent triplets:
  - WALL: straight line (stays fixed)
  - MOVER: bent/staircase shape adjacent to wall (gets repositioned)
  - ANCHOR: static shape defining reflection profile

The mover flips from extending away from the anchor to extending
toward and past it, with dist = anchor_dist + 1 at anchor rows.
"""
import json
from collections import Counter, deque


def find_components(cells, connectivity=8):
    cell_set = set(cells)
    visited = set()
    components = []
    deltas = [(-1,0),(1,0),(0,-1),(0,1)]
    if connectivity == 8:
        deltas += [(-1,-1),(-1,1),(1,-1),(1,1)]
    for start in cells:
        if start in visited:
            continue
        comp = []
        queue = deque([start])
        visited.add(start)
        while queue:
            r, c = queue.popleft()
            comp.append((r, c))
            for dr, dc in deltas:
                nr, nc = r + dr, c + dc
                if (nr, nc) in cell_set and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        components.append(comp)
    return components


def is_straight_line(cells):
    if len(cells) <= 1:
        return True
    return len(set(r for r, c in cells)) == 1 or len(set(c for r, c in cells)) == 1


def solve_triplet(rows, cols, bg, wall_cells, mover_cells, anchor_cells):
    w_set = set(wall_cells)
    w_rs = set(r for r, c in wall_cells)
    w_cs = set(c for r, c in wall_cells)
    wall_vertical = len(w_cs) == 1
    wall_pos = list(w_cs)[0] if wall_vertical else list(w_rs)[0]

    # Find all junction candidates: mover cells adjacent to wall
    junctions = []
    for r, c in mover_cells:
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            if (r + dr, c + dc) in w_set:
                junctions.append((r, c))
                break
    if not junctions:
        return []

    # Define coordinate system based on wall orientation
    ref = junctions[0]
    if wall_vertical:
        side = 'left' if ref[1] < wall_pos else 'right'
        get_dist = lambda r, c: abs(c - wall_pos)
        pos_from_dist = lambda row, dist: (row, wall_pos - dist) if side == 'left' else (row, wall_pos + dist)
        get_row = lambda r, c: r
        grid_limit = rows
    else:
        side = 'above' if ref[0] < wall_pos else 'below'
        get_dist = lambda r, c: abs(r - wall_pos)
        pos_from_dist = lambda row, dist: (wall_pos - dist, row) if side == 'above' else (wall_pos + dist, row)
        get_row = lambda r, c: c
        grid_limit = cols

    # Pick junction closest to anchor along wall axis
    if len(junctions) > 1 and anchor_cells:
        anchor_avg = sum(get_row(r, c) for r, c in anchor_cells) / len(anchor_cells)
        junction = min(junctions, key=lambda j: abs(get_row(*j) - anchor_avg))
    else:
        junction = junctions[0]

    # Anchor distance profile
    anchor_by_row = {}
    for r, c in anchor_cells:
        row, d = get_row(r, c), get_dist(r, c)
        anchor_by_row[row] = max(anchor_by_row.get(row, 0), d)

    junction_row = get_row(*junction)

    # Direction: output goes toward anchor
    if anchor_by_row:
        anchor_center = sum(anchor_by_row.keys()) / len(anchor_by_row)
        direction = -1 if anchor_center < junction_row else 1
    else:
        m_center = sum(get_row(r, c) for r, c in mover_cells) / len(mover_cells)
        direction = -1 if m_center > junction_row else 1

    # Build distance profile
    profile = {}
    if anchor_by_row:
        anchor_rows_sorted = sorted(anchor_by_row.keys(), reverse=(direction == -1))
        first_anchor = anchor_rows_sorted[0]
        last_anchor = anchor_rows_sorted[-1]

        # Approach: linear from junction (dist 1) to first anchor (anchor_dist + 1)
        approach_n = abs(first_anchor - junction_row)
        target_dist = anchor_by_row[first_anchor] + 1
        for step in range(approach_n + 1):
            row = junction_row + direction * step
            t = step / approach_n if approach_n > 0 else 0
            dist = 1 + t * (target_dist - 1)
            profile[row] = round(dist)

        # Anchor rows: dist = anchor_dist + 1
        for row, adist in anchor_by_row.items():
            profile[row] = adist + 1

        # Beyond anchor: decrease by 1 per row
        last_dist = anchor_by_row[last_anchor] + 1
        step = 1
        while True:
            row = last_anchor + direction * step
            if row < 0 or row >= grid_limit:
                break
            dist = last_dist - step
            profile[row] = dist
            step += 1
    else:
        row = junction_row
        while 0 <= row < grid_limit:
            profile[row] = 1
            row += direction

    # Generate output cells, cap at input mover count
    result = []
    row = junction_row
    while 0 <= row < grid_limit and len(result) < len(mover_cells):
        if row in profile:
            pos = pos_from_dist(row, profile[row])
            r, c = pos
            if 0 <= r < rows and 0 <= c < cols:
                result.append(pos)
        row += direction
    return result


def solve(grid):
    rows, cols = len(grid), len(grid[0])
    bg = Counter(grid[r][c] for r in range(rows) for c in range(cols)).most_common(1)[0][0]

    color_cells = {}
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg:
                color_cells.setdefault(grid[r][c], []).append((r, c))

    # Split each color into 8-connected components
    all_components = []
    for color, cells in color_cells.items():
        for comp in find_components(cells, 8):
            all_components.append({
                'color': color,
                'cells': comp,
                'is_line': is_straight_line(comp)
            })

    walls = [c for c in all_components if c['is_line'] and len(c['cells']) > 1]
    non_walls = [c for c in all_components if not c['is_line']]
    singles = [c for c in all_components if c['is_line'] and len(c['cells']) <= 1]

    # Pair each bent shape with its adjacent wall
    triplets = []
    used_walls = set()
    used_movers = set()

    for mi, mcomp in enumerate(non_walls):
        best_wall = None
        for wi, wcomp in enumerate(walls):
            if wi in used_walls:
                continue
            ws = set(wcomp['cells'])
            adjacent = any(
                (r + dr, c + dc) in ws
                for r, c in mcomp['cells']
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]
            )
            if adjacent:
                if best_wall is None or len(wcomp['cells']) > len(walls[best_wall]['cells']):
                    best_wall = wi
        if best_wall is not None:
            triplets.append({
                'wall': walls[best_wall],
                'mover': mcomp,
                'anchor_cells': []
            })
            used_walls.add(best_wall)
            used_movers.add(mi)

    # Remaining components become anchors for nearest triplet
    remaining = (
        [c for i, c in enumerate(non_walls) if i not in used_movers] +
        [c for i, c in enumerate(walls) if i not in used_walls] +
        singles
    )

    for rcomp in remaining:
        rcenter = (
            sum(r for r, c in rcomp['cells']) / len(rcomp['cells']),
            sum(c for r, c in rcomp['cells']) / len(rcomp['cells'])
        )
        best_dist, best_tri = float('inf'), None
        for tri in triplets:
            wc = tri['wall']['cells']
            wcenter = (sum(r for r, c in wc) / len(wc), sum(c for r, c in wc) / len(wc))
            d = abs(rcenter[0] - wcenter[0]) + abs(rcenter[1] - wcenter[1])
            if d < best_dist:
                best_dist, best_tri = d, tri
        if best_tri:
            best_tri['anchor_cells'].extend(rcomp['cells'])

    # Apply transformation
    output = [row[:] for row in grid]
    for tri in triplets:
        for r, c in tri['mover']['cells']:
            output[r][c] = bg
        new_positions = solve_triplet(
            rows, cols, bg,
            tri['wall']['cells'],
            tri['mover']['cells'],
            tri['anchor_cells']
        )
        for r, c in new_positions:
            output[r][c] = tri['mover']['color']

    return output


if __name__ == '__main__':
    with open('ARC_AMD_TRANSFER/data/ARC-AGI-2/data/evaluation/88bcf3b4.json') as f:
        task = json.load(f)

    print("Verifying on training examples...")
    for ti, ex in enumerate(task['train']):
        pred = solve(ex['input'])
        expected = ex['output']
        rows, cols = len(expected), len(expected[0])
        correct = sum(1 for r in range(rows) for c in range(cols) if pred[r][c] == expected[r][c])
        total = rows * cols
        print(f"  Train {ti}: {correct}/{total} {'✓' if correct == total else '✗'}")

    print("\nGenerating test predictions...")
    for ti, test in enumerate(task['test']):
        pred = solve(test['input'])
        print(f"  Test {ti}: {len(pred)}x{len(pred[0])} grid")
