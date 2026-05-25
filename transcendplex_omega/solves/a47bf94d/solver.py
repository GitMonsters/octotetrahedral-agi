"""Solver for ARC-AGI task a47bf94d.

The puzzle features a maze of 8/5/9 cells connecting colored slots on two
opposite sides. Solid 3x3 blocks become diamond patterns (.X./X.X/.X.),
and inverse diamond patterns (X.X/.X./X.X) appear at paired slots on the
opposite side. The maze routing (with 5s as non-connecting crossings)
determines which slots are paired.
"""

import json
import sys


MAZE_VALS = {5, 8, 9}
DIAMOND = [(0, 1), (1, 0), (1, 2), (2, 1)]           # .X./X.X/.X.
INVERSE = [(0, 0), (0, 2), (1, 1), (2, 0), (2, 2)]   # X.X/.X./X.X


def solve(grid):
    H, W = len(grid), len(grid[0])
    out = [row[:] for row in grid]

    # --- 1. Find maze bounding box ---
    maze_rmin = maze_rmax = maze_cmin = maze_cmax = None
    for r in range(H):
        for c in range(W):
            if grid[r][c] in MAZE_VALS:
                if maze_rmin is None:
                    maze_rmin = maze_rmax = r
                    maze_cmin = maze_cmax = c
                else:
                    maze_rmin, maze_rmax = min(maze_rmin, r), max(maze_rmax, r)
                    maze_cmin, maze_cmax = min(maze_cmin, c), max(maze_cmax, c)

    # --- 2. Find colored patterns in input ---
    patterns = _find_patterns(grid)

    # --- 3. Determine orientation from color positions ---
    orientation = _determine_orientation(patterns, maze_rmin, maze_rmax, maze_cmin, maze_cmax)

    # --- 4. Find wire entry/exit positions ---
    if orientation == 'TB':
        wire_positions = _find_wire_cols(grid, maze_rmin, maze_rmax)
        side_a_slots = [(maze_rmin - 3, wp - 1) for wp in wire_positions]  # top
        side_b_slots = [(maze_rmax + 1, wp - 1) for wp in wire_positions]  # bottom
        entry_a = [(maze_rmin, wp) for wp in wire_positions]
        entry_b = [(maze_rmax, wp) for wp in wire_positions]
        init_dir_a = (1, 0)   # going down
        init_dir_b = (-1, 0)  # going up
    else:  # LR
        wire_positions = _find_wire_rows(grid, maze_cmin, maze_cmax)
        side_a_slots = [(wp - 1, maze_cmin - 3) for wp in wire_positions]  # left
        side_b_slots = [(wp - 1, maze_cmax + 1) for wp in wire_positions]  # right
        entry_a = [(wp, maze_cmin) for wp in wire_positions]
        entry_b = [(wp, maze_cmax) for wp in wire_positions]
        init_dir_a = (0, 1)   # going right
        init_dir_b = (0, -1)  # going left

    n = len(wire_positions)

    # --- 5. Trace wires: side_a[i] -> side_b[perm[i]] ---
    perm = {}
    entry_b_set = {pos: idx for idx, pos in enumerate(entry_b)}
    target_set = set(entry_b)
    for i, ep in enumerate(entry_a):
        path = _trace_wire(grid, ep[0], ep[1], init_dir_a[0], init_dir_a[1], targets=target_set)
        end = path[-1]
        if end in entry_b_set:
            perm[i] = entry_b_set[end]

    # --- 6. Collect known colors from input patterns ---
    color_at_a = {}  # slot index -> color
    color_at_b = {}

    for ptype, color, pr, pc in patterns:
        for i, (sr, sc) in enumerate(side_a_slots):
            if pr == sr and pc == sc:
                color_at_a[i] = color
        for i, (sr, sc) in enumerate(side_b_slots):
            if pr == sr and pc == sc:
                color_at_b[i] = color

    # Propagate colors through the permutation
    inv_perm = {v: k for k, v in perm.items()}
    for i in range(n):
        if i in color_at_a and perm.get(i) is not None:
            j = perm[i]
            if j not in color_at_b:
                color_at_b[j] = color_at_a[i]
        if i in color_at_b and inv_perm.get(i) is not None:
            j = inv_perm[i]
            if j not in color_at_a:
                color_at_a[j] = color_at_b[i]

    # --- 7. Generate output ---
    # Clear all non-maze, non-zero cells (colored cells)
    for r in range(H):
        for c in range(W):
            if grid[r][c] not in MAZE_VALS and grid[r][c] != 0:
                out[r][c] = 0

    # Place diamonds on side A
    for i, (sr, sc) in enumerate(side_a_slots):
        if i in color_at_a:
            color = color_at_a[i]
            for dr, dc in DIAMOND:
                out[sr + dr][sc + dc] = color

    # Place inverse diamonds on side B
    for i, (sr, sc) in enumerate(side_b_slots):
        if i in color_at_b:
            color = color_at_b[i]
            for dr, dc in INVERSE:
                out[sr + dr][sc + dc] = color

    return out


def _find_patterns(grid):
    """Find all 3x3 colored patterns (solid, diamond, inverse) in the grid."""
    H, W = len(grid), len(grid[0])
    results = []
    for r in range(H - 2):
        for c in range(W - 2):
            cells = [grid[r + dr][c + dc] for dr in range(3) for dc in range(3)]
            colors = set(v for v in cells if v not in MAZE_VALS and v != 0)
            if len(colors) != 1:
                continue
            color = colors.pop()
            mask = tuple(1 if cells[i] == color else 0 for i in range(9))
            if mask == (1, 1, 1, 1, 1, 1, 1, 1, 1):
                results.append(('solid', color, r, c))
            elif mask == (0, 1, 0, 1, 0, 1, 0, 1, 0):
                results.append(('diamond', color, r, c))
            elif mask == (1, 0, 1, 0, 1, 0, 1, 0, 1):
                results.append(('inverse', color, r, c))
    return results


def _determine_orientation(patterns, maze_rmin, maze_rmax, maze_cmin, maze_cmax):
    """Determine if maze connects top-bottom or left-right."""
    above = below = left = right = 0
    for _, _, r, c in patterns:
        if r + 2 < maze_rmin:
            above += 1
        if r > maze_rmax:
            below += 1
        if c + 2 < maze_cmin:
            left += 1
        if c > maze_cmax:
            right += 1
    tb_score = above + below
    lr_score = left + right
    return 'TB' if tb_score >= lr_score else 'LR'


def _find_wire_cols(grid, maze_rmin, maze_rmax):
    """Find columns that have wire entries at both top and bottom of maze."""
    W = len(grid[0])
    cols = []
    for c in range(W):
        if (grid[maze_rmin][c] in MAZE_VALS and
                (maze_rmin == 0 or grid[maze_rmin - 1][c] not in MAZE_VALS) and
                grid[maze_rmax][c] in MAZE_VALS and
                (maze_rmax == len(grid) - 1 or grid[maze_rmax + 1][c] not in MAZE_VALS)):
            cols.append(c)
    return cols


def _find_wire_rows(grid, maze_cmin, maze_cmax):
    """Find rows that have wire entries at both left and right of maze."""
    H = len(grid)
    rows = []
    for r in range(H):
        if (grid[r][maze_cmin] in MAZE_VALS and
                (maze_cmin == 0 or grid[r][maze_cmin - 1] not in MAZE_VALS) and
                grid[r][maze_cmax] in MAZE_VALS and
                (maze_cmax == len(grid[0]) - 1 or grid[r][maze_cmax + 1] not in MAZE_VALS)):
            rows.append(r)
    return rows


def _trace_wire(grid, start_r, start_c, init_dr, init_dc, targets=None):
    """Trace a wire through the maze using DFS with backtracking.

    At 5 cells: continue straight only (crossing).
    At 8/9 cells: can turn in any direction (but not reverse).
    Prefers continuing straight to minimize wrong turns.
    """
    H, W = len(grid), len(grid[0])
    sys.setrecursionlimit(max(10000, H * W * 4))

    def _dfs(r, c, prev_r, prev_c, visited):
        if targets is not None and (r, c) in targets:
            return [(r, c)]

        val = grid[r][c]
        if val == 5:
            dr, dc = r - prev_r, c - prev_c
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] in MAZE_VALS and (nr, nc) not in visited:
                visited.add((nr, nc))
                result = _dfs(nr, nc, r, c, visited)
                if result is not None:
                    return [(r, c)] + result
                visited.remove((nr, nc))
        else:
            dr, dc = r - prev_r, c - prev_c
            straight = (r + dr, c + dc)
            cands = []
            for ddr, ddc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + ddr, c + ddc
                if ((nr, nc) != (prev_r, prev_c) and
                        0 <= nr < H and 0 <= nc < W and
                        grid[nr][nc] in MAZE_VALS and
                        (nr, nc) not in visited):
                    cands.append((nr, nc))
            # Prefer straight direction to guide search
            cands.sort(key=lambda x: 0 if x == straight else 1)
            for nr, nc in cands:
                visited.add((nr, nc))
                result = _dfs(nr, nc, r, c, visited)
                if result is not None:
                    return [(r, c)] + result
                visited.remove((nr, nc))

        if targets is None:
            return [(r, c)]  # greedy: return what we have
        return None

    visited = {(start_r, start_c)}
    result = _dfs(start_r, start_c, start_r - init_dr, start_c - init_dc, visited)
    return result if result else [(start_r, start_c)]


if __name__ == '__main__':
    with open('/Users/evanpieser/arc-puzzle-catalog/dataset/tasks/a47bf94d.json') as f:
        task = json.load(f)

    all_pass = True
    for i, ex in enumerate(task['train']):
        result = solve(ex['input'])
        expected = ex['output']
        if result == expected:
            print(f"Train {i}: PASS")
        else:
            all_pass = False
            print(f"Train {i}: FAIL")
            H, W = len(expected), len(expected[0])
            for r in range(H):
                for c in range(W):
                    if result[r][c] != expected[r][c]:
                        print(f"  ({r},{c}): got {result[r][c]}, expected {expected[r][c]}")

    print(f"\n{'ALL PASS' if all_pass else 'SOME FAILED'}")
