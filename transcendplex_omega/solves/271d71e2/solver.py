"""
Solver for ARC-AGI-2 puzzle 271d71e2 — "Box Slide & Diagonal Fill"

Rule: Black-bordered rectangles with grey/orange interiors slide toward
maroon rail lines. As a box moves N steps, N new orange cells fill its
interior following a perpendicular sweep order. Fully-orange boxes don't move.

Movement distance = min(distance_between_maroon_lines, remaining_grey_cells)

Fill order (perpendicular sweep from movement corner):
  UP   → row-by-row top→bottom, left→right
  DOWN → row-by-row bottom→top, right→left
  LEFT → column-by-column left→right, bottom→top
  RIGHT→ column-by-column right→left, top→bottom
"""

import json
from collections import deque
from typing import List


def solve(grid: List[List[int]]) -> List[List[int]]:
    rows, cols = len(grid), len(grid[0])
    BG = 6

    # --- Find boxes ---
    visited: set = set()
    boxes = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 0 and (r, c) not in visited:
                q = deque([(r, c)])
                visited.add((r, c))
                cells = [(r, c)]
                while q:
                    cr, cc = q.popleft()
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited and grid[nr][nc] in (0, 5, 7):
                            visited.add((nr, nc))
                            q.append((nr, nc))
                            cells.append((nr, nc))
                min_r = min(r for r, c in cells)
                max_r = max(r for r, c in cells)
                min_c = min(c for r, c in cells)
                max_c = max(c for r, c in cells)
                boxes.append((min_r, min_c, max_r, max_c))

    maroon_set = {(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 9}

    # --- Analyze each box ---
    box_info = []
    for r1, c1, r2, c2 in boxes:
        H, W = r2 - r1 - 1, c2 - c1 - 1
        n_orange = n_grey = 0
        for r in range(r1 + 1, r2):
            for c in range(c1 + 1, c2):
                if grid[r][c] == 7: n_orange += 1
                elif grid[r][c] == 5: n_grey += 1

        side = None
        near_pos = far_pos = distance = 0
        box_width = c2 - c1 + 1
        box_height = r2 - r1 + 1

        # Check ABOVE
        if r1 > 0 and sum(1 for c in range(c1, c2 + 1) if (r1 - 1, c) in maroon_set) == box_width:
            side, near_pos = 'UP', r1 - 1
            far_pos = near_pos
            for sr in range(near_pos - 1, -1, -1):
                if sum(1 for c in range(c1, c2 + 1) if (sr, c) in maroon_set) == box_width:
                    far_pos = sr
            distance = near_pos - far_pos

        # Check BELOW
        if side is None and r2 < rows - 1 and sum(1 for c in range(c1, c2 + 1) if (r2 + 1, c) in maroon_set) == box_width:
            side, near_pos = 'DOWN', r2 + 1
            far_pos = near_pos
            for sr in range(near_pos + 1, rows):
                if sum(1 for c in range(c1, c2 + 1) if (sr, c) in maroon_set) == box_width:
                    far_pos = sr
            distance = far_pos - near_pos

        # Check LEFT
        if side is None and c1 > 0 and sum(1 for r in range(r1, r2 + 1) if (r, c1 - 1) in maroon_set) == box_height:
            side, near_pos = 'LEFT', c1 - 1
            far_pos = near_pos
            for sc in range(near_pos - 1, -1, -1):
                if sum(1 for r in range(r1, r2 + 1) if (r, sc) in maroon_set) == box_height:
                    far_pos = sc
            distance = near_pos - far_pos

        # Check RIGHT
        if side is None and c2 < cols - 1 and sum(1 for r in range(r1, r2 + 1) if (r, c2 + 1) in maroon_set) == box_height:
            side, near_pos = 'RIGHT', c2 + 1
            far_pos = near_pos
            for sc in range(near_pos + 1, cols):
                if sum(1 for r in range(r1, r2 + 1) if (r, sc) in maroon_set) == box_height:
                    far_pos = sc
            distance = far_pos - near_pos

        movement = min(distance, n_grey) if side else 0
        box_info.append({
            'r1': r1, 'c1': c1, 'r2': r2, 'c2': c2,
            'H': H, 'W': W, 'n_orange': n_orange, 'n_grey': n_grey,
            'side': side, 'near_pos': near_pos, 'far_pos': far_pos,
            'distance': distance, 'movement': movement,
        })

    # --- Build output ---
    out = [[BG] * cols for _ in range(rows)]

    for bi in box_info:
        r1, c1, r2, c2 = bi['r1'], bi['c1'], bi['r2'], bi['c2']
        H, W, side, movement = bi['H'], bi['W'], bi['side'], bi['movement']

        dr = dc = 0
        if side == 'UP':     dr = -movement
        elif side == 'DOWN': dr = movement
        elif side == 'LEFT': dc = -movement
        elif side == 'RIGHT':dc = movement

        nr1, nc1, nr2, nc2 = r1 + dr, c1 + dc, r2 + dr, c2 + dc

        # Box border
        for r in range(nr1, nr2 + 1):
            for c in range(nc1, nc2 + 1):
                if r in (nr1, nr2) or c in (nc1, nc2):
                    out[r][c] = 0

        # Fill order
        fill_order = []
        if side == 'UP':
            for r in range(H):
                for c in range(W): fill_order.append((r, c))
        elif side == 'DOWN':
            for r in range(H - 1, -1, -1):
                for c in range(W - 1, -1, -1): fill_order.append((r, c))
        elif side == 'LEFT':
            for c in range(W):
                for r in range(H - 1, -1, -1): fill_order.append((r, c))
        elif side == 'RIGHT':
            for c in range(W - 1, -1, -1):
                for r in range(H): fill_order.append((r, c))
        else:
            for r in range(H):
                for c in range(W): fill_order.append((r, c))

        total_orange = bi['n_orange'] + movement
        for idx, (lr, lc) in enumerate(fill_order):
            out[nr1 + 1 + lr][nc1 + 1 + lc] = 7 if idx < total_orange else 5

        # Maroon
        near, far = bi['near_pos'], bi['far_pos']
        if side and movement > 0:
            if side in ('UP', 'DOWN'):
                for c in range(nc1, nc2 + 1): out[far][c] = 9
                if movement < bi['distance']:
                    new_near = near + dr
                    for c in range(nc1, nc2 + 1): out[new_near][c] = 9
            else:
                for r in range(nr1, nr2 + 1): out[r][far] = 9
                if movement < bi['distance']:
                    new_near = near + dc
                    for r in range(nr1, nr2 + 1): out[r][new_near] = 9
        elif side and movement == 0:
            if side in ('UP', 'DOWN'):
                for c in range(c1, c2 + 1):
                    out[near][c] = 9
                    out[far][c] = 9
            else:
                for r in range(r1, r2 + 1):
                    out[r][near] = 9
                    out[r][far] = 9

    return out


if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else 'puzzle.json'
    with open(path) as f:
        task = json.load(f)
    for i, ex in enumerate(task['train']):
        result = solve(ex['input'])
        ok = result == ex['output']
        print(f"Train {i}: {'PASS ✓' if ok else 'FAIL ✗'}")
    for i, ex in enumerate(task['test']):
        result = solve(ex['input'])
        print(f"Test {i}: predicted {len(result)}x{len(result[0])}")
        print(json.dumps(result))
