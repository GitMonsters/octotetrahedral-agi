#!/usr/bin/env python3
"""
ARC-AGI Solver for puzzle d931c21c

Rule:
- Input: binary grid (0s and 1s)
- 1-cells form contours (connected components, 4-connected)
- Closed contours (enclosing interior 0-regions) get colored:
  - Interior 0-cells 8-adjacent to 1-cells -> 3 (green)
  - Exterior 0-cells 8-adjacent to filled shape -> 2 (red)
  - Concave corners at top/bottom borders extended when adjacent row shifts by exactly 1 column
- Open contours (no enclosed region) remain unchanged
- 1-cells always stay as 1
"""

import json
import sys
from collections import deque
from typing import List, Set, Tuple, Dict


def solve(grid: List[List[int]]) -> List[List[int]]:
    H, W = len(grid), len(grid[0])

    # Find connected components of 1-cells (4-connected)
    one_comp_id = [[-1] * W for _ in range(H)]
    one_comps: List[Set[Tuple[int, int]]] = []
    for r in range(H):
        for c in range(W):
            if grid[r][c] == 1 and one_comp_id[r][c] == -1:
                cid = len(one_comps)
                cells: Set[Tuple[int, int]] = set()
                q = deque([(r, c)])
                one_comp_id[r][c] = cid
                while q:
                    cr, cc = q.popleft()
                    cells.add((cr, cc))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] == 1 and one_comp_id[nr][nc] == -1:
                            one_comp_id[nr][nc] = cid
                            q.append((nr, nc))
                one_comps.append(cells)

    # For each 1-component, find enclosed 0-cells via flood fill from border
    filled_list = []
    for oc_cells in one_comps:
        reachable = [[False] * W for _ in range(H)]
        q = deque()
        for r in range(H):
            for c in [0, W - 1]:
                if (r, c) not in oc_cells and not reachable[r][c]:
                    reachable[r][c] = True
                    q.append((r, c))
        for c in range(W):
            for r in [0, H - 1]:
                if (r, c) not in oc_cells and not reachable[r][c]:
                    reachable[r][c] = True
                    q.append((r, c))
        while q:
            cr, cc = q.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < H and 0 <= nc < W and not reachable[nr][nc] and (nr, nc) not in oc_cells:
                    reachable[nr][nc] = True
                    q.append((nr, nc))

        enclosed = {(r, c) for r in range(H) for c in range(W)
                    if grid[r][c] == 0 and not reachable[r][c]}
        if enclosed:
            filled_list.append((oc_cells, enclosed, oc_cells | enclosed))

    # Build output
    out = [row[:] for row in grid]
    all_filled: Set[Tuple[int, int]] = set()

    for oc_cells, enclosed, filled in filled_list:
        all_filled |= filled
        # Interior 0-cells 8-adjacent to 1-boundary -> 3
        for r, c in enclosed:
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    if (r + dr, c + dc) in oc_cells:
                        out[r][c] = 3
                        break
                if out[r][c] == 3:
                    break

    # Exterior 0-cells 8-adjacent to any filled cell -> 2
    for r in range(H):
        for c in range(W):
            if grid[r][c] == 0 and (r, c) not in all_filled:
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        if (r + dr, c + dc) in all_filled:
                            out[r][c] = 2
                            break
                    if out[r][c] == 2:
                        break

    # Fix concave corners at top/bottom borders where adjacent row shifts by 1
    for oc_cells, enclosed, filled in filled_list:
        row_cols: Dict[int, Tuple[int, int]] = {}
        for r, c in filled:
            if r not in row_cols:
                row_cols[r] = (c, c)
            else:
                row_cols[r] = (min(row_cols[r][0], c), max(row_cols[r][1], c))

        if not row_cols:
            continue
        sorted_rows = sorted(row_cols.keys())
        if len(sorted_rows) < 2:
            continue

        # Top border: extend corner if second row shifts by exactly 1
        above_r = sorted_rows[0] - 1
        if above_r >= 0:
            top_left, top_right = row_cols[sorted_rows[0]]
            next_left, next_right = row_cols[sorted_rows[1]]
            if top_left - next_left == 1:
                c = next_left - 1
                if 0 <= c < W and out[above_r][c] == 0:
                    out[above_r][c] = 2
            if next_right - top_right == 1:
                c = next_right + 1
                if 0 <= c < W and out[above_r][c] == 0:
                    out[above_r][c] = 2

        # Bottom border: extend corner if second-to-last row shifts by exactly 1
        below_r = sorted_rows[-1] + 1
        if below_r < H:
            bot_left, bot_right = row_cols[sorted_rows[-1]]
            prev_left, prev_right = row_cols[sorted_rows[-2]]
            if bot_left - prev_left == 1:
                c = prev_left - 1
                if 0 <= c < W and out[below_r][c] == 0:
                    out[below_r][c] = 2
            if prev_right - bot_right == 1:
                c = prev_right + 1
                if 0 <= c < W and out[below_r][c] == 0:
                    out[below_r][c] = 2

    return out


def main():
    task_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/arc_task_d931c21c.json"
    with open(task_path) as f:
        task = json.load(f)

    # Verify on training examples
    for idx, ex in enumerate(task["train"]):
        predicted = solve(ex["input"])
        match = predicted == ex["output"]
        print(f"Train {idx}: {'PASS' if match else 'FAIL'}")

    # Solve test examples
    for idx, ex in enumerate(task["test"]):
        result = solve(ex["input"])
        print(f"\nTest {idx} output ({len(result)}x{len(result[0])}):")
        for row in result:
            print("".join(str(c) for c in row))

    # Save submission
    submission = {"test": [{"output": solve(ex["input"])} for ex in task["test"]]}
    out_path = task_path.replace(".json", "_solution.json")
    with open(out_path, "w") as f:
        json.dump(submission, f)
    print(f"\nSolution saved to {out_path}")


if __name__ == "__main__":
    main()
