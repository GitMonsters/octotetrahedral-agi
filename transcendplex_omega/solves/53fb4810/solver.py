"""
Puzzle 53fb4810: Extend diamond tails to grid edge.

Rule:
- Grid has 2+ diamond/cross shapes made of 1s on a background of 8s
- Each diamond has a colored "tail" extending in one direction
- Some tails are complete (reach the grid edge); others are short
- Short tails are extended to the grid edge by repeating the pattern cyclically
"""
import json, sys
from collections import deque


def solve(grid: list[list[int]]) -> list[list[int]]:
    H, W = len(grid), len(grid[0])
    bg = 8
    out = [row[:] for row in grid]

    # Find connected components of 1s (diamonds)
    visited = [[False] * W for _ in range(H)]
    diamonds = []
    for r in range(H):
        for c in range(W):
            if grid[r][c] == 1 and not visited[r][c]:
                q = deque([(r, c)])
                visited[r][c] = True
                cells = []
                while q:
                    cr, cc = q.popleft()
                    cells.append((cr, cc))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] == 1 and not visited[nr][nc]:
                            visited[nr][nc] = True
                            q.append((nr, nc))
                diamonds.append(cells)

    for cells in diamonds:
        rows = [r for r, c in cells]
        cols = [c for r, c in cells]
        min_r, max_r = min(rows), max(rows)
        min_c, max_c = min(cols), max(cols)

        top_arm_cols = sorted(set(c for r, c in cells if r == min_r))
        bot_arm_cols = sorted(set(c for r, c in cells if r == max_r))
        left_arm_rows = sorted(set(r for r, c in cells if c == min_c))
        right_arm_rows = sorted(set(r for r, c in cells if c == max_c))

        directions = [
            ("up", top_arm_cols, min_r, True),
            ("down", bot_arm_cols, max_r, True),
            ("left", left_arm_rows, min_c, False),
            ("right", right_arm_rows, max_c, False),
        ]

        for dname, arm_pos, diamond_edge, is_vertical in directions:
            step = -1 if dname in ("up", "left") else 1
            start = diamond_edge + step
            limit = (-1 if step == -1 else (H if is_vertical else W))

            # Collect pattern
            pattern = []
            for pos in range(start, limit, step):
                if is_vertical:
                    vals = [grid[pos][c] for c in arm_pos]
                else:
                    vals = [grid[r][pos] for r in arm_pos]
                if all(v == bg for v in vals):
                    break
                pattern.append(vals)

            if not pattern:
                continue

            K = len(pattern)

            # Check if tail already reaches grid edge
            last_pos = start + step * (K - 1)
            edge_val = 0 if step == -1 else ((H - 1) if is_vertical else (W - 1))
            if (step == -1 and last_pos <= edge_val) or (step == 1 and last_pos >= edge_val):
                continue

            # Extend pattern to grid edge
            for i, pos in enumerate(range(start, limit, step)):
                vals = pattern[i % K]
                if is_vertical:
                    for j, c in enumerate(arm_pos):
                        out[pos][c] = vals[j]
                else:
                    for j, r in enumerate(arm_pos):
                        out[r][pos] = vals[j]

    return out


if __name__ == "__main__":
    path = sys.argv[1]
    with open(path) as f:
        data = json.load(f)
    for i, ex in enumerate(data["train"]):
        result = solve(ex["input"])
        ok = result == ex["output"]
        print(f"Train {i}: {'PASS' if ok else 'FAIL'}")
        if not ok:
            for r in range(len(result)):
                for c in range(len(result[0])):
                    if result[r][c] != ex["output"][r][c]:
                        print(f"  ({r},{c}): got {result[r][c]}, expected {ex['output'][r][c]}")
