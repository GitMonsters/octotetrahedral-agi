"""
ARC Task 4e469f39 solver.

Rule:
  Each 5-bordered rectangular frame has exactly one missing cell (gap) in its
  perimeter.  The transformation:
    1. Fills the frame interior (all 0s inside the bounding box, including the
       gap cell) with 2s.
    2. Shoots a ray of 2s from the gap, in the row/col just outside the gap
       side, extending to the grid boundary in the direction of the *longer*
       arm of the gap (the side of the top/bottom/left/right wall that has
       more 5s past the gap).
"""
from __future__ import annotations
from collections import deque
from typing import List


Grid = List[List[int]]


def solve(grid: Grid) -> Grid:
    H, W = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    visited = [[False] * W for _ in range(H)]

    for sr in range(H):
        for sc in range(W):
            if grid[sr][sc] != 5 or visited[sr][sc]:
                continue

            # BFS: collect connected component of 5s
            comp: list[tuple[int, int]] = []
            q: deque[tuple[int, int]] = deque([(sr, sc)])
            visited[sr][sc] = True
            while q:
                r, c = q.popleft()
                comp.append((r, c))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] == 5 and not visited[nr][nc]:
                        visited[nr][nc] = True
                        q.append((nr, nc))

            comp_set = set(comp)
            rs = [r for r, _ in comp]
            cs = [c for _, c in comp]
            min_r, max_r = min(rs), max(rs)
            min_c, max_c = min(cs), max(cs)

            # Find the single gap: first perimeter cell not in comp_set
            gap: tuple[int, int] | None = None
            gap_side: str | None = None
            for c in range(min_c, max_c + 1):
                if (min_r, c) not in comp_set:
                    gap, gap_side = (min_r, c), "top"
                    break
            if gap is None:
                for c in range(min_c, max_c + 1):
                    if (max_r, c) not in comp_set:
                        gap, gap_side = (max_r, c), "bottom"
                        break
            if gap is None:
                for r in range(min_r + 1, max_r):
                    if (r, min_c) not in comp_set:
                        gap, gap_side = (r, min_c), "left"
                        break
            if gap is None:
                for r in range(min_r + 1, max_r):
                    if (r, max_c) not in comp_set:
                        gap, gap_side = (r, max_c), "right"
                        break
            if gap is None:
                continue  # perfect rectangle — no gap, skip

            gap_r, gap_c = gap

            # 1. Fill all 0s inside the bounding box (interior + gap) with 2
            for r in range(min_r, max_r + 1):
                for c in range(min_c, max_c + 1):
                    if out[r][c] == 0:
                        out[r][c] = 2

            # 2. Shoot ray of 2s from just outside the gap side
            if gap_side == "top":
                la = sum(1 for c in range(min_c, gap_c) if (min_r, c) in comp_set)
                ra = sum(1 for c in range(gap_c + 1, max_c + 1) if (min_r, c) in comp_set)
                rr = min_r - 1
                if 0 <= rr:
                    for c in (range(0, gap_c + 1) if la >= ra else range(gap_c, W)):
                        out[rr][c] = 2

            elif gap_side == "bottom":
                la = sum(1 for c in range(min_c, gap_c) if (max_r, c) in comp_set)
                ra = sum(1 for c in range(gap_c + 1, max_c + 1) if (max_r, c) in comp_set)
                rr = max_r + 1
                if rr < H:
                    for c in (range(0, gap_c + 1) if la >= ra else range(gap_c, W)):
                        out[rr][c] = 2

            elif gap_side == "left":
                ta = sum(1 for r in range(min_r, gap_r) if (r, min_c) in comp_set)
                ba = sum(1 for r in range(gap_r + 1, max_r + 1) if (r, min_c) in comp_set)
                rc = min_c - 1
                if 0 <= rc:
                    for r in (range(0, gap_r + 1) if ta >= ba else range(gap_r, H)):
                        out[r][rc] = 2

            elif gap_side == "right":
                ta = sum(1 for r in range(min_r, gap_r) if (r, max_c) in comp_set)
                ba = sum(1 for r in range(gap_r + 1, max_r + 1) if (r, max_c) in comp_set)
                rc = max_c + 1
                if rc < W:
                    for r in (range(0, gap_r + 1) if ta >= ba else range(gap_r, H)):
                        out[r][rc] = 2

    return out
