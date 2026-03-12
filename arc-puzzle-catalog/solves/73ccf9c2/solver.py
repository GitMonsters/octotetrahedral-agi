"""
Solver for ARC-AGI puzzle 73ccf9c2.

Rule: The input grid contains several clusters of non-zero cells separated by
empty space. Each cluster's bounding box is extracted. Exactly one cluster
lacks horizontal (left-right) mirror symmetry — that cluster IS the output.
"""

from typing import List

Grid = List[List[int]]


def solve(grid: Grid) -> Grid:
    groups = _find_groups(grid, gap=2)
    for group in groups:
        bbox = _extract_bbox(grid, group)
        if not _is_h_symmetric(bbox):
            return bbox
    # Fallback: return largest group
    groups.sort(key=len, reverse=True)
    return _extract_bbox(grid, groups[0])


def _find_groups(grid: Grid, gap: int) -> List[List[tuple]]:
    H, W = len(grid), len(grid[0])
    nz = [(r, c) for r in range(H) for c in range(W) if grid[r][c] != 0]
    if not nz:
        return []
    visited = [False] * len(nz)
    groups: List[List[tuple]] = []
    for i in range(len(nz)):
        if visited[i]:
            continue
        group = [nz[i]]
        visited[i] = True
        queue = [i]
        while queue:
            ci = queue.pop(0)
            cr, cc = nz[ci]
            for j in range(len(nz)):
                if not visited[j]:
                    jr, jc = nz[j]
                    if abs(cr - jr) <= gap and abs(cc - jc) <= gap:
                        visited[j] = True
                        group.append(nz[j])
                        queue.append(j)
        groups.append(group)
    return groups


def _extract_bbox(grid: Grid, group: List[tuple]) -> Grid:
    rs = [r for r, c in group]
    cs = [c for r, c in group]
    min_r, max_r = min(rs), max(rs)
    min_c, max_c = min(cs), max(cs)
    bh, bw = max_r - min_r + 1, max_c - min_c + 1
    bbox = [[0] * bw for _ in range(bh)]
    for r, c in group:
        bbox[r - min_r][c - min_c] = grid[r][c]
    return bbox


def _is_h_symmetric(bbox: Grid) -> bool:
    return all(row == row[::-1] for row in bbox)
