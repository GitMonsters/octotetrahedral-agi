"""
Solver for ARC-AGI task 89565ca0.

Pattern: The grid contains several colored rectangles (table-like grids with
borders and internal dividers) plus a noise color scattered randomly. Each
rectangle's grid lines divide it into cells. The output is a staircase where
each row represents a color, left-filled by its cell count, sorted ascending.
The noise color fills the remaining positions.

Algorithm:
1. Identify noise color (shortest max run length).
2. For each grid-forming color, detect horizontal/vertical grid lines via
   contiguous-run analysis, then count enclosed cells via flood fill.
3. Sort colors by cell count and build the staircase output.
"""
from collections import deque
from typing import List


def solve(grid: List[List[int]]) -> List[List[int]]:
    noise = _find_noise_color(grid)
    colors = {v for row in grid for v in row if v != 0}
    colors.discard(noise)

    results = []
    for c in colors:
        cells = _count_cells(grid, c)
        results.append((cells, c))
    results.sort()

    max_cells = max(cells for cells, _ in results)
    return [[c] * cells + [noise] * (max_cells - cells) for cells, c in results]


def _find_noise_color(grid: List[List[int]]) -> int:
    rows, cols = len(grid), len(grid[0])
    colors = {v for row in grid for v in row if v != 0}
    best, min_run = 0, float('inf')
    for c in colors:
        max_run = 0
        for r in range(rows):
            run = 0
            for co in range(cols):
                if grid[r][co] == c:
                    run += 1
                    max_run = max(max_run, run)
                else:
                    run = 0
        for co in range(cols):
            run = 0
            for r in range(rows):
                if grid[r][co] == c:
                    run += 1
                    max_run = max(max_run, run)
                else:
                    run = 0
        if max_run < min_run:
            min_run = max_run
            best = c
    return best


def _find_segments(positions: List[int], max_gap: int = 3,
                   min_cells: int = 3, min_consec: int = 2):
    """Group nearby positions; keep groups with enough cells and consecutive run."""
    if not positions:
        return []
    positions = sorted(set(positions))
    groups: list[list[int]] = [[positions[0]]]
    for p in positions[1:]:
        if p - groups[-1][-1] <= max_gap:
            groups[-1].append(p)
        else:
            groups.append([p])
    result = []
    for g in groups:
        if len(g) < min_cells:
            continue
        max_run = run = 1
        for i in range(1, len(g)):
            if g[i] == g[i - 1] + 1:
                run += 1
                max_run = max(max_run, run)
            else:
                run = 1
        if max_run >= min_consec:
            result.append((g[0], g[-1]))
    return result


def _count_cells(grid: List[List[int]], color: int) -> int:
    rows, cols = len(grid), len(grid[0])
    positions = [(r, co) for r in range(rows) for co in range(cols)
                 if grid[r][co] == color]
    if not positions:
        return 0

    min_r = min(r for r, _ in positions)
    max_r = max(r for r, _ in positions)
    min_c = min(c for _, c in positions)
    max_c = max(c for _, c in positions)

    # Detect horizontal grid lines
    h_extents: dict[int, tuple[int, int]] = {}
    for r in range(min_r, max_r + 1):
        c_cols = sorted(co for co in range(min_c, max_c + 1)
                        if grid[r][co] == color)
        for seg_s, seg_e in _find_segments(c_cols):
            if r not in h_extents:
                h_extents[r] = (seg_s, seg_e)
            else:
                s, e = h_extents[r]
                h_extents[r] = (min(s, seg_s), max(e, seg_e))
    h_rows = sorted(h_extents)
    if len(h_rows) < 2:
        return 1 if h_rows else 0

    # Detect vertical grid line candidates
    v_cand = set()
    for co in range(min_c, max_c + 1):
        c_rows = sorted(r for r in range(min_r, max_r + 1)
                        if grid[r][co] == color)
        if _find_segments(c_rows):
            v_cand.add(co)
    v_cols = sorted(v_cand)
    if len(v_cols) < 2:
        return 1 if v_cols else 0

    # Extend vertical line extents via band-spanning check
    v_extents: dict[int, tuple[int, int]] = {}
    for co in v_cols:
        first_h = last_h = None
        for i in range(len(h_rows) - 1):
            h_top, h_bot = h_rows[i], h_rows[i + 1]
            if any(grid[r][co] == color for r in range(h_top + 1, h_bot)):
                if first_h is None:
                    first_h = h_top
                last_h = h_bot
        if first_h is not None:
            v_extents[co] = (first_h, last_h)

    # Build wall mask with 1-cell open border for exterior flood fill
    h = max_r - min_r + 3
    w = max_c - min_c + 3
    mask = [[0] * w for _ in range(h)]

    for r in h_rows:
        c_s, c_e = h_extents[r]
        for co in range(c_s, c_e + 1):
            mr, mc = r - min_r + 1, co - min_c + 1
            if 0 <= mr < h and 0 <= mc < w:
                mask[mr][mc] = 1

    for co, (r_s, r_e) in v_extents.items():
        for r in range(r_s, r_e + 1):
            mr, mc = r - min_r + 1, co - min_c + 1
            if 0 <= mr < h and 0 <= mc < w:
                mask[mr][mc] = 1

    # Flood fill exterior from (0, 0)
    visited = [[False] * w for _ in range(h)]
    queue = deque([(0, 0)])
    visited[0][0] = True
    while queue:
        cr, cc = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w and not visited[nr][nc] and mask[nr][nc] == 0:
                visited[nr][nc] = True
                queue.append((nr, nc))

    # Count interior connected components
    cell_count = 0
    for r in range(h):
        for c in range(w):
            if mask[r][c] == 0 and not visited[r][c]:
                cell_count += 1
                queue = deque([(r, c)])
                visited[r][c] = True
                while queue:
                    cr, cc = queue.popleft()
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and not visited[nr][nc] and mask[nr][nc] == 0:
                            visited[nr][nc] = True
                            queue.append((nr, nc))
    return cell_count


if __name__ == "__main__":
    import json, os

    task_path = os.path.join(os.path.dirname(__file__),
                             "../../dataset/tasks/89565ca0.json")
    with open(task_path) as f:
        data = json.load(f)

    all_pass = True
    for split in ("train", "test"):
        for i, pair in enumerate(data.get(split, [])):
            result = solve(pair["input"])
            expected = pair["output"]
            ok = result == expected
            if not ok:
                all_pass = False
            print(f"{split}[{i}]: {'PASS' if ok else 'FAIL'}")
            if not ok:
                print(f"  expected: {expected}")
                print(f"  got:      {result}")

    print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
