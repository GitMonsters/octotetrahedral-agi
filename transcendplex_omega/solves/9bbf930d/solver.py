"""
Solver for ARC task 9bbf930d.

The grid has colored shapes enclosed by a boundary of 6-cells on column 0.
"Gap rows" (rows with fewer colored cells than neighbors) represent openings.
The solver:
  1. Detects gap rows (boundary openings)
  2. Opens gaps (changes col 0 from 6 to 7)
  3. Flood-fills from gap entries to find the "outside" region
  4. Places walls (6-cells) to seal the boundary — one per gap
  5. Walls are placed via corridor tracing toward grid boundaries,
     with depth-based zone tracking for boundary row walls.
"""
import json
from collections import deque


def solve(grid):
    R, C = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    gap_rows = _find_gap_rows(grid, R, C)
    gap_set = set(gap_rows)

    for r in gap_rows:
        out[r][0] = 7

    entries = [(r, 0) for r in gap_rows]
    flood = _do_flood(out, entries, R, C)

    walls = set()
    used_cols = set()

    def is_colored(r, c):
        return 0 <= r < R and 0 <= c < C and grid[r][c] not in (6, 7)

    def corridor_depth(start_r, col, direction):
        r, depth = start_r, 0
        while 0 <= r < R:
            if grid[r][col] != 7:
                break
            depth += 1
            r += direction
        return depth

    def trace_to_boundary(g, col, direction):
        r = g + direction
        while 0 <= r < R:
            if out[r][col] != 7 or (r, col) not in flood or (r, col) in walls:
                return None
            if r == 0 or r == R - 1:
                return (r, col)
            if col == C - 1:
                return (r, col)
            nr = r + direction
            if 0 <= nr < R and is_colored(nr, col):
                return (r, col)
            r += direction
        return None

    def find_corridor_end(g):
        for c in range(2, C):
            if grid[g][c] not in (6, 7):
                return c
        return C

    def mark_zone(wall_col, boundary_row, direction):
        d = corridor_depth(boundary_row, wall_col, direction)
        int_row = boundary_row + 2 * direction
        if not (0 <= int_row < R):
            used_cols.add(wall_col + 1)
            return
        group_end = wall_col
        while group_end + 1 < C and grid[int_row][group_end + 1] == 7:
            group_end += 1
        for c2 in range(wall_col + 1, C):
            d2 = corridor_depth(boundary_row, c2, direction)
            in_group = c2 <= group_end
            is_sep = c2 == group_end + 1 and c2 < C and grid[int_row][c2] not in (6, 7)
            if d2 <= d and (in_group or is_sep):
                used_cols.add(c2)
            else:
                break

    def try_endpoint(g, direction):
        c_end = find_corridor_end(g)
        if c_end >= C:
            return (g, C - 1)
        w = trace_to_boundary(g, c_end - 1, direction)
        if w and w[0] not in gap_set and w not in walls:
            return w
        return None

    def scan_corridor(g, direction):
        c_end = find_corridor_end(g)
        for col in range(2, c_end):
            if col in used_cols:
                continue
            if (g, col) not in flood:
                continue
            w = trace_to_boundary(g, col, direction)
            if w and w[0] not in gap_set and w not in walls:
                if w[0] == 0 or w[0] == R - 1 or w[1] == C - 1:
                    return w
        return None

    def try_secondary(g):
        c_end = find_corridor_end(g)
        sections = []
        seg_start = None
        for c in range(c_end + 1, C):
            if (g, c) in flood and out[g][c] == 7:
                if seg_start is None:
                    seg_start = c
            else:
                if seg_start is not None:
                    sections.append((seg_start, c - 1))
                    seg_start = None
        if seg_start is not None:
            sections.append((seg_start, C - 1))

        for seg_s, seg_e in reversed(sections):
            width = seg_e - seg_s + 1
            if seg_e == C - 1:
                for d in [+1, -1]:
                    w = trace_to_boundary(g, C - 1, d)
                    if w and w[0] not in gap_set and w not in walls:
                        return w
            if width > 1 and is_colored(g, seg_s - 1):
                return (g, seg_s)
            for col in [seg_s, seg_e]:
                for d in [+1, -1]:
                    w = trace_to_boundary(g, col, d)
                    if w and w[0] not in gap_set and w not in walls:
                        return w
        return None

    def place_wall(w, direction=None):
        walls.add(w)
        out[w[0]][w[1]] = 6
        if direction and (w[0] == 0 or w[0] == R - 1):
            mark_zone(w[1], w[0], direction)

    mid = R / 2
    up_gaps = sorted([g for g in gap_rows if g < mid], reverse=True)
    down_gaps = sorted([g for g in gap_rows if g >= mid])

    for g in up_gaps:
        w = try_endpoint(g, -1)
        if w and (w[0] == 0 or w[0] == R - 1 or w[1] == C - 1):
            place_wall(w, +1)
            continue
        w2 = scan_corridor(g, -1)
        if w2:
            place_wall(w2, +1)
            continue
        w3 = try_endpoint(g, +1)
        if w3:
            place_wall(w3)
            continue
        if w:
            place_wall(w, +1)
            continue
        w4 = try_secondary(g)
        if w4:
            place_wall(w4)

    for g in down_gaps:
        w = try_endpoint(g, +1)
        if w and (w[0] == 0 or w[0] == R - 1 or w[1] == C - 1):
            place_wall(w, -1)
            continue
        w2 = scan_corridor(g, +1)
        if w2:
            place_wall(w2, -1)
            continue
        w3 = try_endpoint(g, -1)
        if w3:
            place_wall(w3)
            continue
        if w:
            place_wall(w, -1)
            continue
        w4 = try_secondary(g)
        if w4:
            place_wall(w4)

    return out


def _find_gap_rows(grid, R, C):
    gap_rows = []
    for r in range(1, R - 1):
        if grid[r][0] != 6:
            continue
        total_here = sum(1 for c in range(C) if grid[r][c] not in (6, 7))
        total_above = sum(1 for c in range(C) if grid[r - 1][c] not in (6, 7))
        total_below = sum(1 for c in range(C) if grid[r + 1][c] not in (6, 7))
        if total_here >= total_above or total_here >= total_below:
            continue
        for c_val in range(10):
            if c_val in (6, 7):
                continue
            w_a = sum(1 for c in range(C) if grid[r - 1][c] == c_val)
            w_h = sum(1 for c in range(C) if grid[r][c] == c_val)
            w_b = sum(1 for c in range(C) if grid[r + 1][c] == c_val)
            if w_a > w_h and w_b > w_h:
                if w_h == 0 and total_here > 0:
                    continue
                gap_rows.append(r)
                break
    return gap_rows


def _do_flood(grid, entries, R, C):
    visited = set()
    q = deque()
    for r, c in entries:
        if 0 <= r < R and 0 <= c < C and grid[r][c] == 7 and (r, c) not in visited:
            visited.add((r, c))
            q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if (
                0 <= nr < R
                and 0 <= nc < C
                and (nr, nc) not in visited
                and grid[nr][nc] == 7
            ):
                visited.add((nr, nc))
                q.append((nr, nc))
    return visited


if __name__ == "__main__":
    with open("/Users/evanpieser/arc-puzzle-catalog/dataset/tasks/9bbf930d.json") as f:
        data = json.load(f)

    all_pass = True
    for i, ex in enumerate(data["train"]):
        result = solve(ex["input"])
        expected = ex["output"]
        if result == expected:
            print(f"Train {i}: PASS")
        else:
            R, C = len(expected), len(expected[0])
            diffs = sum(
                1 for r in range(R) for c in range(C) if result[r][c] != expected[r][c]
            )
            print(f"Train {i}: FAIL ({diffs} mismatches)")
            all_pass = False

    for i, ex in enumerate(data["test"]):
        if "output" in ex:
            result = solve(ex["input"])
            expected = ex["output"]
            if result == expected:
                print(f"Test  {i}: PASS")
            else:
                R, C = len(expected), len(expected[0])
                diffs = sum(
                    1
                    for r in range(R)
                    for c in range(C)
                    if result[r][c] != expected[r][c]
                )
                print(f"Test  {i}: FAIL ({diffs} mismatches)")
                all_pass = False
        else:
            result = solve(ex["input"])
            print(f"Test  {i}: produced {len(result)}x{len(result[0])} grid")

    print(f"\n{'ALL PASS' if all_pass else 'SOME FAILURES'}")
