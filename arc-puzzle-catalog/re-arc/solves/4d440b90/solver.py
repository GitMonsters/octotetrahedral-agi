import copy
from collections import Counter


def get_cross_cells(r, c):
    """Return the 9 cells of a + shape centered at (r,c) with arms of length 2."""
    cells = set()
    for d in range(-2, 3):
        cells.add((r, c + d))  # horizontal arm
        cells.add((r + d, c))  # vertical arm
    return cells


def transform(input_grid):
    grid = input_grid
    R = len(grid)
    C = len(grid[0])

    counts = Counter()
    for row in grid:
        counts.update(row)

    n_colors = len(counts)
    marker = min(counts, key=counts.get)

    marker_pos = set()
    for r in range(R):
        for c in range(C):
            if grid[r][c] == marker:
                marker_pos.add((r, c))

    crosses = []

    if n_colors >= 3:
        # 3-color case: all marker cells define crosses
        # Valid cross: all non-marker cells in cross are the same color, ≥2 markers
        candidates = []
        for r in range(2, R - 2):
            for c in range(2, C - 2):
                cross = get_cross_cells(r, c)
                cross_markers = cross & marker_pos
                if len(cross_markers) < 2:
                    continue
                non_marker_vals = set()
                for cr, cc in cross:
                    if (cr, cc) not in marker_pos:
                        non_marker_vals.add(grid[cr][cc])
                if len(non_marker_vals) == 1:
                    candidates.append((r, c, len(cross_markers), cross_markers))

        # Greedy selection by marker count (most first)
        candidates.sort(key=lambda x: -x[2])
        remaining_markers = set(marker_pos)
        for r, c, mc, cm in candidates:
            current_markers = cm & remaining_markers
            if len(current_markers) >= 2:
                crosses.append((r, c))
                remaining_markers -= cm
    else:
        # 2-color case: need pattern-based detection
        candidates = []
        for r in range(2, R - 2):
            for c in range(2, C - 2):
                h_vals = [grid[r][c + d] for d in range(-2, 3)]
                v_vals = [grid[r + d][c] for d in range(-2, 3)]
                h_same = len(set(h_vals)) == 1
                v_same = len(set(v_vals)) == 1

                if not (h_same ^ v_same):  # exactly one must be uniform
                    continue

                if h_same:
                    uniform_color = h_vals[0]
                    ep = (grid[r - 2][c], grid[r + 2][c])
                else:
                    uniform_color = v_vals[0]
                    ep = (grid[r][c - 2], grid[r][c + 2])

                if ep[0] != ep[1] or ep[0] == uniform_color:
                    continue

                # Filter: no arm endpoint at grid edge
                cross = get_cross_cells(r, c)
                at_edge = False
                for cr, cc in cross:
                    if cr < 0 or cr >= R or cc < 0 or cc >= C:
                        at_edge = True
                        break
                    if cr == 0 or cr == R - 1 or cc == 0 or cc == C - 1:
                        # Check if this cell is an arm endpoint
                        if (cr == r - 2 or cr == r + 2 or cc == c - 2 or cc == c + 2):
                            at_edge = True
                            break
                if at_edge:
                    continue

                h_marker = sum(1 for d in range(-2, 3) if grid[r][c + d] == marker)
                v_marker = sum(1 for d in range(-2, 3) if grid[r + d][c] == marker)
                contrast = abs(h_marker - v_marker)

                # Check if perfect cross (one arm ALL marker, perpendicular non-center ALL non-marker)
                is_perfect = False
                if h_marker == 5:
                    if all(grid[r + d][c] != marker for d in [-2, -1, 1, 2]):
                        is_perfect = True
                if v_marker == 5:
                    if all(grid[r][c + d] != marker for d in [-2, -1, 1, 2]):
                        is_perfect = True

                # Distance from grid center (for tiebreaking)
                center_dist = ((r - (R - 1) / 2) ** 2 + (c - (C - 1) / 2) ** 2) ** 0.5

                candidates.append((r, c, is_perfect, contrast, center_dist))

        # Sort: perfect first, then by center distance (closest to grid center first)
        candidates.sort(key=lambda x: (not x[2], x[4]))

        used_cells = set()
        n_perfect = 0
        n_nonperfect = 0
        max_nonperfect = 1  # limit non-perfect crosses
        for r, c, is_perfect, contrast, cd in candidates:
            cross = get_cross_cells(r, c)
            if cross & used_cells:
                continue
            if is_perfect:
                n_perfect += 1
            else:
                if contrast < 3:
                    continue
                if n_nonperfect >= max_nonperfect:
                    continue
                n_nonperfect += 1
            crosses.append((r, c))
            used_cells |= cross

    # Apply crosses: fill non-marker cells with 2
    out = copy.deepcopy(grid)
    for cr, cc in crosses:
        cross = get_cross_cells(cr, cc)
        for r, c in cross:
            if 0 <= r < R and 0 <= c < C:
                if out[r][c] != marker:
                    out[r][c] = 2

    return out
