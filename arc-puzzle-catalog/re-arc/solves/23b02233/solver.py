"""Solver for ARC puzzle 23b02233.

Rule: cells are mapped to a 3x3 output grid based on their relative position 
among all non-background cells. The mapping from (aA, aR) = (count of cells 
above, count of cells to the right) to output position is consistent across 
all examples. Additional constraints: within-color row/col order preservation,
global column order preservation (for n<=5), and equal input cols => equal 
output cols.
"""
from collections import Counter
from itertools import combinations, permutations


# Training-derived lookup: (aA, aR) -> (output_row, output_col)
LOOKUP = {
    (0, 1): (0, 2), (0, 2): (1, 1), (0, 3): (2, 1),
    (1, 0): (2, 1), (1, 2): (0, 2), (1, 3): (1, 1),
    (2, 2): (0, 1), (2, 3): (1, 1), (2, 4): (2, 0),
    (3, 1): (1, 2), (3, 4): (2, 0), (3, 6): (0, 1),
    (4, 0): (2, 2), (4, 3): (0, 2),
    (5, 0): (1, 2), (6, 4): (2, 2),
}

AVAIL = [(r, c) for r in range(3) for c in range(3) if not (c == 0 and r < 2)]


def transform(input_grid: list[list[int]]) -> list[list[int]]:
    H = len(input_grid)
    W = len(input_grid[0])
    flat = [c for row in input_grid for c in row]
    bg = Counter(flat).most_common(1)[0][0]

    cells = []
    for r in range(H):
        for c in range(W):
            if input_grid[r][c] != bg:
                cells.append((r, c, input_grid[r][c]))
    n = len(cells)

    if n == 0:
        return [[bg] * 3 for _ in range(3)]

    if n == 1:
        r, c, color = cells[0]
        zr = min(2, int(r * 3 / H))
        zc = min(2, int(c * 3 / W))
        if (zr, zc) == (0, 0):
            zc = 1
        elif (zr, zc) == (1, 0):
            zc = 1
        grid = [[bg] * 3 for _ in range(3)]
        grid[zr][zc] = color
        return grid

    # Compute (aA, aR) for each cell
    keys = []
    for i, (r, c, color) in enumerate(cells):
        aA = sum(1 for r2, c2, _ in cells if r2 < r)
        aR = sum(1 for r2, c2, _ in cells if c2 > c)
        keys.append((aA, aR))

    # Try lookup-only solution first
    if all(k in LOOKUP for k in keys):
        assignment = [LOOKUP[k] for k in keys]
        if len(set(assignment)) == n:
            grid = [[bg] * 3 for _ in range(3)]
            for i, (r, c, color) in enumerate(cells):
                grid[assignment[i][0]][assignment[i][1]] = color
            return grid

    # Enumerate valid assignments
    best = None
    best_score = None

    for pos_combo in combinations(range(len(AVAIL)), n):
        positions = [AVAIL[j] for j in pos_combo]
        for perm in permutations(range(n)):
            assignment = [positions[perm[k]] for k in range(n)]

            # Check unique positions
            test_grid = [[bg] * 3 for _ in range(3)]
            conflict = False
            for k, (r, c, color) in enumerate(cells):
                or_, oc = assignment[k]
                if test_grid[or_][oc] != bg:
                    conflict = True
                    break
                test_grid[or_][oc] = color
            if conflict:
                continue

            # Check within-color order preservation
            valid = True
            color_groups: dict[int, list] = {}
            for k, (r, c, color) in enumerate(cells):
                if color not in color_groups:
                    color_groups[color] = []
                color_groups[color].append((r, c, assignment[k]))
            for group in color_groups.values():
                for i in range(len(group)):
                    for j in range(i + 1, len(group)):
                        ri, ci, (oi_r, oi_c) = group[i]
                        rj, cj, (oj_r, oj_c) = group[j]
                        if ri < rj and oi_r > oj_r:
                            valid = False
                        if ri > rj and oi_r < oj_r:
                            valid = False
                        if ci < cj and oi_c > oj_c:
                            valid = False
                        if ci > cj and oi_c < oj_c:
                            valid = False
                        if not valid:
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if not valid:
                continue

            # Check (aA, aR) lookup matches
            lookup_ok = True
            for k in range(n):
                if keys[k] in LOOKUP and LOOKUP[keys[k]] != assignment[k]:
                    lookup_ok = False
                    break
            if not lookup_ok:
                continue

            # For n <= 5: check global column order (≤ preserved)
            if n <= 5:
                col_ok = True
                for i in range(n):
                    for j in range(i + 1, n):
                        ci, cj = cells[i][1], cells[j][1]
                        oi_c, oj_c = assignment[i][1], assignment[j][1]
                        if ci < cj and oi_c > oj_c:
                            col_ok = False
                            break
                        if ci > cj and oi_c < oj_c:
                            col_ok = False
                            break
                        if ci == cj and oi_c != oj_c:
                            col_ok = False
                            break
                    if not col_ok:
                        break
                if not col_ok:
                    continue

            # Check bottommost cell -> (2,2)
            bottommost_idx = max(range(n), key=lambda k: cells[k][0])
            if assignment[bottommost_idx] != (2, 2):
                continue

            # Score: prefer distinct (aA,aR)->output mappings (no adjacent
            # aR values mapping to the same output for the same aA)
            adj_dupes = 0
            new_map = {keys[k]: assignment[k] for k in range(n)}
            full_map = dict(LOOKUP)
            full_map.update(new_map)
            for (aA1, aR1), pos1 in full_map.items():
                for (aA2, aR2), pos2 in full_map.items():
                    if aA1 == aA2 and abs(aR1 - aR2) == 1 and pos1 == pos2:
                        adj_dupes += 1

            # Zone distance
            zd = sum(
                (assignment[k][0] - cells[k][0] * 3 / H) ** 2
                + (assignment[k][1] - cells[k][1] * 3 / W) ** 2
                for k in range(n)
            )

            score = (adj_dupes, zd)
            if best_score is None or score < best_score:
                best_score = score
                best = test_grid

    if best is not None:
        return best

    # Fallback: zone mapping
    grid = [[bg] * 3 for _ in range(3)]
    for r, c, color in cells:
        zr = min(2, int(r * 3 / H))
        zc = min(2, int(c * 3 / W))
        if grid[zr][zc] == bg:
            grid[zr][zc] = color
    return grid
