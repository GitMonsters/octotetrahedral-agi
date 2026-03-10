"""
Task 9c1e755f: L-shaped pattern tiling.

Lines of 5s act as rulers. Adjacent rectangular patterns of colored cells
are tiled along the ruler to fill the rectangle defined by the ruler's extent.
"""
import json


def solve(grid: list[list[int]]) -> list[list[int]]:
    rows = len(grid)
    cols = len(grid[0])
    result = [row[:] for row in grid]

    fives = set()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 5:
                fives.add((r, c))

    # Find connected components of non-zero, non-5 cells (the patterns)
    non_zero = {
        (r, c)
        for r in range(rows)
        for c in range(cols)
        if grid[r][c] not in (0, 5)
    }

    visited: set[tuple[int, int]] = set()
    components: list[list[tuple[int, int]]] = []
    for cell in non_zero:
        if cell not in visited:
            comp: list[tuple[int, int]] = []
            queue = [cell]
            visited.add(cell)
            while queue:
                cr, cc = queue.pop(0)
                comp.append((cr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if (nr, nc) in non_zero and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        queue.append((nr, nc))
            components.append(comp)

    def find_vert_run(col: int, touch_rmin: int, touch_rmax: int):
        """Find continuous vertical run of 5s in col overlapping [touch_rmin..touch_rmax]."""
        start = None
        for r in range(touch_rmin, touch_rmax + 1):
            if (r, col) in fives:
                start = r
                break
        if start is None:
            return None
        rs = start
        while rs > 0 and (rs - 1, col) in fives:
            rs -= 1
        re = start
        while re < rows - 1 and (re + 1, col) in fives:
            re += 1
        if re - rs < 1:
            return None
        return (rs, re)

    def find_horiz_run(row: int, touch_cmin: int, touch_cmax: int):
        """Find continuous horizontal run of 5s in row overlapping [touch_cmin..touch_cmax]."""
        start = None
        for c in range(touch_cmin, touch_cmax + 1):
            if (row, c) in fives:
                start = c
                break
        if start is None:
            return None
        cs = start
        while cs > 0 and (row, cs - 1) in fives:
            cs -= 1
        ce = start
        while ce < cols - 1 and (row, ce + 1) in fives:
            ce += 1
        if ce - cs < 1:
            return None
        return (cs, ce)

    for comp in components:
        min_r = min(r for r, c in comp)
        max_r = max(r for r, c in comp)
        min_c = min(c for r, c in comp)
        max_c = max(c for r, c in comp)

        p_rows = max_r - min_r + 1
        p_cols = max_c - min_c + 1

        pattern = [[0] * p_cols for _ in range(p_rows)]
        for r, c in comp:
            pattern[r - min_r][c - min_c] = grid[r][c]

        filled = False

        # Check left side for vertical 5-line
        if not filled and min_c > 0:
            run = find_vert_run(min_c - 1, min_r, max_r)
            if run:
                rs, re = run
                for r in range(rs, re + 1):
                    for c in range(min_c, max_c + 1):
                        result[r][c] = pattern[(r - rs) % p_rows][c - min_c]
                filled = True

        # Check right side
        if not filled and max_c < cols - 1:
            run = find_vert_run(max_c + 1, min_r, max_r)
            if run:
                rs, re = run
                for r in range(rs, re + 1):
                    for c in range(min_c, max_c + 1):
                        result[r][c] = pattern[(r - rs) % p_rows][c - min_c]
                filled = True

        # Check top for horizontal 5-line
        if not filled and min_r > 0:
            run = find_horiz_run(min_r - 1, min_c, max_c)
            if run:
                cs, ce = run
                for r in range(min_r, max_r + 1):
                    for c in range(cs, ce + 1):
                        result[r][c] = pattern[r - min_r][(c - cs) % p_cols]
                filled = True

        # Check bottom
        if not filled and max_r < rows - 1:
            run = find_horiz_run(max_r + 1, min_c, max_c)
            if run:
                cs, ce = run
                for r in range(min_r, max_r + 1):
                    for c in range(cs, ce + 1):
                        result[r][c] = pattern[r - min_r][(c - cs) % p_cols]
                filled = True

    return result


if __name__ == "__main__":
    with open("/Users/evanpieser/arc-puzzle-catalog/dataset/tasks/9c1e755f.json") as f:
        task = json.load(f)

    for split in ("train", "test"):
        for i, ex in enumerate(task[split]):
            output = solve(ex["input"])
            status = "PASS" if output == ex["output"] else "FAIL"
            print(f"{split} {i}: {status}")
            if status == "FAIL":
                for r, (got, exp) in enumerate(zip(output, ex["output"])):
                    if got != exp:
                        print(f"  row {r}: got {got}, expected {exp}")
