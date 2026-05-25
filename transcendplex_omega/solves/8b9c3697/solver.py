"""
Puzzle 8b9c3697: Extending short teeth of comb-like structures.

Each structure has a bar (spine) with perpendicular teeth of varying lengths.
Short teeth aligned with red dots/blocks get extended:
- RED junction fills the "missing" tooth portion
- BLACK (0) bar extends from junction to the red dot
All unaligned red cells are removed (set to background).
"""
from collections import Counter, deque
import json
import sys


def solve(grid: list[list[int]]) -> list[list[int]]:
    rows, cols = len(grid), len(grid[0])
    bg = grid[0][0]
    output = [row[:] for row in grid]

    # Collect structural cells (not bg, not 0, not 2) and red cells
    struct_by_color: dict[int, set[tuple[int, int]]] = {}
    red_cells: set[tuple[int, int]] = set()
    for r in range(rows):
        for c in range(cols):
            v = grid[r][c]
            if v == 2:
                red_cells.add((r, c))
            elif v != bg and v != 0:
                struct_by_color.setdefault(v, set()).add((r, c))

    used_red: set[tuple[int, int]] = set()

    def process_horizontal_bar(comp: set[tuple[int, int]]) -> None:
        """Process a structure with a horizontal bar and vertical teeth."""
        rc = Counter(r for r, c in comp)
        bar_row = max(rc, key=rc.get)
        bar_cols = sorted(c for r, c in comp if r == bar_row)

        teeth: dict[int, int] = {}
        tooth_dir = None
        for tc in range(bar_cols[0], bar_cols[-1] + 1):
            t_rows = [r for r, c in comp if c == tc and r != bar_row]
            if t_rows:
                teeth[tc] = len(t_rows)
                if tooth_dir is None:
                    tooth_dir = -1 if min(t_rows) < bar_row else 1

        if not teeth or tooth_dir is None:
            return
        max_len = max(teeth.values())
        short_cols = sorted(tc for tc in teeth if teeth[tc] < max_len)
        if not short_cols:
            return

        # Group adjacent short columns
        groups: list[list[int]] = [[short_cols[0]]]
        for i in range(1, len(short_cols)):
            if short_cols[i] == short_cols[i - 1] + 1:
                groups[-1].append(short_cols[i])
            else:
                groups.append([short_cols[i]])

        for group in groups:
            short_len = teeth[group[0]]
            tip_row = bar_row + tooth_dir * short_len

            # Search outward for aligned red block
            search_pos = tip_row + tooth_dir
            found = None
            while 0 <= search_pos < rows:
                vals = [grid[search_pos][gc] for gc in group]
                if all(v == 2 for v in vals):
                    red_near = search_pos
                    red_far = search_pos
                    while 0 <= red_far + tooth_dir < rows:
                        nxt = red_far + tooth_dir
                        if all(grid[nxt][gc] == 2 for gc in group):
                            red_far = nxt
                        else:
                            break
                    found = (red_near, red_far)
                    break
                elif any(v != bg for v in vals):
                    break
                search_pos += tooth_dir

            if found:
                red_near, red_far = found
                red_len = abs(red_far - red_near) + 1

                # RED junction
                for step in range(1, red_len + 1):
                    jr = tip_row + tooth_dir * step
                    for gc in group:
                        output[jr][gc] = 2

                # BLACK bar to red block
                pos = tip_row + tooth_dir * (red_len + 1)
                while True:
                    for gc in group:
                        output[pos][gc] = 0
                    if pos == red_far:
                        break
                    pos += tooth_dir

                # Mark red cells as used
                r1, r2 = min(red_near, red_far), max(red_near, red_far)
                for r in range(r1, r2 + 1):
                    for gc in group:
                        used_red.add((r, gc))

    def process_vertical_bar(comp: set[tuple[int, int]]) -> None:
        """Process a structure with a vertical bar and horizontal teeth."""
        cc = Counter(c for r, c in comp)
        bar_col = max(cc, key=cc.get)
        bar_rows = sorted(r for r, c in comp if c == bar_col)

        teeth: dict[int, int] = {}
        tooth_dir = None
        for tr in range(bar_rows[0], bar_rows[-1] + 1):
            t_cols = [c for r, c in comp if r == tr and c != bar_col]
            if t_cols:
                teeth[tr] = len(t_cols)
                if tooth_dir is None:
                    tooth_dir = -1 if min(t_cols) < bar_col else 1

        if not teeth or tooth_dir is None:
            return
        max_len = max(teeth.values())
        short_rows = sorted(tr for tr in teeth if teeth[tr] < max_len)
        if not short_rows:
            return

        groups: list[list[int]] = [[short_rows[0]]]
        for i in range(1, len(short_rows)):
            if short_rows[i] == short_rows[i - 1] + 1:
                groups[-1].append(short_rows[i])
            else:
                groups.append([short_rows[i]])

        for group in groups:
            short_len = teeth[group[0]]
            tip_col = bar_col + tooth_dir * short_len

            search_pos = tip_col + tooth_dir
            found = None
            while 0 <= search_pos < cols:
                vals = [grid[gr][search_pos] for gr in group]
                if all(v == 2 for v in vals):
                    red_near = search_pos
                    red_far = search_pos
                    while 0 <= red_far + tooth_dir < cols:
                        nxt = red_far + tooth_dir
                        if all(grid[gr][nxt] == 2 for gr in group):
                            red_far = nxt
                        else:
                            break
                    found = (red_near, red_far)
                    break
                elif any(v != bg for v in vals):
                    break
                search_pos += tooth_dir

            if found:
                red_near, red_far = found
                red_len = abs(red_far - red_near) + 1

                for step in range(1, red_len + 1):
                    jc = tip_col + tooth_dir * step
                    for gr in group:
                        output[gr][jc] = 2

                pos = tip_col + tooth_dir * (red_len + 1)
                while True:
                    for gr in group:
                        output[gr][pos] = 0
                    if pos == red_far:
                        break
                    pos += tooth_dir

                c1, c2 = min(red_near, red_far), max(red_near, red_far)
                for c in range(c1, c2 + 1):
                    for gr in group:
                        used_red.add((gr, c))

    for sc, cells in struct_by_color.items():
        visited: set[tuple[int, int]] = set()
        for start in cells:
            if start in visited:
                continue
            comp: set[tuple[int, int]] = set()
            q = deque([start])
            visited.add(start)
            while q:
                r, c = q.popleft()
                comp.add((r, c))
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if (nr, nc) in cells and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        q.append((nr, nc))

            rc = Counter(r for r, c in comp)
            cc = Counter(c for r, c in comp)
            if max(rc.values()) >= max(cc.values()):
                process_horizontal_bar(comp)
            else:
                process_vertical_bar(comp)

    # Remove unused red cells
    for r, c in red_cells:
        if (r, c) not in used_red:
            output[r][c] = bg

    return output


if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        data = json.load(f)

    for i, ex in enumerate(data["train"]):
        result = solve(ex["input"])
        if result == ex["output"]:
            print(f"Train {i}: PASS")
        else:
            print(f"Train {i}: FAIL")
            for r in range(len(result)):
                for c in range(len(result[0])):
                    if result[r][c] != ex["output"][r][c]:
                        print(f"  ({r},{c}): got {result[r][c]}, expected {ex['output'][r][c]}")
