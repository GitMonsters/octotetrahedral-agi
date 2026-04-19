"""ARC puzzle f8f52ecc solver.

Rule: For each non-background (1) non-barrier (8) color with 2+ cells,
sort cells row-major and connect consecutive dots via L-shaped paths.
L-shape prefers vertical-first unless it conflicts with other colors;
when both are conflict-free, pick the path adding fewer new cells
(defaulting to vertical-first on ties). Color 8 acts as a barrier.
"""
import copy
from typing import List


def transform(input_grid: List[List[int]]) -> List[List[int]]:
    grid = copy.deepcopy(input_grid)
    rows, cols = len(grid), len(grid[0])
    bg, barrier = 1, 8

    # Collect positions per color (excluding background and barrier)
    color_cells: dict[int, list[tuple[int, int]]] = {}
    for r in range(rows):
        for c in range(cols):
            v = grid[r][c]
            if v != bg and v != barrier:
                color_cells.setdefault(v, []).append((r, c))

    for color, cells in color_cells.items():
        if len(cells) < 2:
            continue
        cells.sort()  # row-major order

        for i in range(len(cells) - 1):
            r1, c1 = cells[i]
            r2, c2 = cells[i + 1]

            # Vertical-first: move to target row at source col, then to target col
            def make_path_v(r1=r1, c1=c1, r2=r2, c2=c2):
                p = []
                sr = 1 if r2 >= r1 else -1
                for r in range(r1, r2 + sr, sr):
                    p.append((r, c1))
                sc = 1 if c2 >= c1 else -1
                for c in range(c1 + sc, c2 + sc, sc):
                    p.append((r2, c))
                return p

            # Horizontal-first: move to target col at source row, then to target row
            def make_path_h(r1=r1, c1=c1, r2=r2, c2=c2):
                p = []
                sc = 1 if c2 >= c1 else -1
                for c in range(c1, c2 + sc, sc):
                    p.append((r1, c))
                sr = 1 if r2 >= r1 else -1
                for r in range(r1 + sr, r2 + sr, sr):
                    p.append((r, c2))
                return p

            def evaluate(p):
                conflicts = new = 0
                for r, c in p:
                    v = grid[r][c]
                    if v != bg and v != color:
                        conflicts += 1
                    if v != color:
                        new += 1
                return conflicts, new

            pv, ph = make_path_v(), make_path_h()
            cv, nv = evaluate(pv)
            ch, nh = evaluate(ph)

            if cv == 0 and ch == 0:
                chosen = pv if nv <= nh else ph
            elif cv == 0:
                chosen = pv
            elif ch == 0:
                chosen = ph
            else:
                chosen = pv if (cv < ch or (cv == ch and nv <= nh)) else ph

            for r, c in chosen:
                grid[r][c] = color

    return grid


if __name__ == "__main__":
    test_input = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 2, 1, 1, 2, 1, 1, 1, 4, 1],
        [1, 1, 1, 1, 1, 8, 1, 1, 1, 1],
        [1, 1, 1, 1, 2, 1, 2, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 1, 1, 8, 8],
        [1, 1, 5, 1, 5, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 7, 1],
        [1, 3, 1, 1, 1, 1, 1, 1, 1, 1],
    ]
    result = transform(test_input)
    for row in result:
        print(row)
