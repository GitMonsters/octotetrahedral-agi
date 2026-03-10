"""
Solver for ARC-AGI task 7b3084d4.

The input contains 4 colored shapes on a 20x20 grid. One shape has a single
cell of color 5 marking the anchor corner. The output assembles all 4 shapes
into a filled rectangle in a 2x2 jigsaw arrangement, with the 5-cell at (0,0).
Shapes interlock: holes in one shape are filled by protrusions from neighbors.
"""
import json
from itertools import permutations


def solve(grid: list[list[int]]) -> list[list[int]]:
    rows, cols = len(grid), len(grid[0])

    # Group non-zero cells by color; find the 5-cell
    color_cells: dict[int, list[tuple[int, int]]] = {}
    five_pos = None
    for r in range(rows):
        for c in range(cols):
            v = grid[r][c]
            if v == 0:
                continue
            if v == 5:
                five_pos = (r, c)
            else:
                color_cells.setdefault(v, []).append((r, c))

    # Determine which color the 5-cell belongs to (8-connected neighbor)
    five_color = None
    r5, c5 = five_pos
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r5 + dr, c5 + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] not in (0, 5):
                five_color = grid[nr][nc]
                break
        if five_color:
            break

    # Extract each shape as a local bounding-box grid
    shapes: dict[int, list[list[int]]] = {}
    for color, cells in color_cells.items():
        if color == five_color:
            cells = cells + [five_pos]
        min_r = min(r for r, _ in cells)
        max_r = max(r for r, _ in cells)
        min_c = min(c for _, c in cells)
        max_c = max(c for _, c in cells)
        h, w = max_r - min_r + 1, max_c - min_c + 1
        local = [[0] * w for _ in range(h)]
        for r, c in cells:
            local[r - min_r][c - min_c] = grid[r][c]
        shapes[color] = local

    # --- geometry helpers ---
    def rotate_cw(s):
        h, w = len(s), len(s[0])
        return [[s[h - 1 - j][i] for j in range(h)] for i in range(w)]

    def flip_h(s):
        return [row[::-1] for row in s]

    def orientations(s):
        out, seen = [], set()
        cur = s
        for _ in range(4):
            for variant in (cur, flip_h(cur)):
                key = tuple(tuple(row) for row in variant)
                if key not in seen:
                    seen.add(key)
                    out.append([list(row) for row in variant])
            cur = rotate_cw(cur)
        return out

    def nz(s):
        return [(i, j, s[i][j]) for i in range(len(s)) for j in range(len(s[0])) if s[i][j] != 0]

    # Orientations of the 5-shape that place 5 at (0,0)
    tl_candidates = [o for o in orientations(shapes[five_color]) if o[0][0] == 5]
    other_colors = [c for c in shapes if c != five_color]
    all_orients = {c: orientations(shapes[c]) for c in other_colors}

    for tl in tl_candidates:
        h_tl, w_tl = len(tl), len(tl[0])
        tl_nz = nz(tl)
        n_tl = len(tl_nz)

        for perm in permutations(other_colors):
            tr_c, bl_c, br_c = perm
            for tr in all_orients[tr_c]:
                h_tr, w_tr = len(tr), len(tr[0])
                tr_nz = nz(tr)
                n_tr = len(tr_nz)
                for bl in all_orients[bl_c]:
                    h_bl, w_bl = len(bl), len(bl[0])
                    bl_nz = nz(bl)
                    n_bl = len(bl_nz)
                    for br in all_orients[br_c]:
                        h_br, w_br = len(br), len(br[0])
                        br_nz = nz(br)
                        total = n_tl + n_tr + n_bl + len(br_nz)

                        for c_tr in range(1, w_tl + 1):
                            W = c_tr + w_tr
                            c_br = W - w_br
                            if c_br < 0:
                                continue
                            for r_bl in range(1, h_tl + 1):
                                H = r_bl + h_bl
                                r_br = H - h_br
                                if r_br < 0:
                                    continue
                                if total != H * W:
                                    continue

                                occ = set()
                                ok = True

                                for i, j, _ in tl_nz:
                                    occ.add((i, j))

                                for i, j, _ in tr_nz:
                                    p = (i, c_tr + j)
                                    if p[0] >= H or p[1] >= W or p in occ:
                                        ok = False
                                        break
                                    occ.add(p)
                                if not ok:
                                    continue

                                for i, j, _ in bl_nz:
                                    p = (r_bl + i, j)
                                    if p[0] >= H or p[1] >= W or p in occ:
                                        ok = False
                                        break
                                    occ.add(p)
                                if not ok:
                                    continue

                                for i, j, _ in br_nz:
                                    p = (r_br + i, c_br + j)
                                    if p[0] >= H or p[1] >= W or p in occ:
                                        ok = False
                                        break
                                    occ.add(p)
                                if not ok:
                                    continue

                                # Build output
                                canvas = [[0] * W for _ in range(H)]
                                for i, j, v in tl_nz:
                                    canvas[i][j] = v
                                for i, j, v in tr_nz:
                                    canvas[i][c_tr + j] = v
                                for i, j, v in bl_nz:
                                    canvas[r_bl + i][j] = v
                                for i, j, v in br_nz:
                                    canvas[r_br + i][c_br + j] = v
                                return canvas
    return None


if __name__ == "__main__":
    with open("/Users/evanpieser/arc-puzzle-catalog/dataset/tasks/7b3084d4.json") as f:
        task = json.load(f)

    for idx, pair in enumerate(task["train"]):
        result = solve(pair["input"])
        expected = pair["output"]
        status = "PASS" if result == expected else "FAIL"
        print(f"Train {idx}: {status}")
        if status == "FAIL":
            print(f"  Expected: {expected}")
            print(f"  Got:      {result}")

    for idx, pair in enumerate(task["test"]):
        result = solve(pair["input"])
        if "output" in pair:
            expected = pair["output"]
            status = "PASS" if result == expected else "FAIL"
            print(f"Test  {idx}: {status}")
            if status == "FAIL":
                print(f"  Expected: {expected}")
                print(f"  Got:      {result}")
        else:
            print(f"Test  {idx}: {result}")
