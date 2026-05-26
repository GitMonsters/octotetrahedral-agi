from collections import defaultdict, Counter


def transform(grid):
    H, W = len(grid), len(grid[0])
    flat = [cell for row in grid for cell in row]
    bg = Counter(flat).most_common(1)[0][0]

    color_cells: dict = defaultdict(list)
    for r in range(H):
        for c in range(W):
            if grid[r][c] != bg:
                color_cells[grid[r][c]].append((r, c))

    def border_disps(cells, oH, oW):
        """All (dr,dc) placing every cell on the border of oH×oW."""
        if not cells:
            return []
        min_r = min(r for r, c in cells)
        max_r = max(r for r, c in cells)
        min_c = min(c for r, c in cells)
        max_c = max(c for r, c in cells)
        if max_r - min_r >= oH or max_c - min_c >= oW:
            return []
        valid = []
        for dr in range(-min_r, oH - max_r):
            for dc in range(-min_c, oW - max_c):
                if all(r+dr == 0 or r+dr == oH-1 or c+dc == 0 or c+dc == oW-1
                       for r, c in cells):
                    valid.append((dr, dc))
        return valid

    def find_best_rect(cells):
        """Largest-area 2×2 sub-rectangle embedded in cells; returns (r1,c1,r2,c2) or None."""
        cell_set = set(cells)
        rows = sorted(set(r for r, c in cells))
        cols = sorted(set(c for r, c in cells))
        best, best_area = None, -1
        for i, r1 in enumerate(rows):
            for r2 in rows[i+1:]:
                for j, c1 in enumerate(cols):
                    for c2 in cols[j+1:]:
                        if (r1,c1) in cell_set and (r1,c2) in cell_set \
                                and (r2,c1) in cell_set and (r2,c2) in cell_set:
                            area = (r2 - r1) * (c2 - c1)
                            if area > best_area:
                                best_area = area
                                best = (r1, c1, r2, c2)
        return best

    # ── Step 1: determine output size ────────────────────────────────────────
    oH = oW = None
    corner_color = None
    corner_rect = None

    # Prefer color whose ALL cells are exactly the 4 corners (clean corner color)
    for color, cells in color_cells.items():
        if len(cells) == 4:
            rows = sorted(set(r for r, c in cells))
            cols = sorted(set(c for r, c in cells))
            if len(rows) == 2 and len(cols) == 2:
                expected = {(rows[0], cols[0]), (rows[0], cols[1]),
                            (rows[1], cols[0]), (rows[1], cols[1])}
                if set(cells) == expected:
                    oH = rows[1] - rows[0] + 1
                    oW = cols[1] - cols[0] + 1
                    corner_color = color
                    corner_rect = (rows[0], cols[0], rows[1], cols[1])
                    break

    # Fall back: look for an embedded rectangle in any multi-cell color
    if oH is None:
        best_area = -1
        for color, cells in color_cells.items():
            if len(cells) < 4:
                continue
            rect = find_best_rect(cells)
            if rect is not None:
                r1, c1, r2, c2 = rect
                area = (r2 - r1) * (c2 - c1)
                if area > best_area:
                    best_area = area
                    oH = r2 - r1 + 1
                    oW = c2 - c1 + 1
                    corner_color = color
                    corner_rect = rect

    if oH is None:
        oH, oW = 7, 6   # fallback default

    # ── Step 2: build list of (cells_subgroup, color) to place ───────────────
    groups: list[tuple[list, int]] = []

    for color, cells in color_cells.items():
        if color == corner_color and corner_rect is not None:
            r1, c1, r2, c2 = corner_rect
            rect_cells = [(r1,c1),(r1,c2),(r2,c1),(r2,c2)]
            rest_cells  = [cell for cell in cells if cell not in rect_cells]
            groups.append((rect_cells, color))
            if rest_cells:
                groups.append((rest_cells, color))
        else:
            groups.append((cells, color))

    # ── Step 3: assign displacements (most-constrained first) ────────────────
    groups.sort(key=lambda g: len(border_disps(g[0], oH, oW)))

    output = [[bg] * oW for _ in range(oH)]

    for cells, color in groups:
        disps = border_disps(cells, oH, oW)
        if not disps:
            continue
        chosen = None
        for dr, dc in disps:
            positions = [(r+dr, c+dc) for r, c in cells]
            if all(output[r][c] == bg for r, c in positions):
                chosen = (dr, dc)
                break
        if chosen is None:
            chosen = disps[0]
        dr, dc = chosen
        for r, c in cells:
            output[r+dr][c+dc] = color

    return output


# ── Verification helpers ──────────────────────────────────────────────────────

def make_grid(H, W, bg, cells_dict):
    g = [[bg] * W for _ in range(H)]
    for (r, c), v in cells_dict.items():
        g[r][c] = v
    return g


train0_in = make_grid(22, 22, 5, {
    (2,17):3,(3,15):1,(3,17):3,(4,12):3,(7,15):3,
    (8,13):1,(8,18):1,(9,15):2,(9,20):2,
    (14,10):0,(14,11):0,(15,15):2,(15,20):2,
    (17,12):0,(17,21):4,(18,20):4,(19,20):4,(21,20):4,
})
train0_exp = [
    [2,4,1,0,0,2],[4,5,5,5,5,3],[4,5,5,5,5,3],
    [3,5,5,5,5,0],[4,5,5,5,5,5],[1,5,5,5,5,1],[2,5,5,3,5,2],
]

train1_in = make_grid(21, 20, 0, {
    (1,16):7,(2,12):7,(4,7):2,(5,12):7,
    (6,14):6,(7,12):2,(8,8):2,(8,16):6,
    (9,11):6,(17,8):1,(18,8):1,(19,3):1,(20,5):1,
})
train1_exp = [
    [0,0,0,6,7,0],[7,0,0,0,0,0],[2,0,0,0,0,6],
    [6,0,0,0,0,1],[7,0,0,0,0,1],[1,0,0,0,0,2],[0,2,1,0,0,0],
]

train2_in = make_grid(21, 21, 0, {
    (2,3):3,(3,12):6,(3,17):6,(5,2):3,
    (8,6):3,(9,8):5,(9,12):6,(9,17):6,
    (10,5):5,(15,8):5,
})
train2_exp = [
    [6,3,0,5,0,6],[5,0,0,0,0,0],[0,0,0,0,0,0],
    [3,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[6,0,0,5,3,6],
]

all_pass = True
for i, (inp, exp) in enumerate([
        (train0_in, train0_exp),
        (train1_in, train1_exp),
        (train2_in, train2_exp)]):
    got = transform(inp)
    ok = (got == exp)
    print(f"Train {i}: {'PASS ✓' if ok else 'FAIL ✗'}")
    if not ok:
        all_pass = False
        print("  Expected:"); [print(" ", r) for r in exp]
        print("  Got:");      [print(" ", r) for r in got]

print()
print("All training pairs pass:", all_pass)

# ── Test outputs ──────────────────────────────────────────────────────────────
test0_in = make_grid(22, 22, 2, {
    (5,6):0,(5,11):0,(6,8):0,(7,3):0,
    (7,13):3,(8,9):3,(10,4):0,(10,6):0,
    (10,11):0,(11,14):3,(15,10):4,
    (17,2):7,(17,14):4,(19,4):7,(19,5):7,(20,13):4,
})
test1_in = make_grid(20, 21, 0, {
    (3,11):6,(3,13):2,(3,15):2,(3,17):6,
    (4,8):4,(5,10):2,(6,13):4,(6,16):2,
    (8,7):4,(8,11):6,(8,17):6,(9,8):4,
    (11,13):3,(12,17):3,(14,11):3,(16,15):3,
})

r0 = transform(test0_in)
r1 = transform(test1_in)
print(f"\nTest 0 output ({len(r0)}×{len(r0[0])}):")
for row in r0: print(row)

print(f"\nTest 1 output ({len(r1)}×{len(r1[0])}):")
for row in r1: print(row)
