from collections import Counter, defaultdict

def transform(grid):
    H = len(grid)
    W = len(grid[0])
    flat = [grid[r][c] for r in range(H) for c in range(W)]
    bg = Counter(flat).most_common(1)[0][0]

    color_cells = defaultdict(list)
    for r in range(H):
        for c in range(W):
            if grid[r][c] != bg:
                color_cells[grid[r][c]].append((r, c))

    def is_on_border(r, c, r1, c1, r2, c2):
        return r1 <= r <= r2 and c1 <= c <= c2 and (r == r1 or r == r2 or c == c1 or c == c2)

    def find_displacement(cells, r1, c1, r2, c2):
        """Find (dr,dc) so all cells land on border of rect [r1..r2]x[c1..c2]."""
        if not cells:
            return (0, 0)
        r0, c0 = cells[0]
        candidates = set()
        for nc in range(c1, c2 + 1):
            candidates.add((r1 - r0, nc - c0))
            candidates.add((r2 - r0, nc - c0))
        for nr in range(r1 + 1, r2):
            candidates.add((nr - r0, c1 - c0))
            candidates.add((nr - r0, c2 - c0))
        for dr, dc in sorted(candidates):
            if all(is_on_border(r + dr, c + dc, r1, c1, r2, c2) for r, c in cells):
                return (dr, dc)
        return None

    def find_rect_corners(cells):
        """Find 4 cells forming rectangle corners. Returns (r1,c1,r2,c2) or None."""
        cell_set = set(cells)
        for i in range(len(cells)):
            r1, c1 = cells[i]
            for j in range(i + 1, len(cells)):
                r2, c2 = cells[j]
                if r1 != r2:
                    continue
                cs, cl = min(c1, c2), max(c1, c2)
                for rb, _ in cells:
                    if rb <= r1:
                        continue
                    if (rb, cs) in cell_set and (rb, cl) in cell_set:
                        return (r1, cs, rb, cl)
        return None

    # Find frame: look for a color with 4+ cells whose subset forms rectangle corners
    frame = None
    frame_color = None
    frame_corners_set = None

    for color, cells in color_cells.items():
        if len(cells) >= 4:
            rect = find_rect_corners(cells)
            if rect is None:
                continue
            r1f, c1f, r2f, c2f = rect
            corners_set = {(r1f, c1f), (r1f, c2f), (r2f, c1f), (r2f, c2f)}
            other_cells = [c for c in cells if c not in corners_set]
            if other_cells and find_displacement(other_cells, r1f, c1f, r2f, c2f) is None:
                continue
            if all(find_displacement(ocells, r1f, c1f, r2f, c2f) is not None
                   for oc, ocells in color_cells.items() if oc != color):
                frame = rect
                frame_color = color
                frame_corners_set = corners_set
                break

    if frame is None:
        for oH, oW in [(7, 6), (6, 7), (6, 6), (8, 6), (6, 8)]:
            found = False
            for r1 in range(H - oH + 1):
                for c1 in range(W - oW + 1):
                    r2, c2 = r1 + oH - 1, c1 + oW - 1
                    if all(find_displacement(cells, r1, c1, r2, c2) is not None
                           for cells in color_cells.values()):
                        frame = (r1, c1, r2, c2)
                        found = True
                        break
                if found:
                    break
            if frame:
                break
        if not frame:
            return [[bg] * 6 for _ in range(7)]

    r1, c1, r2, c2 = frame
    oH, oW = r2 - r1 + 1, c2 - c1 + 1
    output = [[bg] * oW for _ in range(oH)]

    for color, cells in color_cells.items():
        if color == frame_color and frame_corners_set:
            for r, c in cells:
                if (r, c) in frame_corners_set:
                    or_ = 0 if r == r1 else oH - 1
                    oc = 0 if c == c1 else oW - 1
                    output[or_][oc] = color
            other = [(r, c) for r, c in cells if (r, c) not in frame_corners_set]
            if other:
                d = find_displacement(other, r1, c1, r2, c2)
                if d:
                    dr, dc = d
                    for r, c in other:
                        output[r + dr - r1][c + dc - c1] = color
        else:
            d = find_displacement(cells, r1, c1, r2, c2)
            if d:
                dr, dc = d
                for r, c in cells:
                    output[r + dr - r1][c + dc - c1] = color

    return output


# ---- Verification ----

train0_input = [
    [5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5],
    [5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5],
    [5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,3,5,5,5,5],
    [5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,1,5,3,5,5,5,5],
    [5,5,5,5,5,5,5,5,5,5,5,5,3,5,5,5,5,5,5,5,5,5],
    [5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5],
    [5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5],
    [5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,3,5,5,5,5,5,5],
    [5,5,5,5,5,5,5,5,5,5,5,5,5,1,5,5,5,5,1,5,5,5],
    [5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,2,5,5,5,5,2,5],
    [5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5],
    [5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5],
    [5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5],
    [5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5],
    [5,5,5,5,5,5,5,5,5,5,0,0,5,5,5,5,5,5,5,5,5,5],
    [5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,2,5,5,5,5,2,5],
    [5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5],
    [5,5,5,5,5,5,5,5,5,5,5,5,0,5,5,5,5,5,5,5,5,4],
    [5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,4,5],
    [5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,4,5],
    [5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5],
    [5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,4,5],
]

train0_expected = [
    [2,4,1,0,0,2],
    [4,5,5,5,5,3],
    [4,5,5,5,5,3],
    [3,5,5,5,5,0],
    [4,5,5,5,5,5],
    [1,5,5,5,5,1],
    [2,5,5,3,5,2],
]

train1_input = [
    [0]*20,
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,7,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,7,0,0,0,0,0,0,0],
    [0]*20,
    [0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,7,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,6,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,6,0,0,0,0,0,0,0,0],
    [0]*20,
    [0]*20,
    [0]*20,
    [0]*20,
    [0]*20,
    [0]*20,
    [0]*20,
    [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
]

train1_expected = [
    [0,0,0,6,7,0],
    [7,0,0,0,0,0],
    [2,0,0,0,0,6],
    [6,0,0,0,0,1],
    [7,0,0,0,0,1],
    [1,0,0,0,0,2],
    [0,2,1,0,0,0],
]

train2_input = [
    [0]*21,
    [0]*21,
    [0,0,0,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,6,0,0,0,0,6,0,0,0],
    [0]*21,
    [0,0,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0]*21,
    [0]*21,
    [0,0,0,0,0,0,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,5,0,0,0,6,0,0,0,0,6,0,0,0],
    [0,0,0,0,0,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0]*21,
    [0]*21,
    [0]*21,
    [0]*21,
    [0,0,0,0,0,0,0,0,5,0,0,0,0,0,0,0,0,0,0,0,0],
    [0]*21,
    [0]*21,
    [0]*21,
    [0]*21,
    [0]*21,
]

train2_expected = [
    [6,3,0,5,0,6],
    [5,0,0,0,0,0],
    [0,0,0,0,0,0],
    [3,0,0,0,0,0],
    [0,0,0,0,0,0],
    [0,0,0,0,0,0],
    [6,0,0,5,3,6],
]

def grid_eq(a, b):
    return all(a[r][c] == b[r][c] for r in range(len(a)) for c in range(len(a[0])))

def print_grid(g):
    for row in g:
        print(row)

print("=== Train 0 ===")
r0 = transform(train0_input)
print_grid(r0)
print("PASS" if grid_eq(r0, train0_expected) else "FAIL")
print()

print("=== Train 1 ===")
r1 = transform(train1_input)
print_grid(r1)
print("PASS" if grid_eq(r1, train1_expected) else "FAIL")
print()

print("=== Train 2 ===")
r2 = transform(train2_input)
print_grid(r2)
print("PASS" if grid_eq(r2, train2_expected) else "FAIL")
print()

# Test inputs
test0_input_raw = {
    (5,6):0,(5,11):0,(6,8):0,(7,3):0,(7,13):3,(8,9):3,
    (10,4):0,(10,6):0,(10,11):0,(11,14):3,(15,10):4,
    (17,2):7,(17,14):4,(19,4):7,(19,5):7,(20,13):4
}
test0_bg = 2
test0 = [[test0_bg]*22 for _ in range(22)]
for (r,c),v in test0_input_raw.items():
    test0[r][c] = v

print("=== Test 0 ===")
t0 = transform(test0)
print_grid(t0)
print()

test1_input_raw = {
    (3,11):6,(3,13):2,(3,15):2,(3,17):6,(4,8):4,(5,10):2,
    (6,13):4,(6,16):2,(8,7):4,(8,11):6,(8,17):6,(9,8):4,
    (11,13):3,(12,17):3,(14,11):3,(16,15):3
}
test1_bg = 0
test1 = [[test1_bg]*21 for _ in range(20)]
for (r,c),v in test1_input_raw.items():
    test1[r][c] = v

print("=== Test 1 ===")
t1 = transform(test1)
print_grid(t1)
