"""
Solver for ARC-AGI task 3a25b0d8.

Pattern: The input contains two shapes (connected components of non-background
cells) sharing the same border color. One shape is "filled" (has multiple
interior colors), the other is an "outline" (interior is empty/background).
Both shapes have the same topological structure. The output is the outline
shape's bounding box with its interior chambers filled using the colors from
the corresponding chambers in the filled shape, matched by normalized centroid
proximity.
"""

import json
from collections import deque


def solve(grid: list[list[int]]) -> list[list[int]]:
    rows = len(grid)
    cols = len(grid[0])

    # Background = most common color
    cnt: dict[int, int] = {}
    for r in range(rows):
        for c in range(cols):
            cnt[grid[r][c]] = cnt.get(grid[r][c], 0) + 1
    bg = max(cnt, key=cnt.get)

    # Border = most common non-background color
    non_bg_cnt: dict[int, int] = {}
    for r in range(rows):
        for c in range(cols):
            v = grid[r][c]
            if v != bg:
                non_bg_cnt[v] = non_bg_cnt.get(v, 0) + 1
    border = max(non_bg_cnt, key=non_bg_cnt.get)

    # Separate shapes: BFS from colored cells (non-bg, non-border) through
    # 4-connected non-bg cells to find the filled shape. Remaining non-bg
    # cells form the outline.
    DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    filled_cells: set[tuple[int, int]] = set()
    q: deque[tuple[int, int]] = deque()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg and grid[r][c] != border:
                filled_cells.add((r, c))
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in filled_cells and grid[nr][nc] != bg:
                filled_cells.add((nr, nc))
                q.append((nr, nc))

    outline_cells: set[tuple[int, int]] = set()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg and (r, c) not in filled_cells:
                outline_cells.add((r, c))

    # Bounding boxes
    def bbox(cells):
        rs = [r for r, c in cells]
        cs = [c for r, c in cells]
        return min(rs), max(rs), min(cs), max(cs)

    fr0, fr1, fc0, fc1 = bbox(filled_cells)
    or0, or1, oc0, oc1 = bbox(outline_cells)

    # Extract subgrids, masking cells not belonging to each shape with bg
    fg = [[grid[r][c] if (r, c) in filled_cells else bg
           for c in range(fc0, fc1 + 1)] for r in range(fr0, fr1 + 1)]
    og = [[grid[r][c] if (r, c) in outline_cells else bg
           for c in range(oc0, oc1 + 1)] for r in range(or0, or1 + 1)]

    fh, fw = len(fg), len(fg[0])
    oh, ow = len(og), len(og[0])

    # Colored regions in filled shape (same-color 4-connectivity)
    fvis = [[False] * fw for _ in range(fh)]
    colored_regs: list[tuple[int, list[tuple[int, int]]]] = []

    for r in range(fh):
        for c in range(fw):
            v = fg[r][c]
            if not fvis[r][c] and v != bg and v != border:
                reg: list[tuple[int, int]] = []
                q = deque([(r, c)])
                fvis[r][c] = True
                while q:
                    cr, cc = q.popleft()
                    reg.append((cr, cc))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < fh and 0 <= nc < fw and not fvis[nr][nc] and fg[nr][nc] == v:
                            fvis[nr][nc] = True
                            q.append((nr, nc))
                colored_regs.append((v, reg))

    # Interior chambers in outline shape: bg cells enclosed by border
    ext = [[False] * ow for _ in range(oh)]
    q: deque[tuple[int, int]] = deque()
    for r in range(oh):
        for c in range(ow):
            if (r in (0, oh - 1) or c in (0, ow - 1)) and og[r][c] == bg and not ext[r][c]:
                ext[r][c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < oh and 0 <= nc < ow and not ext[nr][nc] and og[nr][nc] == bg:
                ext[nr][nc] = True
                q.append((nr, nc))

    chambers: list[list[tuple[int, int]]] = []
    ovis = [[False] * ow for _ in range(oh)]
    for r in range(oh):
        for c in range(ow):
            if og[r][c] == bg and not ext[r][c] and not ovis[r][c]:
                ch: list[tuple[int, int]] = []
                q = deque([(r, c)])
                ovis[r][c] = True
                while q:
                    cr, cc = q.popleft()
                    ch.append((cr, cc))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < oh and 0 <= nc < ow and not ovis[nr][nc] and og[nr][nc] == bg and not ext[nr][nc]:
                            ovis[nr][nc] = True
                            q.append((nr, nc))
                chambers.append(ch)

    # Match chambers to colored regions by normalized centroid proximity
    def centroid(cells):
        n = len(cells)
        return (sum(r for r, c in cells) / n, sum(c for r, c in cells) / n)

    def norm(pt, h, w):
        return (pt[0] / max(h - 1, 1), pt[1] / max(w - 1, 1))

    reg_info = [(norm(centroid(reg), fh, fw), color) for color, reg in colored_regs]

    out = [row[:] for row in og]
    for ch in chambers:
        cn = norm(centroid(ch), oh, ow)
        best = min(reg_info, key=lambda ri: (cn[0] - ri[0][0]) ** 2 + (cn[1] - ri[0][1]) ** 2)
        for r, c in ch:
            out[r][c] = best[1]

    return out


if __name__ == "__main__":
    import sys

    with open("/Users/evanpieser/arc-puzzle-catalog/dataset/tasks/3a25b0d8.json") as f:
        task = json.load(f)

    train = task.get("train", [])
    test = task.get("test", [])
    all_pass = True

    for i, pair in enumerate(train + test):
        result = solve(pair["input"])
        expected = pair["output"]
        status = "PASS" if result == expected else "FAIL"
        if status == "FAIL":
            all_pass = False
        label = "train" if i < len(train) else "test"
        idx = i if i < len(train) else i - len(train)
        print(f"{label}[{idx}]: {status}")
        if status == "FAIL":
            print(f"  Expected size: {len(expected)}x{len(expected[0])}")
            print(f"  Got size:      {len(result)}x{len(result[0])}")
            for r in range(max(len(expected), len(result))):
                er = expected[r] if r < len(expected) else None
                gr = result[r] if r < len(result) else None
                if er != gr:
                    print(f"  Row {r}: expected {er}")
                    print(f"          got      {gr}")

    sys.exit(0 if all_pass else 1)
