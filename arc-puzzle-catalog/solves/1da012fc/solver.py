import copy
from collections import deque


def solve(grid: list[list[int]]) -> list[list[int]]:
    rows = len(grid)
    cols = len(grid[0])
    result = copy.deepcopy(grid)

    # 1. Find palette (bounding box of all 5-valued cells)
    five_cells = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 5]
    if not five_cells:
        return result
    pr0 = min(r for r, c in five_cells)
    pr1 = max(r for r, c in five_cells)
    pc0 = min(c for r, c in five_cells)
    pc1 = max(c for r, c in five_cells)
    palette_cells = {(r, c) for r in range(pr0, pr1 + 1) for c in range(pc0, pc1 + 1)}

    # 2. Find colored dots inside palette
    dots = []
    for r in range(pr0, pr1 + 1):
        for c in range(pc0, pc1 + 1):
            v = grid[r][c]
            if v != 5 and v != 0:
                dots.append((r, c, v))

    ph = pr1 - pr0  # palette_rows - 1
    pw = pc1 - pc0  # palette_cols - 1
    dot_norm = []
    for r, c, color in dots:
        nr = (r - pr0) / ph if ph > 0 else 0.5
        nc = (c - pc0) / pw if pw > 0 else 0.5
        dot_norm.append((nr, nc, color))

    # 3. Find shapes outside palette (connected components of non-0, non-5)
    visited: set[tuple[int, int]] = set()
    shapes: list[list[tuple[int, int]]] = []

    for r in range(rows):
        for c in range(cols):
            if (r, c) in palette_cells or (r, c) in visited:
                continue
            if grid[r][c] == 0 or grid[r][c] == 5:
                continue
            # BFS
            q = deque([(r, c)])
            visited.add((r, c))
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr2, nc2 = cr + dr, cc + dc
                    if 0 <= nr2 < rows and 0 <= nc2 < cols and (nr2, nc2) not in visited and (nr2, nc2) not in palette_cells:
                        if grid[nr2][nc2] != 0 and grid[nr2][nc2] != 5:
                            visited.add((nr2, nc2))
                            q.append((nr2, nc2))
            shapes.append(cells)

    if not shapes or not dots:
        return result

    # 4. Compute shape centers
    centers = []
    for cells in shapes:
        ar = sum(r for r, c in cells) / len(cells)
        ac = sum(c for r, c in cells) / len(cells)
        centers.append((ar, ac))

    # 5. Normalize shape centers within their bounding box
    if len(centers) == 1:
        shape_norm = [(0.5, 0.5)]
    else:
        sr0 = min(r for r, c in centers)
        sr1 = max(r for r, c in centers)
        sc0 = min(c for r, c in centers)
        sc1 = max(c for r, c in centers)
        rng_r = sr1 - sr0 if sr1 != sr0 else 1.0
        rng_c = sc1 - sc0 if sc1 != sc0 else 1.0
        shape_norm = [((r - sr0) / rng_r, (c - sc0) / rng_c) for r, c in centers]

    # 6. Match each shape to nearest dot (Euclidean in normalized space)
    for i, (snr, snc) in enumerate(shape_norm):
        best_dist = float('inf')
        best_color = 0
        for dnr, dnc, color in dot_norm:
            d = (snr - dnr) ** 2 + (snc - dnc) ** 2
            if d < best_dist:
                best_dist = d
                best_color = color
        for r, c in shapes[i]:
            result[r][c] = best_color

    return result


if __name__ == "__main__":
    import json
    with open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/1da012fc.json") as f:
        task = json.load(f)
    for i, ex in enumerate(task["train"]):
        result = solve(ex["input"])
        status = "PASS" if result == ex["output"] else "FAIL"
        print(f"Train {i}: {status}")
        if status == "FAIL":
            for r_idx in range(len(result)):
                if result[r_idx] != ex["output"][r_idx]:
                    print(f"  Row {r_idx}: got {result[r_idx]}")
                    print(f"  Row {r_idx}: exp {ex['output'][r_idx]}")
    for i, ex in enumerate(task["test"]):
        result = solve(ex["input"])
        print(f"Test {i}: produced {len(result)}x{len(result[0])}")
