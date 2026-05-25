from collections import deque


def solve(grid: list[list[int]]) -> list[list[int]]:
    """Find two same-shaped objects, identify damaged one (has 0s where template has values),
    output a mask of damaged cells (8) vs intact cells (0), only rows with damage."""
    R, C = len(grid), len(grid[0])

    # Find connected components of non-zero cells
    visited = [[False] * C for _ in range(R)]
    components: list[list[tuple[int, int]]] = []
    for r in range(R):
        for c in range(C):
            if grid[r][c] != 0 and not visited[r][c]:
                comp: list[tuple[int, int]] = []
                q = deque([(r, c)])
                visited[r][c] = True
                while q:
                    cr, cc = q.popleft()
                    comp.append((cr, cc))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < R and 0 <= nc < C and not visited[nr][nc] and grid[nr][nc] != 0:
                            visited[nr][nc] = True
                            q.append((nr, nc))
                components.append(comp)

    # Sort by size descending, take 2 largest as the two objects
    components.sort(key=len, reverse=True)

    def bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
        rs = [p[0] for p in cells]
        cs = [p[1] for p in cells]
        return min(rs), min(cs), max(rs), max(cs)

    def center(cells: list[tuple[int, int]]) -> tuple[float, float]:
        rs = [p[0] for p in cells]
        cs = [p[1] for p in cells]
        return sum(rs) / len(rs), sum(cs) / len(cs)

    # Assign any smaller fragments to the nearest of the two main objects
    obj_cells = [list(components[0]), list(components[1])]
    centers = [center(obj_cells[0]), center(obj_cells[1])]
    for comp in components[2:]:
        cc = center(comp)
        d0 = (cc[0] - centers[0][0]) ** 2 + (cc[1] - centers[0][1]) ** 2
        d1 = (cc[0] - centers[1][0]) ** 2 + (cc[1] - centers[1][1]) ** 2
        if d0 < d1:
            obj_cells[0].extend(comp)
        else:
            obj_cells[1].extend(comp)

    bb0 = bbox(obj_cells[0])
    bb1 = bbox(obj_cells[1])

    h0 = bb0[2] - bb0[0] + 1
    w0 = bb0[3] - bb0[1] + 1
    h1 = bb1[2] - bb1[0] + 1
    w1 = bb1[3] - bb1[1] + 1

    h, w = max(h0, h1), max(w0, w1)

    def extract(bb: tuple[int, int, int, int]) -> list[list[int]]:
        r0, c0 = bb[0], bb[1]
        sub = []
        for i in range(h):
            row = []
            for j in range(w):
                rr, cc = r0 + i, c0 + j
                if 0 <= rr < R and 0 <= cc < C:
                    row.append(grid[rr][cc])
                else:
                    row.append(0)
            sub.append(row)
        return sub

    sub0 = extract(bb0)
    sub1 = extract(bb1)

    # Damaged object has more 0s where the other (template) has non-0
    zeros_in_0 = sum(1 for i in range(h) for j in range(w) if sub0[i][j] == 0 and sub1[i][j] != 0)
    zeros_in_1 = sum(1 for i in range(h) for j in range(w) if sub1[i][j] == 0 and sub0[i][j] != 0)

    if zeros_in_0 > zeros_in_1:
        damaged, template = sub0, sub1
    else:
        damaged, template = sub1, sub0

    # Output: 8 where damaged has 0 but template has non-0, only rows with damage
    result = []
    for i in range(h):
        row = [8 if damaged[i][j] == 0 and template[i][j] != 0 else 0 for j in range(w)]
        if any(v == 8 for v in row):
            result.append(row)

    return result


if __name__ == "__main__":
    import json
    with open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/2037f2c7.json") as f:
        task = json.load(f)
    for i, ex in enumerate(task["train"]):
        result = solve(ex["input"])
        status = "PASS" if result == ex["output"] else "FAIL"
        print(f"Train {i}: {status}")
    for i, ex in enumerate(task["test"]):
        result = solve(ex["input"])
        print(f"Test {i}: produced {len(result)}x{len(result[0])}")
