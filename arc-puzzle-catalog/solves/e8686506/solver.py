import json
from collections import Counter, deque


def solve(grid: list[list[int]]) -> list[list[int]]:
    rows = len(grid)
    cols = len(grid[0])

    bg = Counter(
        grid[r][c] for r in range(rows) for c in range(cols)
    ).most_common(1)[0][0]

    color_cells: dict[int, list[tuple[int, int]]] = {}
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg:
                color_cells.setdefault(grid[r][c], []).append((r, c))

    # A valid frame candidate is a set of colors whose bounding box
    # contains only those colors + bg, and has at least one hole.
    def check_candidate(color_set: frozenset):
        cells = []
        for c in color_set:
            cells.extend(color_cells.get(c, []))
        if not cells:
            return None
        r0 = min(r for r, _ in cells)
        r1 = max(r for r, _ in cells)
        c0 = min(c for _, c in cells)
        c1 = max(c for _, c in cells)
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if grid[r][c] != bg and grid[r][c] not in color_set:
                    return None
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if area == len(cells):
            return None
        return (color_set, cells, (r0, r1, c0, c1))

    candidates = []
    all_colors = list(color_cells.keys())
    for c in all_colors:
        res = check_candidate(frozenset({c}))
        if res:
            candidates.append(res)
    for i in range(len(all_colors)):
        for j in range(i + 1, len(all_colors)):
            res = check_candidate(frozenset({all_colors[i], all_colors[j]}))
            if res:
                candidates.append(res)

    # Sort candidates: more frame cells first (best-first search)
    candidates.sort(key=lambda x: -len(x[1]))

    for frame_colors, frame_cells, (r0, r1, c0, c1) in candidates:
        fh, fw = r1 - r0 + 1, c1 - c0 + 1

        out = [[bg] * fw for _ in range(fh)]
        frame_local: set[tuple[int, int]] = set()
        for r, c in frame_cells:
            lr, lc = r - r0, c - c0
            out[lr][lc] = grid[r][c]
            frame_local.add((lr, lc))

        holes: set[tuple[int, int]] = set()
        for lr in range(fh):
            for lc in range(fw):
                if (lr, lc) not in frame_local:
                    holes.add((lr, lc))

        # Collect external connected components (same-color BFS)
        vis = [[False] * cols for _ in range(rows)]
        ext: list[tuple[int, tuple[tuple[int, int], ...]]] = []
        for r in range(rows):
            for c in range(cols):
                v = grid[r][c]
                if v != bg and v not in frame_colors and not vis[r][c]:
                    q = deque([(r, c)])
                    vis[r][c] = True
                    comp: list[tuple[int, int]] = []
                    while q:
                        cr, cc = q.popleft()
                        comp.append((cr, cc))
                        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                            nr, nc = cr + dr, cc + dc
                            if (
                                0 <= nr < rows
                                and 0 <= nc < cols
                                and not vis[nr][nc]
                                and grid[nr][nc] == v
                            ):
                                vis[nr][nc] = True
                                q.append((nr, nc))
                    mr = min(r for r, _ in comp)
                    mc = min(c for _, c in comp)
                    shape = tuple(
                        sorted((r - mr, c - mc) for r, c in comp)
                    )
                    ext.append((v, shape))

        # Quick filter: total external cells must equal total holes
        if sum(len(s) for _, s in ext) != len(holes):
            continue

        ext.sort(key=lambda x: -len(x[1]))

        placements: dict[tuple[int, int], int] = {}

        def backtrack(idx: int, remaining: set[tuple[int, int]]) -> bool:
            if idx == len(ext):
                return len(remaining) == 0

            # Fast-path: all remaining pieces are single-cell same color
            if all(len(ext[j][1]) == 1 for j in range(idx, len(ext))):
                left_colors = set(ext[j][0] for j in range(idx, len(ext)))
                if len(left_colors) == 1 and len(remaining) == len(ext) - idx:
                    clr = next(iter(left_colors))
                    for pos in remaining:
                        placements[pos] = clr
                    return True

            color, shape = ext[idx]
            sr0, sc0 = shape[0]

            for hr, hc in sorted(remaining):
                dr, dc = hr - sr0, hc - sc0
                placed: list[tuple[int, int]] = []
                ok = True
                for sr, sc in shape:
                    nr, nc = dr + sr, dc + sc
                    if (nr, nc) not in remaining:
                        ok = False
                        break
                    placed.append((nr, nc))
                if ok:
                    new_rem = remaining - set(placed)
                    if backtrack(idx + 1, new_rem):
                        for p in placed:
                            placements[p] = color
                        return True

            return False

        if backtrack(0, holes):
            for (lr, lc), color in placements.items():
                out[lr][lc] = color
            return out

    return []


if __name__ == "__main__":
    with open(
        "/Users/evanpieser/arc-puzzle-catalog/dataset/tasks/e8686506.json"
    ) as f:
        task = json.load(f)

    for i, pair in enumerate(task.get("train", [])):
        result = solve(pair["input"])
        expected = pair["output"]
        status = "PASS" if result == expected else "FAIL"
        print(f"Train {i}: {status}")
        if status == "FAIL":
            print(f"  Expected: {expected}")
            print(f"  Got:      {result}")

    for i, pair in enumerate(task.get("test", [])):
        result = solve(pair["input"])
        if "output" in pair:
            expected = pair["output"]
            status = "PASS" if result == expected else "FAIL"
            print(f"Test  {i}: {status}")
            if status == "FAIL":
                print(f"  Expected: {expected}")
                print(f"  Got:      {result}")
        else:
            print(f"Test  {i}: (no expected output) Result: {result}")
