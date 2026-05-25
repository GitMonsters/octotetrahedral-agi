"""
Puzzle 20270e3b — Shape Portal Stitching

Rule: Two yellow (4) shapes are separated by a blue (1) gap. Orange (7) markers
on each shape indicate matching "ports". The mobile shape is translated so its
port-face aligns with the base shape's orange markers, closing the gap.
Output grid = tight bounding box of the combined shape, filled with blue.
"""
import json, os
from collections import deque
from typing import List, Set, Tuple, Dict

Grid = List[List[int]]
Cell = Tuple[int, int]
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _find_components(grid: Grid, color: int) -> List[Set[Cell]]:
    H, W = len(grid), len(grid[0])
    cells = {(r, c) for r in range(H) for c in range(W) if grid[r][c] == color}
    visited: Set[Cell] = set()
    comps: List[Set[Cell]] = []
    for cell in sorted(cells):
        if cell in visited:
            continue
        comp: Set[Cell] = set()
        q = deque([cell])
        while q:
            r, c = q.popleft()
            if (r, c) in visited:
                continue
            visited.add((r, c))
            comp.add((r, c))
            for dr, dc in DIRS:
                nr, nc = r + dr, c + dc
                if (nr, nc) in cells and (nr, nc) not in visited:
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def _min_dist(a: Set[Cell], b: Set[Cell]) -> int:
    return min(abs(r1 - r2) + abs(c1 - c2) for r1, c1 in a for r2, c2 in b)


def solve(grid: Grid) -> Grid:
    H, W = len(grid), len(grid[0])

    # 1. Yellow connected components
    comps = _find_components(grid, 4)

    # 2. Orange cells, assigned to adjacent yellow component
    orange_cells = [(r, c) for r in range(H) for c in range(W) if grid[r][c] == 7]
    comp_orange: Dict[int, list] = {i: [] for i in range(len(comps))}
    for r, c in orange_cells:
        for i, comp in enumerate(comps):
            if any((r + dr, c + dc) in comp for dr, dc in DIRS):
                comp_orange[i].append((r, c))
                break

    # 3. Identify two main components (those with orange neighbours)
    main_ids = [i for i in range(len(comps)) if comp_orange[i]]
    assert len(main_ids) == 2, f"Expected 2 main comps, got {len(main_ids)}"

    # Base = larger component; tie-break by topmost/leftmost
    main_ids.sort(key=lambda i: (-len(comps[i]), min(comps[i])))
    base_idx, mobile_idx = main_ids

    base = comps[base_idx]
    mobile_main = comps[mobile_idx]

    orange_base = sorted(comp_orange[base_idx])
    orange_mobile = sorted(comp_orange[mobile_idx])

    # 4. Face of mobile: cells adjacent to its orange markers
    om_set = set(orange_mobile)
    face_mobile = sorted(
        (r, c) for r, c in mobile_main
        if any((r + dr, c + dc) in om_set for dr, dc in DIRS)
    )

    # 5. Translation: face_mobile[0] -> orange_base[0]
    T = (orange_base[0][0] - face_mobile[0][0],
         orange_base[0][1] - face_mobile[0][1])

    # 6. Classify extra (non-main) components by proximity
    combined: Set[Cell] = set(base)
    for i in range(len(comps)):
        if i == base_idx:
            continue
        if i == mobile_idx:
            for r, c in comps[i]:
                combined.add((r + T[0], c + T[1]))
        else:
            d_base = _min_dist(comps[i], base)
            d_mobile = _min_dist(comps[i], mobile_main)
            if d_mobile <= d_base:
                for r, c in comps[i]:
                    combined.add((r + T[0], c + T[1]))
            else:
                combined |= comps[i]

    # 7. Bounding box -> output grid
    min_r = min(r for r, _ in combined)
    max_r = max(r for r, _ in combined)
    min_c = min(c for _, c in combined)
    max_c = max(c for _, c in combined)
    out_H = max_r - min_r + 1
    out_W = max_c - min_c + 1
    out = [[1] * out_W for _ in range(out_H)]
    for r, c in combined:
        out[r - min_r][c - min_c] = 4
    return out


# ── Validation ──────────────────────────────────────────────────────
def validate():
    task_path = os.path.join(os.path.dirname(__file__), "..", "..", "dataset", "tasks", "20270e3b.json")
    with open(task_path) as f:
        task = json.load(f)

    all_pass = True
    for split in ["train", "test"]:
        for i, pair in enumerate(task[split]):
            result = solve(pair["input"])
            expected = pair["output"]
            if result == expected:
                print(f"{split.capitalize()} {i}: ✅ 0 diffs")
            else:
                diffs = 0
                rH, eH = len(result), len(expected)
                rW = len(result[0]) if result else 0
                eW = len(expected[0]) if expected else 0
                if rH != eH or rW != eW:
                    print(f"{split.capitalize()} {i}: ❌ size mismatch {rH}x{rW} vs {eH}x{eW}")
                    all_pass = False
                    continue
                for r in range(rH):
                    for c in range(rW):
                        if result[r][c] != expected[r][c]:
                            diffs += 1
                            if diffs <= 5:
                                print(f"  ({r},{c}): got {result[r][c]}, expected {expected[r][c]}")
                print(f"{split.capitalize()} {i}: ❌ {diffs} diffs")
                all_pass = False
    return all_pass


if __name__ == "__main__":
    validate()
