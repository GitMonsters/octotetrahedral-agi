import json
from itertools import combinations
from collections import deque


def solve(grid):
    """Shapes of color 1 that are 'stretched' versions of the color-2 reference
    (under any rotation/reflection) get recolored to 2. Others stay as 1.
    Stretching = repeating rows/columns (non-uniform nearest-neighbor upscale)."""
    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]

    # Extract reference pattern from color-2 cells
    ref_cells = {(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 2}
    if not ref_cells:
        return result

    rmin = min(r for r, c in ref_cells)
    rmax = max(r for r, c in ref_cells)
    cmin = min(c for r, c in ref_cells)
    cmax = max(c for r, c in ref_cells)
    ref = [[1 if (r, c) in ref_cells else 0
            for c in range(cmin, cmax + 1)]
           for r in range(rmin, rmax + 1)]

    orientations = _orientations(ref)

    # Find connected components of color 1
    visited = set()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1 and (r, c) not in visited:
                comp = _flood(grid, r, c, visited, rows, cols)
                cr_min = min(r for r, _ in comp)
                cr_max = max(r for r, _ in comp)
                cc_min = min(c for _, c in comp)
                cc_max = max(c for _, c in comp)
                pat = [[1 if (r, c) in comp else 0
                        for c in range(cc_min, cc_max + 1)]
                       for r in range(cr_min, cr_max + 1)]
                if _matches(orientations, pat):
                    for r2, c2 in comp:
                        result[r2][c2] = 2
    return result


def _flood(grid, sr, sc, visited, rows, cols):
    comp = set()
    q = deque([(sr, sc)])
    visited.add((sr, sc))
    while q:
        r, c = q.popleft()
        comp.add((r, c))
        for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
            nr, nc = r+dr, c+dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited and grid[nr][nc] == 1:
                visited.add((nr, nc))
                q.append((nr, nc))
    return comp


def _orientations(pat):
    seen = set()
    result = []
    p = [row[:] for row in pat]
    for _ in range(4):
        for q in (p, [row[::-1] for row in p]):
            key = tuple(tuple(row) for row in q)
            if key not in seen:
                seen.add(key)
                result.append([list(row) for row in q])
        h, w = len(p), len(p[0])
        p = [[p[h-1-c][r] for c in range(h)] for r in range(w)]
    return result


def _matches(orientations, target):
    return any(_can_stretch(ref, target) for ref in orientations)


def _can_stretch(ref, target):
    h, w = len(ref), len(ref[0])
    H, W = len(target), len(target[0])
    if H < h or W < w:
        return False

    for rsplits in combinations(range(1, H), h - 1):
        rb = [0] + list(rsplits) + [H]
        rg = [(rb[i], rb[i+1]) for i in range(h)]
        # Quick check: first row of each group must start matching
        ok = True
        for ri in range(h):
            if target[rg[ri][0]][0] != ref[ri][0]:
                ok = False
                break
        if not ok:
            continue

        for csplits in combinations(range(1, W), w - 1):
            cb = [0] + list(csplits) + [W]
            cg = [(cb[j], cb[j+1]) for j in range(w)]
            if _check(ref, target, rg, cg, h, w):
                return True
    return False


def _check(ref, target, rg, cg, h, w):
    for ri in range(h):
        rs, re = rg[ri]
        for ci in range(w):
            cs, ce = cg[ci]
            exp = ref[ri][ci]
            for r in range(rs, re):
                for c in range(cs, ce):
                    if target[r][c] != exp:
                        return False
    return True


if __name__ == '__main__':
    import sys
    task = json.load(open(sys.argv[1]))
    ok = True
    for i, pair in enumerate(task['train']):
        res = solve(pair['input'])
        if res == pair['output']:
            print(f'Train {i}: PASS')
        else:
            print(f'Train {i}: FAIL')
            ok = False
            for r in range(len(pair['output'])):
                for c in range(len(pair['output'][0])):
                    if res[r][c] != pair['output'][r][c]:
                        print(f'  ({r},{c}): got {res[r][c]} expected {pair["output"][r][c]}')
    for i, pair in enumerate(task['test']):
        res = solve(pair['input'])
        print(f'Test {i} output:')
        for row in res:
            print(row)
    if ok:
        print('\nAll training examples PASSED!')
