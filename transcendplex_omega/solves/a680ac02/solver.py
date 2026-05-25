"""ARC-AGI task a680ac02 solver.

Pattern: Input has 4x4 "frame" objects (colored border, 0 interior) and
4x4 "solid" objects (all one color). Output = frames only, concatenated
horizontally (sorted by col) if col_spread >= row_spread, else vertically
(sorted by row).
"""
import json
from typing import List

Grid = List[List[int]]


def solve(grid: Grid) -> Grid:
    rows, cols = len(grid), len(grid[0])
    visited = [[False]*cols for _ in range(rows)]
    frames = []

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0 and not visited[r][c]:
                color = grid[r][c]
                stack, cells = [(r, c)], []
                while stack:
                    cr, cc = stack.pop()
                    if 0 <= cr < rows and 0 <= cc < cols and not visited[cr][cc] and grid[cr][cc] == color:
                        visited[cr][cc] = True
                        cells.append((cr, cc))
                        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                            stack.append((cr+dr, cc+dc))
                if not cells:
                    continue
                mr = min(x[0] for x in cells)
                mc = min(x[1] for x in cells)
                h = max(x[0] for x in cells) - mr + 1
                w = max(x[1] for x in cells) - mc + 1
                if h != 4 or w != 4:
                    continue
                block = [row[mc:mc+4] for row in grid[mr:mr+4]]
                # Frame = interior cells are 0
                if block[1][1] == 0 and block[1][2] == 0 and block[2][1] == 0 and block[2][2] == 0:
                    frames.append({'row': mr, 'col': mc, 'block': block})

    if not frames:
        return grid

    rs = [f['row'] for f in frames]
    cs = [f['col'] for f in frames]
    if max(cs)-min(cs) >= max(rs)-min(rs):
        frames.sort(key=lambda f: f['col'])
        return [[v for f in frames for v in f['block'][i]] for i in range(4)]
    else:
        frames.sort(key=lambda f: f['row'])
        out = []
        for f in frames:
            out.extend([list(r) for r in f['block']])
        return out


if __name__ == '__main__':
    with open('/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/a680ac02.json') as f:
        task = json.load(f)
    ok = True
    for i, p in enumerate(task['train']):
        res = solve(p['input'])
        m = res == p['output']
        print(f"Train {i}: {'PASS' if m else 'FAIL'}")
        if not m:
            ok = False
            print(f"  Exp: {p['output']}")
            print(f"  Got: {res}")
    for i, p in enumerate(task['test']):
        res = solve(p['input'])
        print(f"Test {i}: {len(res)}x{len(res[0])}")
        for r in res: print(f"  {r}")
    print(f"\nAll train pass: {ok}")
