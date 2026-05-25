"""
Puzzle 7b80bb43: Straighten diagonal connections into right angles.

Rule:
- FG cells form line segments (horizontal/vertical) with some diagonal shortcuts
- Diagonal cells (FG with no h/v FG neighbors, only diagonal FG neighbors) are removed
- Gaps in lines that were bridged by the diagonals are filled with FG
- Gap is filled only if its length <= diagonal chain length + 1
"""
import json, sys
from collections import deque


def solve(grid: list[list[int]]) -> list[list[int]]:
    H, W = len(grid), len(grid[0])
    bg = grid[0][0]
    fg = None
    fg_set = set()
    for r in range(H):
        for c in range(W):
            if grid[r][c] != bg:
                fg = grid[r][c]
                fg_set.add((r, c))
    if fg is None:
        return [row[:] for row in grid]

    diag_cells = set()
    for r, c in fg_set:
        has_hv = any((r + dr, c + dc) in fg_set for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)])
        has_diag = any((r + dr, c + dc) in fg_set for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)])
        if not has_hv and has_diag:
            diag_cells.add((r, c))

    visited = set()
    chains = []
    for start in diag_cells:
        if start in visited:
            continue
        chain = []
        q = deque([start])
        visited.add(start)
        while q:
            cr, cc = q.popleft()
            chain.append((cr, cc))
            for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nr, nc = cr + dr, cc + dc
                if (nr, nc) in diag_cells and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    q.append((nr, nc))
        chains.append(chain)

    out_fg = fg_set - diag_cells

    for chain in chains:
        chain_len = len(chain)
        for r, c in chain:
            for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nr, nc = r + dr, c + dc
                if (nr, nc) in fg_set and (nr, nc) not in diag_cells:
                    d_row = r - nr
                    d_col = c - nc

                    # Horizontal gap fill
                    cc = nc + d_col
                    gap = []
                    while 0 <= cc < W:
                        if (nr, cc) in out_fg:
                            break
                        gap.append((nr, cc))
                        cc += d_col
                    else:
                        gap = []
                    if gap and len(gap) <= chain_len + 1:
                        for cell in gap:
                            out_fg.add(cell)

                    # Vertical gap fill
                    rr = nr + d_row
                    gap = []
                    while 0 <= rr < H:
                        if (rr, nc) in out_fg:
                            break
                        gap.append((rr, nc))
                        rr += d_row
                    else:
                        gap = []
                    if gap and len(gap) <= chain_len + 1:
                        for cell in gap:
                            out_fg.add(cell)

    # Cleanup: remove dangling stubs off lines that were diagonal connections
    changed = True
    while changed:
        changed = False
        to_remove = set()
        for r, c in out_fg:
            hv_nbrs = [(r + dr, c + dc) for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                       if (r + dr, c + dc) in out_fg]
            if len(hv_nbrs) != 1:
                continue
            if not any((r + dr, c + dc) in diag_cells
                       for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]):
                continue
            nr, nc = hv_nbrs[0]
            if sum(1 for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                   if (nr + dr, nc + dc) in out_fg and (nr + dr, nc + dc) != (r, c)) >= 1:
                to_remove.add((r, c))
        if to_remove:
            out_fg -= to_remove
            changed = True

    out = [[bg] * W for _ in range(H)]
    for r, c in out_fg:
        out[r][c] = fg
    return out


if __name__ == "__main__":
    path = sys.argv[1]
    with open(path) as f:
        data = json.load(f)
    for i, ex in enumerate(data["train"]):
        result = solve(ex["input"])
        ok = result == ex["output"]
        print(f"Train {i}: {'PASS' if ok else 'FAIL'}")
        if not ok:
            for r in range(len(result)):
                for c in range(len(result[0])):
                    if result[r][c] != ex["output"][r][c]:
                        print(f"  ({r},{c}): got {result[r][c]}, expected {ex['output'][r][c]}")
