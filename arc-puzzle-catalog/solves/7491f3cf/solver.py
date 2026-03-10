"""
Puzzle 7491f3cf: Panel combination guided by dividing line.

The 7x25 grid has 4 panels of 5x5 separated by a border color.
Panel 1 defines a dividing line that splits the 5x5 area into two regions.
The smaller region is filled with Panel 2's pattern, the larger with Panel 3's.
On the dividing line itself, Panel 2 takes priority if non-background, else Panel 3.
Panel 4 (initially blank) gets the combined result.
"""
from collections import deque
import json
import sys


def solve(grid: list[list[int]]) -> list[list[int]]:
    H, W = 5, 5
    border = grid[0][0]
    bg = grid[1][19]  # Panel 4 background

    # Extract panels (rows 1-5, different column offsets)
    p1 = [[grid[r + 1][c + 1] for c in range(W)] for r in range(H)]
    p2 = [[grid[r + 1][c + 7] for c in range(W)] for r in range(H)]
    p3 = [[grid[r + 1][c + 13] for c in range(W)] for r in range(H)]

    # Panel 1 mask: colored (non-bg) cells form the dividing line
    mask = [[p1[r][c] != bg for c in range(W)] for r in range(H)]

    # Flood-fill to find connected components of non-line cells
    comp = [[-1] * W for _ in range(H)]
    cid = 0
    sizes: dict[int, int] = {}
    for sr in range(H):
        for sc in range(W):
            if mask[sr][sc] or comp[sr][sc] != -1:
                continue
            q = deque([(sr, sc)])
            comp[sr][sc] = cid
            sz = 0
            while q:
                r, c = q.popleft()
                sz += 1
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and not mask[nr][nc] and comp[nr][nc] == -1:
                        comp[nr][nc] = cid
                        q.append((nr, nc))
            sizes[cid] = sz
            cid += 1

    # Smaller component gets Panel 2, larger gets Panel 3
    sorted_c = sorted(sizes.keys(), key=lambda k: sizes[k])
    p2_comp = sorted_c[0]
    p3_comp = sorted_c[1]

    output = [row[:] for row in grid]

    for r in range(H):
        for c in range(W):
            if not mask[r][c]:
                # Cell belongs to a region
                if comp[r][c] == p2_comp:
                    output[r + 1][c + 19] = p2[r][c]
                else:
                    output[r + 1][c + 19] = p3[r][c]
            else:
                # Dividing line cell: check which regions it borders
                b2 = any(
                    0 <= r + dr < H and 0 <= c + dc < W
                    and not mask[r + dr][c + dc]
                    and comp[r + dr][c + dc] == p2_comp
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                )
                b3 = any(
                    0 <= r + dr < H and 0 <= c + dc < W
                    and not mask[r + dr][c + dc]
                    and comp[r + dr][c + dc] == p3_comp
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                )
                if b2 and b3:
                    output[r + 1][c + 19] = p2[r][c] if p2[r][c] != bg else p3[r][c]
                elif b2:
                    output[r + 1][c + 19] = p2[r][c]
                elif b3:
                    output[r + 1][c + 19] = p3[r][c]
                else:
                    output[r + 1][c + 19] = bg

    return output


if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        data = json.load(f)

    for i, ex in enumerate(data["train"]):
        result = solve(ex["input"])
        if result == ex["output"]:
            print(f"Train {i}: PASS")
        else:
            print(f"Train {i}: FAIL")
            for r in range(len(result)):
                for c in range(len(result[0])):
                    if result[r][c] != ex["output"][r][c]:
                        print(f"  ({r},{c}): got {result[r][c]}, expected {ex['output'][r][c]}")
