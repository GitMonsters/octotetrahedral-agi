from collections import Counter

def transform(grid):
    H, W = len(grid), len(grid[0])
    bg = Counter(v for r in grid for v in r).most_common(1)[0][0]

    blocks = []
    used = set()
    for r in range(H - 1):
        for c in range(W - 1):
            if (r, c) in used:
                continue
            vals = [grid[r][c], grid[r][c+1], grid[r+1][c], grid[r+1][c+1]]
            if all(v != bg for v in vals):
                blocks.append((r, c))
                used.update([(r, c), (r, c+1), (r+1, c), (r+1, c+1)])

    out = [row[:] for row in grid]

    for br, bc in blocks:
        tl = grid[br][bc]
        tr = grid[br][bc+1]
        bl = grid[br+1][bc]
        brc = grid[br+1][bc+1]

        emissions = [
            (tl, +2, +2),
            (tr, +2, -2),
            (bl, -2, +2),
            (brc, -2, -2),
        ]

        for val, dr, dc in emissions:
            nr, nc = br + dr, bc + dc
            for r2 in range(nr, nr + 2):
                for c2 in range(nc, nc + 2):
                    if 0 <= r2 < H and 0 <= c2 < W and out[r2][c2] == bg:
                        out[r2][c2] = val

    return out
