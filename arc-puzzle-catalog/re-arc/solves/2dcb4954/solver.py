from collections import Counter


def transform(grid):
    H, W = len(grid), len(grid[0])
    bg = Counter(c for row in grid for c in row).most_common(1)[0][0]
    output = [row[:] for row in grid]

    colored = [(r, c, grid[r][c]) for r in range(H) for c in range(W) if grid[r][c] != bg]

    for r0, c0, color in colored:
        dr = 1 if r0 <= (H - 1) / 2 else -1

        for dc in [1, -1]:
            r, c = r0, c0
            cdr = dr
            while True:
                r += cdr
                c += dc
                if c < 0 or c >= W:
                    break
                if r < 0:
                    r = 1
                    cdr = 1
                elif r >= H:
                    r = H - 2
                    cdr = -1
                output[r][c] = color

    return output
