"""
ARC puzzle 4afd0e50

Transformation rule:
  The input contains a core block of cells already 4-fold rotationally
  symmetric, plus scattered "fragment" cells representing only 1/4 of their
  orbit.  The output completes 4-fold (90°/180°/270°) rotational symmetry
  for every non-background cell.

Finding the rotation center:
  Centre (r0, c0) is stored as integers (R, C) = (2*r0, 2*c0).
  We pick the (R, C) that maximises the number of 180°-symmetric same-colour
  pairs in the input: for cell (r,c)=v, its 180° partner is (R-r, C-c).
  The core block contributes the most pairs; isolated scattered cells don't
  pair up, so they don't shift the winning centre.

Rotation formulas (centre = R/2, C/2):
  180°  : (r, c) → (R-r, C-c)
  90°CW : (r, c) → ((R+2c-C)//2, (R+C-2r)//2)   [valid when R+C is even]
  270°CW: (r, c) → ((R+C-2c)//2, (C+2r-R)//2)
"""


def transform(grid):
    H = len(grid)
    W = len(grid[0])
    bg = 7

    cells = {
        (r, c): grid[r][c]
        for r in range(H)
        for c in range(W)
        if grid[r][c] != bg
    }
    if not cells:
        return [row[:] for row in grid]

    # Find rotation centre (R/2, C/2) by maximising 180° symmetric pairs
    best_score, best_R, best_C = -1, 0, 0
    for R in range(2 * H):
        for C in range(2 * W):
            score = sum(
                1 for (r, c), v in cells.items()
                if cells.get((R - r, C - c)) == v
            )
            if score > best_score:
                best_score, best_R, best_C = score, R, C

    output = [row[:] for row in grid]
    R, C = best_R, best_C

    def place(r, c, v):
        if 0 <= r < H and 0 <= c < W:
            output[r][c] = v

    for (r, c), v in cells.items():
        place(R - r, C - c, v)                              # 180°
        if (R + C) % 2 == 0:
            place((R + 2*c - C) // 2, (R + C - 2*r) // 2, v)  # 90° CW
            place((R + C - 2*c) // 2, (C + 2*r - R) // 2, v)  # 270° CW

    return output
