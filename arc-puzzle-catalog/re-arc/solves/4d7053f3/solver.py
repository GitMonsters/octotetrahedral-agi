"""Solver for ARC task 4d7053f3.

The input is a tiled pattern where a rectangular region has been replaced
with a uniform fill color. The output is the restored content of that
masked region, reconstructed from the visible tile repetitions.
"""

import numpy as np


def find_largest_uniform_rect(inp):
    """Find the largest rectangle where all cells share one value."""
    H, W = inp.shape
    best_area = 0
    best_rect = None

    for color in range(10):
        right = np.zeros((H, W), dtype=int)
        for r in range(H):
            count = 0
            for c in range(W - 1, -1, -1):
                if inp[r][c] == color:
                    count += 1
                else:
                    count = 0
                right[r][c] = count

        for c0 in range(W):
            for r0 in range(H):
                min_width = right[r0][c0]
                if min_width == 0:
                    continue
                for r1 in range(r0, H):
                    min_width = min(min_width, right[r1][c0])
                    if min_width == 0:
                        break
                    area = (r1 - r0 + 1) * min_width
                    if area > best_area:
                        best_area = area
                        best_rect = (color, r0, r1, c0, c0 + min_width - 1)
    return best_rect


def find_period(inp, is_masked, axis):
    """Find the smallest tiling period along the given axis."""
    H, W = inp.shape
    limit = H if axis == 0 else W

    for p in range(1, limit):
        valid = True
        count = 0
        for r in range(H):
            for c in range(W):
                nr = r + p if axis == 0 else r
                nc = c + p if axis == 1 else c
                if nr >= H or nc >= W:
                    continue
                if is_masked[r][c] or is_masked[nr][nc]:
                    continue
                count += 1
                if inp[r][c] != inp[nr][nc]:
                    valid = False
                    break
            if not valid:
                break
        if valid and count > 0:
            return p
    return None


def transform(grid):
    inp = np.array(grid)
    H, W = inp.shape

    mcol, mr0, mr1, mc0, mc1 = find_largest_uniform_rect(inp)

    is_masked = np.zeros((H, W), dtype=bool)
    is_masked[mr0:mr1 + 1, mc0:mc1 + 1] = True

    pr = find_period(inp, is_masked, axis=0)
    pc = find_period(inp, is_masked, axis=1)

    result = inp.copy()
    for r in range(mr0, mr1 + 1):
        for c in range(mc0, mc1 + 1):
            found = False
            for kr in range(-H // pr - 1, H // pr + 2):
                for kc in range(-W // pc - 1, W // pc + 2):
                    nr = r + kr * pr
                    nc = c + kc * pc
                    if 0 <= nr < H and 0 <= nc < W and not is_masked[nr][nc]:
                        result[r][c] = inp[nr][nc]
                        found = True
                        break
                if found:
                    break

    return result[mr0:mr1 + 1, mc0:mc1 + 1].tolist()
