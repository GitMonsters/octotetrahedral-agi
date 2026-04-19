"""
Solver for ARC task 097428fa.

Rule: The input contains a periodic tiled pattern partially obscured by a
single-color "blob". The output is the fully-revealed periodic pattern.

Algorithm:
1. For each candidate blob color and tile size (ph, pw):
   - Mask cells of the blob color
   - Check if unmasked cells are consistent with a (ph, pw) periodic tile
   - If yes, reconstruct the full tile and apply it to the grid
2. Return the result using the smallest valid tile found.
"""
import numpy as np


def transform(grid):
    inp = np.array(grid, dtype=int)
    H, W = inp.shape
    colors = set(inp.flatten().tolist())

    # Try all (blob_color, ph, pw) sorted by tile area for smallest-first
    candidates = []
    for blob_c in colors:
        for ph in range(2, H):
            for pw in range(2, W):
                candidates.append((ph * pw, blob_c, ph, pw))
    candidates.sort()

    for _, blob_c, ph, pw in candidates:
        mask = (inp != blob_c)
        tile = np.full((ph, pw), -1, dtype=int)
        valid = True

        for tr in range(ph):
            if not valid:
                break
            for tc in range(pw):
                val = -1
                consistent = True
                for r in range(tr, H, ph):
                    for c in range(tc, W, pw):
                        if mask[r, c]:
                            if val == -1:
                                val = inp[r, c]
                            elif inp[r, c] != val:
                                consistent = False
                                break
                    if not consistent:
                        break

                if not consistent:
                    valid = False
                    break

                tile[tr, tc] = val if val != -1 else blob_c

        if not valid:
            continue

        # Build tiled output and check disagreements are single-color
        tiled = np.tile(tile, ((H + ph - 1) // ph, (W + pw - 1) // pw))[:H, :W]
        disagree = inp != tiled
        n_disagree = disagree.sum()

        if n_disagree > 0:
            disagree_vals = set(inp[disagree].tolist())
            if len(disagree_vals) == 1:
                return tiled.tolist()

    return grid
