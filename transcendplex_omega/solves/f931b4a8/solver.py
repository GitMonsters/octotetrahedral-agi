import json
import numpy as np

def solve(input_grid):
    grid = np.array(input_grid)
    H, W = grid.shape
    hh, hw = (H // 2, W // 2)
    TL = grid[:hh, :hw]
    TR = grid[:hh, hw:]
    BL = grid[hh:, :hw]
    BR = grid[hh:, hw:]
    tl_count = int(np.sum(TL != 0))
    tr_count = int(np.sum(TR != 0))
    br_nonblack = int(np.sum(BR != 0))
    if br_nonblack > 0:
        pattern = BR
        fill = BL
    else:
        pattern = BL
        fill = BR
    pH, pW = pattern.shape
    out_rows = tl_count if tl_count > 0 else pH
    out_cols = tr_count if tr_count > 0 else pW
    output = np.zeros((out_rows, out_cols), dtype=int)
    for r in range(out_rows):
        for c in range(out_cols):
            pr = r % pH
            pc = c % pW
            if pattern[pr][pc] != 0:
                output[r][c] = pattern[pr][pc]
            else:
                tr = r // pH
                tc = c // pW
                fr = tr % fill.shape[0]
                fc = tc % fill.shape[1]
                output[r][c] = fill[fr][fc]
    return output.tolist()
EMOJI = ['⬛', '🔴', '🟢', '💚', '🟡', '⬜', '🟣', '🟠', '🔷', '🟫']
DIR = '/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI-2/data/evaluation/'
