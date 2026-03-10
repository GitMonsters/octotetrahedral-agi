import json
import numpy as np

def solve(input_grid):
    grid = np.array(input_grid)
    H, W = grid.shape
    hh, hw = H // 2, W // 2
    
    TL = grid[:hh, :hw]
    TR = grid[:hh, hw:]
    BL = grid[hh:, :hw]
    BR = grid[hh:, hw:]
    
    # Count non-black cells
    tl_count = int(np.sum(TL != 0))
    tr_count = int(np.sum(TR != 0))
    
    # Determine pattern and fill
    br_nonblack = int(np.sum(BR != 0))
    
    if br_nonblack > 0:
        pattern = BR
        fill = BL
    else:
        pattern = BL
        fill = BR
    
    pH, pW = pattern.shape
    
    # Output dimensions
    out_rows = tl_count if tl_count > 0 else pH
    out_cols = tr_count if tr_count > 0 else pW
    
    # Build output
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

# Test
EMOJI = ['⬛','🔴','🟢','💚','🟡','⬜','🟣','🟠','🔷','🟫']
DIR = '/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI-2/data/evaluation/'

with open(f'{DIR}f931b4a8.json') as f:
    task = json.load(f)

all_pass = True
for i, ex in enumerate(task['train']):
    result = solve(ex['input'])
    expected = ex['output']
    match = result == expected
    print(f"Train {i}: {'PASS' if match else 'FAIL'}")
    if not match:
        all_pass = False
        print("Expected:")
        for r in expected:
            print("".join(EMOJI[v] for v in r))
        print("Got:")
        for r in result:
            print("".join(EMOJI[v] for v in r))

print(f"\nAll pass: {all_pass}")
