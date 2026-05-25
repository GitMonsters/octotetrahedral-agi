"""Solver for 2ba387bc — pair frame rectangles with solid rectangles"""
import json
from typing import List
import numpy as np

def solve(grid: List[List[int]]) -> List[List[int]]:
    g = np.array(grid)
    H, W = g.shape
    
    # Find all 4x4 rectangular blocks (non-zero)
    visited = np.zeros_like(g, dtype=bool)
    blocks = []
    
    for r in range(H-3):
        for c in range(W-3):
            if g[r,c] != 0 and not visited[r,c]:
                # Check if this is the top-left of a 4x4 block
                block = g[r:r+4, c:c+4]
                color = g[r,c]
                
                # Check if it's a solid block
                if np.all(block == color):
                    blocks.append({'type': 'solid', 'color': int(color), 'row': r, 'col': c,
                                   'data': block.copy()})
                    visited[r:r+4, c:c+4] = True
                    continue
                
                # Check if it's a frame (border = color, interior = 0)
                border_ok = (np.all(block[0,:] == color) and np.all(block[3,:] == color) and
                            np.all(block[:,0] == color) and np.all(block[:,3] == color))
                interior_ok = (block[1,1] == 0 and block[1,2] == 0 and 
                              block[2,1] == 0 and block[2,2] == 0)
                
                if border_ok and interior_ok:
                    blocks.append({'type': 'frame', 'color': int(color), 'row': r, 'col': c,
                                   'data': block.copy()})
                    visited[r:r+4, c:c+4] = True
    
    # Sort frames and solids by position (row first, then col)
    frames = sorted([b for b in blocks if b['type'] == 'frame'], key=lambda b: (b['row'], b['col']))
    solids = sorted([b for b in blocks if b['type'] == 'solid'], key=lambda b: (b['row'], b['col']))
    
    # Pair them: 1st frame with 1st solid, etc.
    n_pairs = max(len(frames), len(solids))
    
    # Build output: each pair is a row of 8 cols (frame 4x4 + solid 4x4)
    out = np.zeros((4 * n_pairs, 8), dtype=int)
    
    for i in range(n_pairs):
        r_start = i * 4
        if i < len(frames):
            out[r_start:r_start+4, 0:4] = frames[i]['data']
        # else: leave as 0 (empty frame)
        
        if i < len(solids):
            out[r_start:r_start+4, 4:8] = solids[i]['data']
        # else: leave as 0 (empty solid)
    
    return out.tolist()

if __name__ == '__main__':
    import sys
    with open(sys.argv[1]) as f: task = json.load(f)
    for i, ex in enumerate(task['train']):
        res = solve(ex['input'])
        status = 'PASS ✓' if res == ex['output'] else 'FAIL'
        print(f"Train {i}: {status}")
        if status == 'FAIL':
            out = np.array(ex['output'])
            got = np.array(res)
            print(f"  Expected: {out.shape}, Got: {got.shape}")
            if out.shape == got.shape:
                diff = np.argwhere(got != out)
                print(f"  Diffs: {len(diff)}")
                if len(diff) > 0: print(f"  First: {diff[0]}")
