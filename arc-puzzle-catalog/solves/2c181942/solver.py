"""Solver for 2c181942 — expand cross arms with matching-color shapes"""
import json
from typing import List
import numpy as np

def solve(grid: List[List[int]]) -> List[List[int]]:
    g = np.array(grid)
    H, W = g.shape
    bg = 8
    
    # Find the 4-arm cross pattern
    cross_pos = None
    arms = None
    for r in range(H-3):
        for c in range(W-3):
            center = g[r:r+4, c:c+4]
            if (center[1,1] == bg and center[1,2] == bg and 
                center[2,1] == bg and center[2,2] == bg and
                center[0,0] == bg and center[0,3] == bg and 
                center[3,0] == bg and center[3,3] == bg):
                top, bottom = center[0,1], center[3,1]
                left, right = center[1,0], center[1,3]
                if (top == center[0,2] and top != bg and
                    bottom == center[3,2] and bottom != bg and
                    left == center[2,0] and left != bg and
                    right == center[2,3] and right != bg and
                    len({int(top), int(bottom), int(left), int(right)}) == 4):
                    cross_pos = (r, c)
                    arms = {'top': int(top), 'bottom': int(bottom),
                            'left': int(left), 'right': int(right)}
                    break
        if cross_pos:
            break
    
    cr, cc = cross_pos
    cross_cells = set()
    for dr in range(4):
        for dc in range(4):
            if g[cr+dr, cc+dc] != bg:
                cross_cells.add((cr+dr, cc+dc))
    
    from collections import defaultdict
    color_cells = defaultdict(list)
    for r in range(H):
        for c in range(W):
            if g[r,c] != bg and (r,c) not in cross_cells:
                color_cells[int(g[r,c])].append((r,c))
    
    out = np.full_like(g, bg)
    for (r,c) in cross_cells:
        out[r,c] = g[r,c]
    
    arm_meta = {
        'top':    {'arm_rows': (cr, cr), 'arm_cols': (cc+1, cc+2)},
        'bottom': {'arm_rows': (cr+3, cr+3), 'arm_cols': (cc+1, cc+2)},
        'left':   {'arm_rows': (cr+1, cr+2), 'arm_cols': (cc, cc)},
        'right':  {'arm_rows': (cr+1, cr+2), 'arm_cols': (cc+3, cc+3)},
    }
    
    for direction, arm_color in arms.items():
        if arm_color not in color_cells or not color_cells[arm_color]:
            continue
        
        cells = color_cells[arm_color]
        min_r = min(r for r,c in cells)
        max_r = max(r for r,c in cells)
        min_c = min(c for r,c in cells)
        max_c = max(c for r,c in cells)
        
        shape = np.zeros((max_r-min_r+1, max_c-min_c+1), dtype=int)
        for r,c in cells:
            shape[r-min_r, c-min_c] = 1
        
        meta = arm_meta[direction]
        arm_r1, arm_r2 = meta['arm_rows']
        arm_c1, arm_c2 = meta['arm_cols']
        
        # Try all 4 rotations, pick the one with matching connecting face (prefer longest face)
        candidates = []
        for k in range(4):
            rot = np.rot90(shape, k)
            rh, rw = rot.shape
            
            if direction in ('left', 'right'):
                arm_center = (arm_r1 + arm_r2) / 2.0
                ext_r_start = round(arm_center - (rh - 1) / 2.0)
                
                face = rot[:, -1] if direction == 'left' else rot[:, 0]
                
                expected = np.zeros(rh, dtype=int)
                for ar in range(arm_r1, arm_r2 + 1):
                    idx = ar - ext_r_start
                    if 0 <= idx < rh:
                        expected[idx] = 1
                
                if np.array_equal(face, expected):
                    candidates.append((len(face), k, rot, ext_r_start))
            else:  # top or bottom
                arm_center = (arm_c1 + arm_c2) / 2.0
                ext_c_start = round(arm_center - (rw - 1) / 2.0)
                
                face = rot[-1, :] if direction == 'top' else rot[0, :]
                
                expected = np.zeros(rw, dtype=int)
                for ac in range(arm_c1, arm_c2 + 1):
                    idx = ac - ext_c_start
                    if 0 <= idx < rw:
                        expected[idx] = 1
                
                if np.array_equal(face, expected):
                    candidates.append((len(face), k, rot, ext_c_start))
        
        if not candidates:
            continue
        
        # Pick candidate with longest face
        candidates.sort(key=lambda x: -x[0])
        _, k, rot, start = candidates[0]
        rh, rw = rot.shape
        
        if direction in ('left', 'right'):
            ext_r_start = start
            if direction == 'left':
                ext_c_start = arm_c1 - rw
            else:
                ext_c_start = arm_c2 + 1
            
            for r in range(rh):
                for c in range(rw):
                    if rot[r, c]:
                        pr, pc = ext_r_start + r, ext_c_start + c
                        if 0 <= pr < H and 0 <= pc < W:
                            out[pr, pc] = arm_color
        else:
            ext_c_start = start
            if direction == 'top':
                ext_r_start = arm_r1 - rh
            else:
                ext_r_start = arm_r2 + 1
            
            for r in range(rh):
                for c in range(rw):
                    if rot[r, c]:
                        pr, pc = ext_r_start + r, ext_c_start + c
                        if 0 <= pr < H and 0 <= pc < W:
                            out[pr, pc] = arm_color
    
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
            diff = np.argwhere(got != out)
            if len(diff) > 0:
                print(f"  Diffs ({len(diff)}): first at {diff[0]}: exp {out[tuple(diff[0])]}, got {got[tuple(diff[0])]}")
