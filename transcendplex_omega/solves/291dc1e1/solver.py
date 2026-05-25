"""Solver for 291dc1e1 — extract blocks, rotate if needed, stack centered"""
import json
from typing import List
import numpy as np
from collections import deque

def solve(grid: List[List[int]]) -> List[List[int]]:
    g = np.array(grid)
    H, W = g.shape
    BG = 8
    
    # Find corner (value 0 at grid corner)
    corner = None
    for cr, cc in [(0,0), (0,W-1), (H-1,0), (H-1,W-1)]:
        if g[cr, cc] == 0:
            corner = (cr, cc)
            break
    cr, cc = corner
    
    # Determine which border has blue(1) and which has red(2)
    # Row of corner
    row_test_c = 1 if cc == 0 else W-2
    row_color = int(g[cr, row_test_c])
    col_test_r = 1 if cr == 0 else H-2
    col_color = int(g[col_test_r, cc])
    
    blue_on_row = (row_color == 1)  # blue border along the corner's row
    # If blue is on the row → primary axis is horizontal (along columns)
    # If blue is on the column → primary axis is vertical (along rows)
    
    # Interior bounds (exclude the border row and column)
    ir_s = 1 if cr == 0 else 0
    ir_e = H if cr == 0 else H - 1
    ic_s = 1 if cc == 0 else 0
    ic_e = W if cc == 0 else W - 1
    
    interior = g[ir_s:ir_e, ic_s:ic_e]
    iH, iW = interior.shape
    
    # Find blocks: connected non-BG regions in interior
    visited = np.zeros((iH, iW), dtype=bool)
    blocks = []
    
    for r in range(iH):
        for c in range(iW):
            if interior[r,c] != BG and not visited[r,c]:
                # BFS to find connected component
                q = deque([(r, c)])
                visited[r, c] = True
                cells = [(r, c)]
                while q:
                    qr, qc = q.popleft()
                    for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
                        nr, nc = qr+dr, qc+dc
                        if 0<=nr<iH and 0<=nc<iW and not visited[nr,nc] and interior[nr,nc] != BG:
                            visited[nr,nc] = True
                            q.append((nr,nc))
                            cells.append((nr,nc))
                
                min_r = min(r for r,c in cells)
                max_r = max(r for r,c in cells)
                min_c = min(c for r,c in cells)
                max_c = max(c for r,c in cells)
                
                data = interior[min_r:max_r+1, min_c:max_c+1].copy()
                blocks.append({
                    'data': data,
                    'min_r': min_r, 'max_r': max_r,
                    'min_c': min_c, 'max_c': max_c,
                })
    
    # Determine reading direction
    # Primary: along blue border (away from corner)
    # Secondary: along red border (away from corner)
    
    if blue_on_row:
        # Primary = horizontal (columns), secondary = vertical (rows)
        # Primary direction: left→right if corner on left, right→left if corner on right
        primary_reverse = (cc != 0)  # if corner on right, read right to left
        secondary_reverse = (cr != 0)  # if corner on bottom, read bottom to top
        
        # Group blocks by row band (secondary groups = row groups)
        # Sort blocks within each row group by column (primary order)
        row_groups = _group_by_rows(blocks, iH)
        if secondary_reverse:
            row_groups.reverse()
        
        ordered_blocks = []
        for group in row_groups:
            group.sort(key=lambda b: b['min_c'], reverse=primary_reverse)
            ordered_blocks.extend(group)
        
        # No rotation needed (blocks are already horizontal)
        block_datas = [b['data'] for b in ordered_blocks]
    else:
        # Primary = vertical (rows), secondary = horizontal (columns)
        primary_reverse = (cr != 0)  # if corner at bottom, read bottom to top
        secondary_reverse = (cc != 0)  # if corner on right, read right to left
        
        # Group blocks by column band (secondary groups = column groups)
        col_groups = _group_by_cols(blocks, iW)
        if secondary_reverse:
            col_groups.reverse()
        
        ordered_blocks = []
        for group in col_groups:
            group.sort(key=lambda b: b['min_r'], reverse=primary_reverse)
            ordered_blocks.extend(group)
        
        # Rotate blocks 90° CCW (since they're vertical, need to make horizontal)
        block_datas = [np.rot90(b['data'], 1) for b in ordered_blocks]
    
    # Stack all blocks vertically, centered within max width
    max_w = max(d.shape[1] for d in block_datas)
    
    rows = []
    for data in block_datas:
        h, w = data.shape
        pad_left = (max_w - w) // 2
        pad_right = max_w - w - pad_left
        for r in range(h):
            row = [BG] * pad_left + list(data[r]) + [BG] * pad_right
            rows.append(row)
    
    return rows

def _group_by_rows(blocks, iH):
    """Group blocks into row bands (separated by all-BG rows)."""
    if not blocks:
        return []
    # Find all unique row ranges
    row_ranges = []
    for b in blocks:
        row_ranges.append((b['min_r'], b['max_r']))
    
    # Cluster overlapping row ranges
    row_ranges.sort()
    clusters = []
    cur_start, cur_end = row_ranges[0]
    for s, e in row_ranges[1:]:
        if s <= cur_end + 1:
            cur_end = max(cur_end, e)
        else:
            clusters.append((cur_start, cur_end))
            cur_start, cur_end = s, e
    clusters.append((cur_start, cur_end))
    
    # Assign blocks to clusters
    groups = [[] for _ in clusters]
    for b in blocks:
        for i, (cs, ce) in enumerate(clusters):
            if b['min_r'] >= cs and b['max_r'] <= ce:
                groups[i].append(b)
                break
    
    return groups

def _group_by_cols(blocks, iW):
    """Group blocks into column bands."""
    if not blocks:
        return []
    col_ranges = []
    for b in blocks:
        col_ranges.append((b['min_c'], b['max_c']))
    
    col_ranges_sorted = sorted(set(col_ranges))
    clusters = []
    cur_start, cur_end = col_ranges_sorted[0]
    for s, e in col_ranges_sorted[1:]:
        if s <= cur_end + 1:
            cur_end = max(cur_end, e)
        else:
            clusters.append((cur_start, cur_end))
            cur_start, cur_end = s, e
    clusters.append((cur_start, cur_end))
    
    groups = [[] for _ in clusters]
    for b in blocks:
        for i, (cs, ce) in enumerate(clusters):
            if b['min_c'] >= cs and b['max_c'] <= ce:
                groups[i].append(b)
                break
    
    return groups

if __name__ == '__main__':
    import sys
    with open(sys.argv[1]) as f: task = json.load(f)
    for i, ex in enumerate(task['train']):
        res = solve(ex['input'])
        status = 'PASS ✓' if res == ex['output'] else 'FAIL'
        print(f"Train {i}: {status}")
        if status == 'FAIL':
            out = ex['output']
            print(f"  Expected: {len(out)}×{len(out[0])}, Got: {len(res)}×{len(res[0]) if res else 0}")
            for r in range(min(len(res), len(out))):
                if res[r] != out[r]:
                    print(f"  Row {r} diff:")
                    print(f"    Exp: {out[r]}")
                    print(f"    Got: {res[r]}")
                    break
