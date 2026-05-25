"""Solver for 13e47133 — fill regions with concentric rectangles"""
import json, numpy as np
from typing import List
from collections import Counter, deque

def solve(grid: List[List[int]]) -> List[List[int]]:
    g = np.array(grid)
    H, W = g.shape
    result = g.copy()
    
    flat = g.flatten()
    cc = Counter(flat)
    bg_color = int(cc.most_common(1)[0][0])
    
    non_bg = [int(c) for c in cc if c != bg_color]
    divider_color = None
    max_seg = 0
    for color in non_bg:
        mask = (g == color)
        for r in range(H):
            seg = 0
            for c in range(W):
                if mask[r][c]: seg += 1
                else:
                    if seg > max_seg: max_seg = seg; divider_color = color
                    seg = 0
            if seg > max_seg: max_seg = seg; divider_color = color
        for c in range(W):
            seg = 0
            for r in range(H):
                if mask[r][c]: seg += 1
                else:
                    if seg > max_seg: max_seg = seg; divider_color = color
                    seg = 0
            if seg > max_seg: max_seg = seg; divider_color = color
    
    div_mask = (g == divider_color)
    
    region_map = np.full((H, W), -1, dtype=int)
    regions = []
    rid = 0
    for r in range(H):
        for c in range(W):
            if not div_mask[r][c] and region_map[r][c] == -1:
                cells = set()
                q = deque([(r, c)])
                region_map[r][c] = rid
                while q:
                    cr, ccc = q.popleft()
                    cells.add((cr, ccc))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, ccc+dc
                        if 0<=nr<H and 0<=nc<W and not div_mask[nr][nc] and region_map[nr][nc]==-1:
                            region_map[nr][nc] = rid
                            q.append((nr, nc))
                regions.append(cells)
                rid += 1
    
    cut_rows = set()
    cut_cols = set()
    for r in range(H):
        seg_start = None
        for c in range(W+1):
            if c < W and div_mask[r][c]:
                if seg_start is None: seg_start = c
            else:
                if seg_start is not None and c - seg_start >= 2:
                    cut_rows.add(r)
                seg_start = None
    for c in range(W):
        seg_start = None
        for r in range(H+1):
            if r < H and div_mask[r][c]:
                if seg_start is None: seg_start = r
            else:
                if seg_start is not None and r - seg_start >= 2:
                    cut_cols.add(c)
                seg_start = None
    
    row_bounds = sorted(set([-1] + list(cut_rows) + [H]))
    col_bounds = sorted(set([-1] + list(cut_cols) + [W]))
    
    candidate_rects = []
    for i in range(len(row_bounds)):
        for j in range(i+1, len(row_bounds)):
            r_s = row_bounds[i] + 1
            r_e = row_bounds[j] - 1
            if r_s > r_e: continue
            for k in range(len(col_bounds)):
                for l in range(k+1, len(col_bounds)):
                    c_s = col_bounds[k] + 1
                    c_e = col_bounds[l] - 1
                    if c_s > c_e: continue
                    ok = True
                    for rr in range(r_s, r_e+1):
                        for ccc in range(c_s, c_e+1):
                            if div_mask[rr][ccc]:
                                ok = False; break
                        if not ok: break
                    if ok:
                        candidate_rects.append((r_s, r_e, c_s, c_e))
    
    for region_cells in regions:
        dots = []
        for r, c in region_cells:
            val = int(g[r][c])
            if val != bg_color and val != divider_color:
                dots.append((r, c, val))
        
        if not dots: continue
        
        region_rects = []
        for rect in candidate_rects:
            r_s, r_e, c_s, c_e = rect
            has_cell = False
            for r, c in region_cells:
                if r_s <= r <= r_e and c_s <= c <= c_e:
                    has_cell = True; break
            if not has_cell: continue
            region_rects.append(rect)
        
        if not region_rects: continue
        
        # For each dot, use the MAX distance across all enclosing rects
        # to avoid small sub-rects distorting the true distance
        dot_dists_list = []
        for r, c, col in dots:
            best_d = -1
            for rect in region_rects:
                r_s, r_e, c_s, c_e = rect
                if r_s <= r <= r_e and c_s <= c <= c_e:
                    d = min(r - r_s, r_e - r, c - c_s, c_e - c)
                    if d > best_d:
                        best_d = d
            if best_d >= 0:
                dot_dists_list.append((best_d, col))
        
        if not dot_dists_list: continue
        
        # Determine color cycle
        color_by_dist = {}
        for d, col in sorted(dot_dists_list):
            if d not in color_by_dist:
                color_by_dist[d] = col
        
        sorted_dists = sorted(color_by_dist.keys())
        n_colors = len(sorted_dists)
        
        if n_colors == 1 and sorted_dists[0] == 0:
            for r, c in region_cells:
                result[r][c] = color_by_dist[0]
            continue
        
        # Build color cycle
        if n_colors >= 2 and sorted_dists == list(range(n_colors)):
            cycle_len = n_colors
            cycle_colors = [color_by_dist[i] for i in range(n_colors)]
        elif n_colors == 1:
            d0 = sorted_dists[0]
            cycle_len = 2
            if d0 % 2 == 0:
                cycle_colors = [color_by_dist[d0], bg_color]
            else:
                cycle_colors = [bg_color, color_by_dist[d0]]
        else:
            cycle_len = 2
            c0 = color_by_dist[sorted_dists[0]]
            c1 = color_by_dist[sorted_dists[1]]
            if sorted_dists[0] % 2 == 0:
                cycle_colors = [c0, c1]
            else:
                cycle_colors = [c1, c0]
        
        # Fill cells
        for r, c in region_cells:
            max_d = 0
            for rect in region_rects:
                r_s, r_e, c_s, c_e = rect
                if r_s <= r <= r_e and c_s <= c <= c_e:
                    d = min(r - r_s, r_e - r, c - c_s, c_e - c)
                    max_d = max(max_d, d)
            result[r][c] = cycle_colors[max_d % cycle_len]
    
    return result.tolist()

if __name__ == '__main__':
    import sys
    with open(sys.argv[1]) as f: task = json.load(f)
    for i, ex in enumerate(task['train']):
        print(f"Train {i}: {'PASS ✓' if solve(ex['input'])==ex['output'] else 'FAIL'}")
