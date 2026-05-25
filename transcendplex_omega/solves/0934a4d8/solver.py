"""Solver for 0934a4d8 — reconstruct occluded area using context matching"""
import json, numpy as np
from typing import List
from collections import Counter

def solve(grid: List[List[int]]) -> List[List[int]]:
    g = np.array(grid)
    H, W = g.shape
    
    mask8 = (g == 8)
    rows8 = np.where(mask8.any(axis=1))[0]
    cols8 = np.where(mask8.any(axis=0))[0]
    r_min, r_max = int(rows8.min()), int(rows8.max())
    c_min, c_max = int(cols8.min()), int(cols8.max())
    
    oh = r_max - r_min + 1
    ow = c_max - c_min + 1
    result = [[0]*ow for _ in range(oh)]
    
    # Precompute: for each row, its context (values at columns outside 8-rect)
    ctx_cols = [cc for cc in range(W) if cc < c_min or cc > c_max]
    ctx_rows = [rr for rr in range(H) if rr < r_min or rr > r_max]
    
    for r in range(r_min, r_max+1):
        for c in range(c_min, c_max+1):
            # Row-based prediction
            row_ctx = [int(g[r][cc]) for cc in ctx_cols]
            row_scores = []
            for cr in range(H):
                if mask8[cr][c]:
                    continue
                score = sum(1 for i, cc in enumerate(ctx_cols) if g[cr][cc] == row_ctx[i])
                row_scores.append((-score, cr))  # negative for min-heap behavior
            row_scores.sort()
            row_pred = int(g[row_scores[0][1]][c]) if row_scores else -1
            row_score = -row_scores[0][0] if row_scores else 0
            
            # Column-based prediction
            col_ctx = [int(g[rr][c]) for rr in ctx_rows]
            col_scores = []
            for cc_cand in range(W):
                if mask8[r][cc_cand]:
                    continue
                score = sum(1 for i, rr in enumerate(ctx_rows) if g[rr][cc_cand] == col_ctx[i])
                col_scores.append((-score, cc_cand))
            col_scores.sort()
            col_pred = int(g[r][col_scores[0][1]]) if col_scores else -1
            col_score = -col_scores[0][0] if col_scores else 0
            
            # Check if either dimension found a perfect (twin) match
            row_perfect = (row_score == len(ctx_cols))
            col_perfect = (col_score == len(ctx_rows))
            
            if row_perfect or col_perfect:
                # At least one twin found — use higher-confidence prediction
                if row_score >= col_score:
                    result[r-r_min][c-c_min] = row_pred
                else:
                    result[r-r_min][c-c_min] = col_pred
            elif c < H and r < W and not mask8[c][r]:
                # No twin in either dimension — use transpose symmetry
                result[r-r_min][c-c_min] = int(g[c][r])
            else:
                if row_score >= col_score:
                    result[r-r_min][c-c_min] = row_pred
                else:
                    result[r-r_min][c-c_min] = col_pred
    
    return result

if __name__ == '__main__':
    import sys
    with open(sys.argv[1]) as f: task = json.load(f)
    for i, ex in enumerate(task['train']):
        out = solve(ex['input'])
        exp = ex['output']
        match = out == exp
        print(f"Train {i}: {'PASS ✓' if match else 'FAIL'}")
        if not match:
            cnt = 0
            for r in range(len(exp)):
                for c in range(len(exp[0])):
                    if out[r][c] != exp[r][c]:
                        if cnt < 8: print(f"  ({r},{c}): exp={exp[r][c]}, got={out[r][c]}")
                        cnt += 1
            print(f"  Total diffs: {cnt}")
