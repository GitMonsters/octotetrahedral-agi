"""Solver for 135a2760 — fix repeating patterns in bordered panels"""
import json, math, numpy as np
from typing import List
from collections import Counter

def solve(grid: List[List[int]]) -> List[List[int]]:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    result = g.copy()

    # Detect frame/border color (non-background color forming lines)
    bg = int(g[0, 0])
    color_line_counts: Counter = Counter()
    for r in range(H):
        for color, count in Counter(int(v) for v in g[r]).items():
            if color != bg and count > W * 0.5:
                color_line_counts[color] += 1
    for c in range(W):
        for color, count in Counter(int(g[r, c]) for r in range(H)).items():
            if color != bg and count > H * 0.5:
                color_line_counts[color] += 1
    if not color_line_counts:
        return result.tolist()
    border_color = color_line_counts.most_common(1)[0][0]

    # Find border rows and border columns
    border_rows = [r for r in range(H)
                   if int(np.sum(g[r] == border_color)) > W * 0.5]
    border_cols = [c for c in range(W)
                   if int(np.sum(g[:, c] == border_color)) > H * 0.5]

    # Panel interior ranges between consecutive borders
    row_ranges = [(border_rows[i] + 1, border_rows[i + 1] - 1)
                  for i in range(len(border_rows) - 1)
                  if border_rows[i + 1] - border_rows[i] > 1]
    col_ranges = [(border_cols[i] + 1, border_cols[i + 1] - 1)
                  for i in range(len(border_cols) - 1)
                  if border_cols[i + 1] - border_cols[i] > 1]

    for r_start, r_end in row_ranges:
        for c_start, c_end in col_ranges:
            interior = g[r_start:r_end + 1, c_start:c_end + 1].copy()
            ih, iw = interior.shape
            if ih < 1 or iw < 1:
                continue

            best_period = None
            best_errors = ih * iw + 1

            for pr in range(1, ih + 1):
                for pc in range(1, iw + 1):
                    # Require ≥3 samples at EVERY tile position for reliable majority vote
                    min_reps = (math.ceil((ih - pr + 1) / pr)
                                * math.ceil((iw - pc + 1) / pc))
                    if min_reps < 3:
                        continue
                    tile = np.zeros((pr, pc), dtype=int)
                    errors = 0
                    for tr in range(pr):
                        for tc in range(pc):
                            vals = [int(interior[rr][cc])
                                    for rr in range(tr, ih, pr)
                                    for cc in range(tc, iw, pc)]
                            if vals:
                                majority = Counter(vals).most_common(1)[0]
                                tile[tr][tc] = majority[0]
                                errors += len(vals) - majority[1]

                    if errors < best_errors:
                        best_errors = errors
                        best_period = (pr, pc, tile.copy())
                        if errors == 0:
                            break
                if best_errors == 0:
                    break

            if best_period and best_errors > 0:
                pr, pc, tile = best_period
                for rr in range(ih):
                    for cc in range(iw):
                        result[r_start + rr][c_start + cc] = int(
                            tile[rr % pr][cc % pc])

    return result.tolist()

if __name__ == '__main__':
    import sys
    with open(sys.argv[1]) as f: task = json.load(f)
    for split in ['train', 'test']:
        for i, ex in enumerate(task.get(split, [])):
            out = solve(ex['input'])
            exp = ex['output']
            match = out == exp
            print(f"{split.title()} {i}: {'PASS ✓' if match else 'FAIL'}")
            if not match:
                oa, ea = np.array(out), np.array(exp)
                diff = oa != ea
                for r in range(diff.shape[0]):
                    for c in range(diff.shape[1]):
                        if diff[r][c]:
                            print(f"  ({r},{c}): exp={ea[r][c]}, got={oa[r][c]}")
                print(f"  Total diffs: {int(np.sum(diff))}")
