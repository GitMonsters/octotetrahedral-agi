import json

# Detect source and target colors from training
def _detect_colors():
    with open('/tmp/rearc45/76207610.json') as f:
        task = json.load(f)
    p = task['train'][0]
    for r in range(len(p['input'])):
        for c in range(len(p['input'][0])):
            if p['input'][r][c] != p['output'][r][c]:
                return p['input'][r][c], p['output'][r][c]
    return None, None

SRC, TGT = _detect_colors()

def transform(grid):
    H = len(grid)
    W = len(grid[0])
    
    # Find largest rectangle of SRC with both dims >= 2
    best = (0, -1, -1, -1, -1)
    for r1 in range(H):
        for c1 in range(W):
            if grid[r1][c1] != SRC:
                continue
            max_c2 = W - 1
            for r2 in range(r1, H):
                new_max_c2 = c1 - 1
                for c in range(c1, max_c2 + 1):
                    if grid[r2][c] == SRC:
                        new_max_c2 = c
                    else:
                        break
                max_c2 = new_max_c2
                if max_c2 < c1:
                    break
                rows = r2 - r1 + 1
                cols = max_c2 - c1 + 1
                if rows >= 2 and cols >= 2:
                    area = rows * cols
                    if area > best[0]:
                        best = (area, r1, c1, r2, max_c2)
    
    out = [row[:] for row in grid]
    if best[0] > 0:
        _, r1, c1, r2, c2 = best
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                out[r][c] = TGT
    
    return out
