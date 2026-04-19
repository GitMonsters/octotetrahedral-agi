import json
from collections import Counter

with open('/tmp/rearc45/0f1a1853.json') as f:
    _task = json.load(f)

WALL_COLOR = None
for _pair in _task['train']:
    _out_colors = set(v for row in _pair['output'] for v in row)
    _inp_colors = set(v for row in _pair['input'] for v in row)
    _new = _out_colors - _inp_colors
    if _new:
        WALL_COLOR = _new.pop()
        break
if WALL_COLOR is None:
    for _pair in _task['train']:
        oc = Counter(v for row in _pair['output'] for v in row)
        WALL_COLOR = oc.most_common(1)[0][0]
        break


def transform(grid):
    H = len(grid)
    W = len(grid[0])
    
    out = [[WALL_COLOR] * W for _ in range(H)]
    path = set()
    
    # Seg 1: row 1, cols 1 to W-1 (right)
    for c in range(1, W):
        path.add((1, c))
    # Seg 2: col 1, rows 2 to H-2 (down)
    for r in range(2, H - 1):
        path.add((r, 1))
    
    k = 1
    while True:
        # RIGHT+UP pair
        # H-right: row H-2k, cols [2k, W-2k]
        rh = H - 2 * k
        cs, ce = 2 * k, W - 2 * k
        if rh < 3 or cs >= ce:  # need at least 2 cols
            break
        for c in range(cs, ce + 1):
            path.add((rh, c))
        
        # V-up: col W-2k, rows [2k+1, H-2k-1]
        cv = W - 2 * k
        rs, re = 2 * k + 1, H - 2 * k - 1
        if cv < 2 or rs > re:
            break
        for r in range(rs, re + 1):
            path.add((r, cv))
        
        # LEFT+DOWN pair
        # H-left: row 2k+1, cols [2k+1, W-2k-1]
        rh2 = 2 * k + 1
        cs2, ce2 = 2 * k + 1, W - 2 * k - 1
        if rh2 >= H - 1 or cs2 > ce2:
            break
        for c in range(cs2, ce2 + 1):
            path.add((rh2, c))
        
        # V-down: col 2k+1, rows [2k+2, H-2k-2]
        cv2 = 2 * k + 1
        rs2, re2 = 2 * k + 2, H - 2 * k - 2
        # Stop if V-down column is adjacent to V-up column
        if cv2 + 1 >= cv or rs2 > re2:
            break
        for r in range(rs2, re2 + 1):
            path.add((r, cv2))
        
        k += 1
    
    for r, c in path:
        out[r][c] = grid[r][c]
    
    return out
