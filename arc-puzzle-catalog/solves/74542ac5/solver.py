import json

# Detect fill color from training data
def _detect_fill_color():
    with open('/tmp/rearc45/74542ac5.json') as f:
        task = json.load(f)
    for p in task['train']:
        inp, out = p['input'], p['output']
        for r in range(len(inp)):
            for c in range(len(inp[0])):
                if inp[r][c] != out[r][c]:
                    return out[r][c]
    return 8

FILL = _detect_fill_color()

def transform(grid):
    H = len(grid)
    W = len(grid[0])
    period = 2 * (W - 1)
    
    out = [row[:] for row in grid]
    
    v_index = 0
    while True:
        r_start = v_index * period
        r_end = r_start + period
        
        if r_end >= H:  # incomplete V, don't fill
            break
        
        center = r_start + (W - 1)
        
        if v_index % 2 == 0:
            # Fill upper-left triangle (left of descending arm)
            for r in range(r_start, center):
                diag_col = center - r
                for c in range(0, diag_col):
                    out[r][c] = FILL
        else:
            # Fill inside diamond (right of both arms)
            for r in range(r_start + 1, r_end):
                diag_col = abs(r - center)
                for c in range(diag_col + 1, W):
                    out[r][c] = FILL
        
        v_index += 1
    
    return out
