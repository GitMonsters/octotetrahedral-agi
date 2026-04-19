import json
from collections import Counter

# Determine output bg from training data
with open('/tmp/rearc45/0c3779d6.json') as f:
    _task = json.load(f)

# Find the output bg color (most frequent in outputs)
_out_colors = Counter()
for _pair in _task['train']:
    for row in _pair['output']:
        _out_colors.update(row)
OUTPUT_BG = _out_colors.most_common(1)[0][0]


def transform(grid):
    H = len(grid)
    W = len(grid[0])
    
    flat = [v for row in grid for v in row]
    color_counts = Counter(flat)
    
    # Scale = count of output bg color in input
    scale = color_counts.get(OUTPUT_BG, 0)
    if scale == 0:
        scale = 1
    
    # N_active = count of non-output-bg color in input
    n_active = sum(cnt for c, cnt in color_counts.items() if c != OUTPUT_BG)
    
    # Create output grid filled with output bg
    Ho = H * scale
    Wo = W * scale
    out = [[OUTPUT_BG] * Wo for _ in range(Ho)]
    
    # Determine active tile positions (bottom-right, column-major)
    full_cols = n_active // scale
    remainder = n_active % scale
    
    active_tiles = set()
    for col_offset in range(full_cols):
        tc = scale - 1 - col_offset
        for tr in range(scale):
            active_tiles.add((tr, tc))
    
    if remainder > 0:
        tc = scale - 1 - full_cols
        for tr in range(scale - remainder, scale):
            active_tiles.add((tr, tc))
    
    # Place input copies at active tile positions
    for tr, tc in active_tiles:
        for r in range(H):
            for c in range(W):
                out[tr * H + r][tc * W + c] = grid[r][c]
    
    return out
