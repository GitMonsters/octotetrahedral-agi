from collections import Counter

def transform(grid):
    H = len(grid)
    W = len(grid[0])
    
    flat = [c for row in grid for c in row]
    bg = Counter(flat).most_common(1)[0][0]
    
    # Scan edges for non-bg cells
    edge_info = {
        'top': {c: grid[0][c] for c in range(W) if grid[0][c] != bg},
        'bottom': {c: grid[H-1][c] for c in range(W) if grid[H-1][c] != bg},
        'left': {r: grid[r][0] for r in range(H) if grid[r][0] != bg},
        'right': {r: grid[r][W-1] for r in range(H) if grid[r][W-1] != bg},
    }
    
    signal_edge = None
    deflector_edge = None
    signal_color = None
    
    for name, cells in edge_info.items():
        colors = set(cells.values())
        if colors and colors <= {8}:
            deflector_edge = name
        elif colors and 8 not in colors:
            signal_edge = name
            signal_color = next(c for c in colors if c != 8)
    
    signal_positions = [p for p, c in edge_info[signal_edge].items() if c != 8]
    deflector_set = set(p for p, c in edge_info[deflector_edge].items() if c == 8)
    
    # Build output: start with background, preserve 8s
    out = [[bg] * W for _ in range(H)]
    for r in range(H):
        for c in range(W):
            if grid[r][c] == 8:
                out[r][c] = 8
    
    # Trace rays from each signal perpendicular to its edge
    # Deflection: +1 step away from the deflector edge at each 8
    if signal_edge == 'bottom':
        d = +1 if deflector_edge == 'left' else -1
        for sc in signal_positions:
            shift = 0
            for r in range(H - 1, -1, -1):
                if r in deflector_set:
                    shift += d
                col = sc + shift
                if 0 <= col < W:
                    out[r][col] = signal_color
    
    elif signal_edge == 'top':
        d = +1 if deflector_edge == 'left' else -1
        for sc in signal_positions:
            shift = 0
            for r in range(H):
                if r in deflector_set:
                    shift += d
                col = sc + shift
                if 0 <= col < W:
                    out[r][col] = signal_color
    
    elif signal_edge == 'left':
        d = +1 if deflector_edge == 'top' else -1
        for sr in signal_positions:
            shift = 0
            for c in range(W):
                if c in deflector_set:
                    shift += d
                row = sr + shift
                if 0 <= row < H:
                    out[row][c] = signal_color
    
    elif signal_edge == 'right':
        d = +1 if deflector_edge == 'top' else -1
        for sr in signal_positions:
            shift = 0
            for c in range(W - 1, -1, -1):
                if c in deflector_set:
                    shift += d
                row = sr + shift
                if 0 <= row < H:
                    out[row][c] = signal_color
    
    return out
