from collections import Counter

def transform(grid):
    H = len(grid)
    W = len(grid[0])
    flat = [c for row in grid for c in row]
    bg = Counter(flat).most_common(1)[0][0]
    
    # Find bounding box of non-bg cells
    non_bg = [(r, c) for r in range(H) for c in range(W) if grid[r][c] != bg]
    if not non_bg:
        return [row[:] for row in grid]
    
    min_r = min(r for r, c in non_bg)
    max_r = max(r for r, c in non_bg)
    min_c = min(c for r, c in non_bg)
    max_c = max(c for r, c in non_bg)
    
    height = max_r - min_r + 1
    width = max_c - min_c + 1
    
    out = [row[:] for row in grid]
    
    # Try vertical repeat with different periods
    for divisor in range(2, height + 1):
        if height % divisor != 0:
            continue
        period = height // divisor
        v_repeat = True
        for rep in range(1, divisor):
            for i in range(period):
                if grid[min_r + i] != grid[min_r + rep * period + i]:
                    v_repeat = False
                    break
            if not v_repeat:
                break
        if v_repeat:
            for r in range(H):
                idx = (r - min_r) % period
                out[r] = grid[min_r + idx][:]
            return out
    
    # Try horizontal repeat with different periods
    for divisor in range(2, width + 1):
        if width % divisor != 0:
            continue
        period = width // divisor
        h_repeat = True
        for rep in range(1, divisor):
            for r in range(H):
                for j in range(period):
                    if grid[r][min_c + j] != grid[r][min_c + rep * period + j]:
                        h_repeat = False
                        break
                if not h_repeat:
                    break
            if not h_repeat:
                break
        if h_repeat:
            for r in range(H):
                for c in range(W):
                    idx = (c - min_c) % period
                    out[r][c] = grid[r][min_c + idx]
            return out
    
    return out
