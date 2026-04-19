from collections import Counter

def transform(grid):
    H = len(grid); W = len(grid[0])
    bg = Counter(v for row in grid for v in row).most_common(1)[0][0]
    non_bg = [(r, c, grid[r][c]) for r in range(H) for c in range(W) if grid[r][c] != bg]
    
    if not non_bg:
        return [row[:] for row in grid]
    
    # Count entirely blank rows and cols
    blank_rows = sum(1 for r in range(H) if all(grid[r][c] == bg for c in range(W)))
    blank_cols = sum(1 for c in range(W) if all(grid[r][c] == bg for r in range(H)))
    
    # Find tile period and content from non-bg cells (at input offset r_in=0, c_in=0)
    for pH in range(1, min(H+1, 10)):
        for pW in range(1, min(W+1, 10)):
            tile_map = {}
            valid = True
            for r, c, v in non_bg:
                ti, tj = r % pH, c % pW
                if (ti, tj) in tile_map and tile_map[(ti, tj)] != v:
                    valid = False; break
                tile_map[(ti, tj)] = v
            
            if not valid:
                continue
            
            # Valid tile found - determine output offset
            # Rule: if blank_rows >= pH, use row shift; else use col shift
            if blank_rows >= pH:
                r_out = pH - 1
                c_out = 0
            else:
                r_out = 0
                c_out = blank_cols % pW
            
            tile = [[tile_map.get((i, j), bg) for j in range(pW)] for i in range(pH)]
            
            out = []
            for r in range(H):
                row = [tile[(r - r_out) % pH][(c - c_out) % pW] for c in range(W)]
                out.append(row)
            return out
    
    return [row[:] for row in grid]
