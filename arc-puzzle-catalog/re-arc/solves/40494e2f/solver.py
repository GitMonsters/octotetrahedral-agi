"""
ARC-AGI Task 40494e2f - Alternative Solver (Simplified Version)

This is a simpler approach to verify the solution.
Pattern: Stamp a cross/plus pattern into rectangles, then erase the original stamp.
"""

import json
from collections import Counter, deque


def transform(grid):
    """Simplified transform for validation."""
    H, W = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    
    # Detect colors by frequency
    flat = [grid[r][c] for r in range(H) for c in range(W)]
    colors = Counter(flat).most_common()
    
    if len(colors) < 3:
        return out
    
    bg = colors[0][0]
    rect_color = colors[1][0]
    marker_color = colors[2][0]
    
    # Find rectangles using BFS
    visited = set()
    rectangles = []
    
    for r in range(H):
        for c in range(W):
            if grid[r][c] == rect_color and (r, c) not in visited:
                # BFS
                queue = deque([(r, c)])
                cells = []
                
                while queue:
                    cr, cc = queue.popleft()
                    if (cr, cc) in visited:
                        continue
                    if cr < 0 or cr >= H or cc < 0 or cc >= W:
                        continue
                    if grid[cr][cc] != rect_color:
                        continue
                    
                    visited.add((cr, cc))
                    cells.append((cr, cc))
                    
                    for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                        queue.append((cr+dr, cc+dc))
                
                if cells:
                    rs = [r for r, c in cells]
                    cs = [c for r, c in cells]
                    min_r, max_r = min(rs), max(rs)
                    min_c, max_c = min(cs), max(cs)
                    
                    # Check if it's a filled rectangle
                    area = (max_r - min_r + 1) * (max_c - min_c + 1)
                    if area == len(cells) and area >= 30:
                        rectangles.append((min_r, max_r, min_c, max_c))
    
    # Find stamp (non-bg cells outside rectangles)
    rect_cells = set()
    for min_r, max_r, min_c, max_c in rectangles:
        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                rect_cells.add((r, c))
    
    stamp = {}
    for r in range(H):
        for c in range(W):
            if grid[r][c] != bg and (r, c) not in rect_cells:
                stamp[(r, c)] = grid[r][c]
    
    if not stamp:
        return out
    
    # Find stamp center (bbox center with neighbor check)
    rs = [r for r, c in stamp.keys()]
    cs = [c for r, c in stamp.keys()]
    center_r = (min(rs) + max(rs)) // 2
    center_c = (min(cs) + max(cs)) // 2
    
    # Build pattern
    pattern = {}
    for (r, c), val in stamp.items():
        pattern[(r - center_r, c - center_c)] = val
    
    # Detect extensions
    has_ext_down = (pattern.get((1, 0)) == marker_color and 
                    pattern.get((2, 0)) == marker_color)
    has_ext_up = (pattern.get((-1, 0)) == marker_color and 
                  pattern.get((-2, 0)) == marker_color)
    has_ext_right = (pattern.get((0, 1)) == marker_color and 
                     pattern.get((0, 2)) == marker_color)
    has_ext_left = (pattern.get((0, -1)) == marker_color and 
                    pattern.get((0, -2)) == marker_color)
    
    # Place stamp in each rectangle
    for min_r, max_r, min_c, max_c in rectangles:
        rect_h = max_r - min_r + 1
        rect_w = max_c - min_c + 1
        
        # Center in rectangle
        cr = min_r + rect_h // 2
        cc = min_c + rect_w // 2
        
        # Place pattern
        for (dr, dc), val in pattern.items():
            if val == marker_color:
                nr, nc = cr + dr, cc + dc
                if min_r <= nr <= max_r and min_c <= nc <= max_c:
                    out[nr][nc] = marker_color
        
        # Extend arms
        if has_ext_up:
            for r in range(min_r, cr):
                out[r][cc] = marker_color
        if has_ext_down:
            for r in range(cr + 1, max_r + 1):
                out[r][cc] = marker_color
        if has_ext_left:
            for c in range(min_c, cc):
                out[cr][c] = marker_color
        if has_ext_right:
            for c in range(cc + 1, max_c + 1):
                out[cr][c] = marker_color
    
    # Erase stamp
    for (r, c) in stamp.keys():
        out[r][c] = bg
    
    return out


if __name__ == '__main__':
    with open('40494e2f.json') as f:
        data = json.load(f)
    
    print("Testing simplified solver (v2):")
    for i, pair in enumerate(data['train']):
        inp = [row[:] for row in pair['input']]
        expected = pair['output']
        result = transform(inp)
        match = result == expected
        print(f"Train {i}: {'PASS' if match else 'FAIL'}")
        
        if not match:
            diffs = sum(1 for r in range(len(result)) for c in range(len(result[0]))
                       if result[r][c] != expected[r][c])
            print(f"  {diffs} differences")
