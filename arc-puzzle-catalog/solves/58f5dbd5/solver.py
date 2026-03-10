import json, sys
from collections import Counter, deque

def solve(grid):
    rows, cols = len(grid), len(grid[0])
    
    border = [grid[0][c] for c in range(cols)] + [grid[rows-1][c] for c in range(cols)]
    border += [grid[r][0] for r in range(rows)] + [grid[r][cols-1] for r in range(rows)]
    bg = Counter(border).most_common(1)[0][0]
    
    # Group non-bg cells by color
    color_cells = {}
    for r in range(rows):
        for c in range(cols):
            v = grid[r][c]
            if v != bg:
                color_cells.setdefault(v, []).append((r, c))
    
    # For each color, find connected components and identify blocks vs patterns
    blocks = {}    # color -> list of (r, c, h, w)
    patterns = {}  # color -> 2D pattern (bounding box with 0/1)
    
    for color, cells in color_cells.items():
        cell_set = set(cells)
        visited = set()
        components = []
        for r, c in cells:
            if (r, c) in visited:
                continue
            comp = []
            queue = deque([(r, c)])
            while queue:
                cr, cc = queue.popleft()
                if (cr, cc) in visited:
                    continue
                visited.add((cr, cc))
                comp.append((cr, cc))
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = cr+dr, cc+dc
                    if (nr, nc) in cell_set and (nr, nc) not in visited:
                        queue.append((nr, nc))
            components.append(comp)
        
        # Identify blocks (filled rectangles >= 4x4)
        block_cells = set()
        for comp in components:
            min_r = min(r for r, c in comp)
            max_r = max(r for r, c in comp)
            min_c = min(c for r, c in comp)
            max_c = max(c for r, c in comp)
            h = max_r - min_r + 1
            w = max_c - min_c + 1
            if h * w == len(comp) and h >= 4 and w >= 4:
                blocks.setdefault(color, []).append((min_r, min_c, h, w))
                block_cells.update(comp)
        
        # Remaining cells form the pattern (may be disconnected)
        remaining = [(r, c) for r, c in cells if (r, c) not in block_cells]
        if remaining:
            min_r = min(r for r, c in remaining)
            max_r = max(r for r, c in remaining)
            min_c = min(c for r, c in remaining)
            max_c = max(c for r, c in remaining)
            h = max_r - min_r + 1
            w = max_c - min_c + 1
            pat = [[0]*w for _ in range(h)]
            for r, c in remaining:
                pat[r - min_r][c - min_c] = 1
            patterns[color] = pat
    
    # Collect all blocks
    all_block_positions = []
    for color, blist in blocks.items():
        for b in blist:
            all_block_positions.append((b, color))
    
    if not all_block_positions:
        return grid
    
    # Compute output bounding box (blocks + 1 cell border)
    all_r, all_c = [], []
    for (br, bc, bh, bw), _ in all_block_positions:
        all_r.extend([br, br + bh - 1])
        all_c.extend([bc, bc + bw - 1])
    
    out_min_r = min(all_r) - 1
    out_max_r = max(all_r) + 1
    out_min_c = min(all_c) - 1
    out_max_c = max(all_c) + 1
    out_h = out_max_r - out_min_r + 1
    out_w = out_max_c - out_min_c + 1
    
    out = [[bg]*out_w for _ in range(out_h)]
    
    for (br, bc, bh, bw), color in all_block_positions:
        # Fill block with color
        for dr in range(bh):
            for dc in range(bw):
                out[br - out_min_r + dr][bc - out_min_c + dc] = color
        
        # Stamp complement into interior
        if color in patterns:
            pat = patterns[color]
            ph, pw = len(pat), len(pat[0])
            int_r = br - out_min_r + 1
            int_c = bc - out_min_c + 1
            for pr in range(ph):
                for pc in range(pw):
                    if pat[pr][pc] == 1:
                        out[int_r + pr][int_c + pc] = bg
    
    return out

if __name__ == '__main__':
    with open(sys.argv[1]) as f:
        task = json.load(f)
    for i, ex in enumerate(task['train']):
        result = solve(ex['input'])
        ok = result == ex['output']
        print(f"Train {i}: {'PASS' if ok else 'FAIL'}")
        if not ok:
            for r in range(min(len(result), len(ex['output']))):
                for c in range(min(len(result[0]), len(ex['output'][0]))):
                    if result[r][c] != ex['output'][r][c]:
                        print(f"  ({r},{c}): got {result[r][c]}, exp {ex['output'][r][c]}")
            if len(result) != len(ex['output']) or len(result[0]) != len(ex['output'][0]):
                print(f"  Size mismatch: {len(result)}x{len(result[0])} vs {len(ex['output'])}x{len(ex['output'][0])}")
