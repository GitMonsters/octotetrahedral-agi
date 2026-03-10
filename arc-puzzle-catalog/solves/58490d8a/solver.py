import json, sys
from collections import Counter, deque

def solve(grid):
    rows, cols = len(grid), len(grid[0])
    
    # Find the legend: rectangular block of 0s (black cells)
    zero_cells = set()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 0:
                zero_cells.add((r, c))
    
    if not zero_cells:
        return grid
    
    min_r = min(r for r, c in zero_cells)
    max_r = max(r for r, c in zero_cells)
    min_c = min(c for r, c in zero_cells)
    max_c = max(c for r, c in zero_cells)
    legend_h = max_r - min_r + 1
    legend_w = max_c - min_c + 1
    
    # Legend area = entire rectangle (including colored dots inside)
    legend_area = set()
    for r in range(min_r, max_r + 1):
        for c in range(min_c, max_c + 1):
            legend_area.add((r, c))
    
    # Extract legend dot colors by relative row
    legend_colors = {}
    for r in range(min_r, max_r + 1):
        for c in range(min_c, max_c + 1):
            v = grid[r][c]
            if v != 0:
                rel_r = r - min_r
                legend_colors[rel_r] = v
    
    # Find main area background
    main_counts = Counter()
    for r in range(rows):
        for c in range(cols):
            if (r, c) not in legend_area:
                main_counts[grid[r][c]] += 1
    bg = main_counts.most_common(1)[0][0]
    
    # Collect non-bg cells in main area by color
    color_cells = {}
    for r in range(rows):
        for c in range(cols):
            if (r, c) in legend_area:
                continue
            v = grid[r][c]
            if v != bg and v != 0:
                color_cells.setdefault(v, set()).add((r, c))
    
    # Count 8-connected components for each legend color
    dirs8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    
    color_counts = {}
    for rel_r, color in legend_colors.items():
        cells = color_cells.get(color, set())
        visited = set()
        count = 0
        for r, c in cells:
            if (r, c) in visited:
                continue
            count += 1
            queue = deque([(r, c)])
            while queue:
                cr, cc = queue.popleft()
                if (cr, cc) in visited:
                    continue
                visited.add((cr, cc))
                for dr, dc in dirs8:
                    nr, nc = cr+dr, cc+dc
                    if (nr, nc) in cells and (nr, nc) not in visited:
                        queue.append((nr, nc))
        color_counts[rel_r] = (color, count)
    
    # Build output
    out = [[0] * legend_w for _ in range(legend_h)]
    for rel_r, (color, count) in color_counts.items():
        for i in range(count):
            col = 1 + 2 * i
            if col < legend_w:
                out[rel_r][col] = color
    
    return out

if __name__ == '__main__':
    with open(sys.argv[1]) as f:
        task = json.load(f)
    for i, ex in enumerate(task['train']):
        result = solve(ex['input'])
        ok = result == ex['output']
        print(f"Train {i}: {'PASS' if ok else 'FAIL'}")
        if not ok:
            print(f"  Size: {len(result)}x{len(result[0])} vs {len(ex['output'])}x{len(ex['output'][0])}")
            for r in range(min(len(result), len(ex['output']))):
                for c in range(min(len(result[0]), len(ex['output'][0]))):
                    if result[r][c] != ex['output'][r][c]:
                        print(f"  ({r},{c}): got {result[r][c]}, exp {ex['output'][r][c]}")
