import json, sys
from collections import Counter, deque

def solve(grid):
    rows, cols = len(grid), len(grid[0])
    g = [row[:] for row in grid]
    
    # Find connected components of non-bg cells (8-connected)
    visited = [[False]*cols for _ in range(rows)]
    components = []
    
    for r in range(rows):
        for c in range(cols):
            if g[r][c] != 0 and not visited[r][c]:
                comp = []
                queue = deque([(r, c)])
                visited[r][c] = True
                while queue:
                    cr, cc = queue.popleft()
                    comp.append((cr, cc, g[cr][cc]))
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0: continue
                            nr, nc = cr+dr, cc+dc
                            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and g[nr][nc] != 0:
                                visited[nr][nc] = True
                                queue.append((nr, nc))
                components.append(comp)
    
    for comp in components:
        color_counts = Counter(v for _, _, v in comp)
        if len(color_counts) < 2:
            continue  # No marker color — skip
        
        # Majority = shape, minority = marker
        sorted_colors = color_counts.most_common()
        shape_color = sorted_colors[0][0]
        marker_color = sorted_colors[-1][0]
        
        shape_cells = [(r, c) for r, c, v in comp if v == shape_color]
        marker_cells = [(r, c) for r, c, v in comp if v == marker_color]
        
        if not marker_cells or not shape_cells:
            continue
        
        n_markers = len(marker_cells)
        
        # Marker centroid
        mc_r = sum(r for r, c in marker_cells) / n_markers
        mc_c = sum(c for r, c in marker_cells) / n_markers
        
        # Find tip: for each cardinal direction, check width at the shape's extreme
        shape_rows = [r for r, c in shape_cells]
        shape_cols = [c for r, c in shape_cells]
        min_r, max_r = min(shape_rows), max(shape_rows)
        min_c, max_c = min(shape_cols), max(shape_cols)
        
        candidates = []
        
        # UP: cells in min_r row
        up_cells = [(r, c) for r, c in shape_cells if r == min_r]
        candidates.append(('UP', len(up_cells), up_cells))
        
        # DOWN: cells in max_r row
        down_cells = [(r, c) for r, c in shape_cells if r == max_r]
        candidates.append(('DOWN', len(down_cells), down_cells))
        
        # LEFT: cells in min_c col
        left_cells = [(r, c) for r, c in shape_cells if c == min_c]
        candidates.append(('LEFT', len(left_cells), left_cells))
        
        # RIGHT: cells in max_c col
        right_cells = [(r, c) for r, c in shape_cells if c == max_c]
        candidates.append(('RIGHT', len(right_cells), right_cells))
        
        # Adjacent width: how wide the shape is one step inside from the tip
        adj_w = {
            'UP': sum(1 for r, c in shape_cells if r == min_r + 1),
            'DOWN': sum(1 for r, c in shape_cells if r == max_r - 1),
            'LEFT': sum(1 for r, c in shape_cells if c == min_c + 1),
            'RIGHT': sum(1 for r, c in shape_cells if c == max_c - 1),
        }
        
        # For each candidate, compute potential beam length
        dir_map = {'UP': (-1, 0), 'DOWN': (1, 0), 'LEFT': (0, -1), 'RIGHT': (0, 1)}
        
        def tip_score(cand):
            direction, width, cells = cand
            aw = adj_w[direction]
            # Pick tip cell (farthest from marker centroid)
            tip_cell = max(cells, key=lambda rc: (rc[0]-mc_r)**2 + (rc[1]-mc_c)**2)
            dr2, dc2 = dir_map[direction]
            # Count available beam cells
            avail = 0
            br, bc = tip_cell[0] + dr2, tip_cell[1] + dc2
            while 0 <= br < rows and 0 <= bc < cols and g[br][bc] == 0:
                avail += 1
                br += dr2
                bc += dc2
            beam_len = min(n_markers, avail)
            max_dist = max(((r-mc_r)**2 + (c-mc_c)**2)**0.5 for r, c in cells)
            return (width, -aw, -beam_len, -max_dist)
        
        candidates.sort(key=tip_score)
        
        best_dir, best_width, best_cells = candidates[0]
        
        # Pick tip cell: farthest from marker centroid
        tip = max(best_cells, key=lambda rc: (rc[0]-mc_r)**2 + (rc[1]-mc_c)**2)
        
        # Beam direction
        dir_map = {'UP': (-1, 0), 'DOWN': (1, 0), 'LEFT': (0, -1), 'RIGHT': (0, 1)}
        dr, dc = dir_map[best_dir]
        
        # Remove markers
        for r, c in marker_cells:
            g[r][c] = 0
        
        # Draw beam
        beam_r, beam_c = tip[0] + dr, tip[1] + dc
        drawn = 0
        while 0 <= beam_r < rows and 0 <= beam_c < cols and drawn < n_markers:
            if g[beam_r][beam_c] != 0:
                break
            g[beam_r][beam_c] = marker_color
            drawn += 1
            beam_r += dr
            beam_c += dc
    
    return g

if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        task = json.load(f)

    for i, ex in enumerate(task['train']):
        result = solve(ex['input'])
        expected = ex['output']
        match = result == expected
        print(f"Train {i}: {'PASS' if match else 'FAIL'}")
        if not match:
            for r in range(len(expected)):
                for c in range(len(expected[0])):
                    if result[r][c] != expected[r][c]:
                        print(f"  ({r},{c}): got {result[r][c]}, expected {expected[r][c]}")
