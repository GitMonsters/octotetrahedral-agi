import math
from collections import Counter

def get_objects(grid):
    # Find contiguous blocks of color 2 and 3
    objects = {}
    visited = set()
    rows = len(grid)
    cols = len(grid[0])
    
    for r in range(rows):
        for c in range(cols):
            val = grid[r][c]
            if val in [2, 3] and (r, c) not in visited:
                # BFS to find component
                cells = []
                q = [(r, c)]
                visited.add((r, c))
                while q:
                    curr_r, curr_c = q.pop(0)
                    cells.append((curr_r, curr_c))
                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nr, nc = curr_r + dr, curr_c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and \
                           grid[nr][nc] == val and (nr, nc) not in visited:
                            visited.add((nr, nc))
                            q.append((nr, nc))
                objects[val] = cells
    return objects

def get_bounds(cells):
    rs = [r for r, c in cells]
    cs = [c for r, c in cells]
    return min(rs), max(rs), min(cs), max(cs)

def is_blocked(grid, r, c, blocking_colors):
    if not (0 <= r < len(grid) and 0 <= c < len(grid[0])):
        return True
    return grid[r][c] in blocking_colors

def solve_grid(grid):
    objects = get_objects(grid)
    if 2 not in objects or 3 not in objects:
        return grid # Cannot solve
    
    cells2 = objects[2]
    cells3 = objects[3]
    
    min_r2, max_r2, min_c2, max_c2 = get_bounds(cells2)
    min_r3, max_r3, min_c3, max_c3 = get_bounds(cells3)
    
    # Determine orientation
    # If height > width -> Vertical
    h2 = max_r2 - min_r2 + 1
    w2 = max_c2 - min_c2 + 1
    h3 = max_r3 - min_r3 + 1
    w3 = max_c3 - min_c3 + 1
    
    # Assume both have same orientation
    is_vertical = h2 > w2 or h3 > w3
    
    # Determine Background and Preserved Colors
    flat_grid = [c for row in grid for c in row]
    bg_color = Counter(flat_grid).most_common(1)[0][0]
    
    unique_colors = set(flat_grid)
    preserved_colors = unique_colors - {bg_color, 2, 3}
    
    # Heuristic Blocking
    blocking_colors = {8} 
    
    output = [row[:] for row in grid]
    rows = len(grid)
    cols = len(grid[0])

    if is_vertical:
        # Connect via Horizontal Channel (P1, P2)
        c_start = min_c3 
        c_end = min_c2
        
        y_start_gap = min(max_r2, max_r3) + 1
        y_end_gap = max(min_r2, min_r3) - 1
        
        mid_y = int((y_start_gap + y_end_gap) / 2)
        
        search_order = []
        
        # 1. Inside Gap: Search Center-Out
        if y_start_gap <= y_end_gap:
            inside_rows = []
            radius = max(rows - mid_y, mid_y)
            for i in range(radius + 1):
                r_plus = mid_y + i
                r_minus = mid_y - i
                if y_start_gap <= r_plus <= y_end_gap:
                    inside_rows.append(r_plus)
                if i > 0 and y_start_gap <= r_minus <= y_end_gap:
                    inside_rows.append(r_minus)
            seen = set()
            for r in inside_rows:
                if r not in seen:
                    search_order.append(r)
                    seen.add(r)
                    
        # 2. Outside Gap: Search Top-Down and Bottom-Up
        for r in range(y_start_gap):
             if r not in search_order: search_order.append(r)
        for r in range(rows - 1, y_end_gap, -1):
             if r not in search_order: search_order.append(r)
             
        best_row = -1
            
        for r in search_order:
            c1, c2 = min(c_start, c_end), max(c_start, c_end)
            clean = True
            for c in range(c1, c2 + 1):
                if is_blocked(grid, r, c, blocking_colors):
                    clean = False; break
            
            if clean:
                r3_closest = max_r3 if r > max_r3 else min_r3
                r1_v, r2_v = min(r, r3_closest), max(r, r3_closest)
                for k in range(r1_v, r2_v + 1):
                    if is_blocked(grid, k, c_start, blocking_colors):
                        clean = False; break
                if not clean: continue

                r2_closest = max_r2 if r > max_r2 else min_r2
                r1_v, r2_v = min(r, r2_closest), max(r, r2_closest)
                for k in range(r1_v, r2_v + 1):
                    if is_blocked(grid, k, c_end, blocking_colors):
                        clean = False; break
                if not clean: continue
                
                best_row = r
                break
        
        if best_row != -1:
            # Draw path
            # Vertical from 3
            r3_closest = max_r3 if best_row > max_r3 else min_r3
            step = 1 if best_row >= r3_closest else -1
            for r in range(r3_closest, best_row + step, step):
                if output[r][c_start] not in [2] and output[r][c_start] not in preserved_colors:
                     output[r][c_start] = 3
            
            # Vertical from 2
            r2_closest = max_r2 if best_row > max_r2 else min_r2
            step = 1 if best_row >= r2_closest else -1
            for r in range(r2_closest, best_row + step, step):
                if output[r][c_end] not in [2] and output[r][c_end] not in preserved_colors:
                    output[r][c_end] = 3
                    
            # Horizontal
            c1, c2 = min(c_start, c_end), max(c_start, c_end)
            for c in range(c1, c2 + 1):
                if output[best_row][c] not in [2] and output[best_row][c] not in preserved_colors:
                    output[best_row][c] = 3
                    
    else:
        # Horizontal blocks -> Vertical Channel (P3)
        # Variable Width -> Optimize for Squareness
        
        r_start = min_r3
        r_end = min_r2
        
        y_center_2 = (min_r2 + max_r2) / 2
        y_center_3 = (min_r3 + max_r3) / 2
        target_height = abs(y_center_2 - y_center_3)
        
        cols = len(grid[0])
        x_center_blocks = (min_c2 + max_c2 + min_c3 + max_c3) / 4
        
        go_right = x_center_blocks < (cols / 2)
        
        if go_right:
            start_x = max(max_c2, max_c3)
            target_x = start_x + target_height + 1
        else:
            start_x = min(min_c2, min_c3)
            target_x = start_x - (target_height + 1)
            
        target_x = int(target_x)
        target_x = max(0, min(cols - 1, target_x))
        
        search_order = []
        for i in range(cols):
            if target_x + i < cols: search_order.append(target_x + i)
            if i > 0 and target_x - i >= 0: search_order.append(target_x - i)
            
        best_col = -1
        
        for c in search_order:
            r1, r2 = min(r_start, r_end), max(r_start, r_end)
            clean = True
            for r in range(r1, r2 + 1):
                if is_blocked(grid, r, c, blocking_colors):
                    clean = False; break
            
            if clean:
                c3_closest = max_c3 if c > max_c3 else min_c3
                c1, c2 = min(c, c3_closest), max(c, c3_closest)
                for k in range(c1, c2 + 1):
                    if is_blocked(grid, r_start, k, blocking_colors):
                        clean = False; break
                if not clean: continue
                
                c2_closest = max_c2 if c > max_c2 else min_c2
                c1, c2 = min(c, c2_closest), max(c, c2_closest)
                for k in range(c1, c2 + 1):
                    if is_blocked(grid, r_end, k, blocking_colors):
                        clean = False; break
                if not clean: continue
                
                best_col = c
                break
                
        if best_col != -1:
            # Draw path
            # Horizontal from 3
            c3_closest = max_c3 if best_col > max_c3 else min_c3
            step = 1 if best_col >= c3_closest else -1
            for c in range(c3_closest, best_col + step, step):
                if output[r_start][c] not in [2] and output[r_start][c] not in preserved_colors:
                    output[r_start][c] = 3
                    
            # Horizontal from 2
            c2_closest = max_c2 if best_col > max_c2 else min_c2
            step = 1 if best_col >= c2_closest else -1
            for c in range(c2_closest, best_col + step, step):
                if output[r_end][c] not in [2] and output[r_end][c] not in preserved_colors:
                    output[r_end][c] = 3
            
            # Vertical
            r1, r2 = min(r_start, r_end), max(r_start, r_end)
            for r in range(r1, r2 + 1):
                if output[r][best_col] not in [2] and output[r][best_col] not in preserved_colors:
                    output[r][best_col] = 3

    return output

def transform(grid):
    return solve_grid(grid)
