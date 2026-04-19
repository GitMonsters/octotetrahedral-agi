def transform(input_grid):
    rows = len(input_grid)
    cols = len(input_grid[0])
    
    # Detect background
    counts = {}
    for r in range(rows):
        for c in range(cols):
            val = input_grid[r][c]
            counts[val] = counts.get(val, 0) + 1
    # Handle empty counts (unlikely)
    if not counts: return input_grid
    bg_color = max(counts, key=counts.get)
    
    # Output grid copy
    output_grid = [row[:] for row in input_grid]
    
    # Find objects (8-connected)
    objects = []
    visited = set()
    
    for r in range(rows):
        for c in range(cols):
            if input_grid[r][c] != bg_color and (r, c) not in visited:
                color = input_grid[r][c]
                # BFS
                q = [(r, c)]
                visited.add((r, c))
                obj_coords = []
                while q:
                    curr_r, curr_c = q.pop(0)
                    obj_coords.append((curr_r, curr_c))
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr==0 and dc==0: continue
                            nr, nc = curr_r + dr, curr_c + dc
                            if 0 <= nr < rows and 0 <= nc < cols:
                                if input_grid[nr][nc] == color and (nr, nc) not in visited:
                                    visited.add((nr, nc))
                                    q.append((nr, nc))
                objects.append((color, obj_coords))
                
    for color, coords in objects:
        rs = [p[0] for p in coords]
        cs = [p[1] for p in coords]
        min_r, max_r = min(rs), max(rs)
        min_c, max_c = min(cs), max(cs)
        bbox_width = max_c - min_c + 1
        
        # Rule 1: Big Objects (Width >= 4) -> Full Rectangle
        if bbox_width >= 4:
            for r in range(min_r, max_r + 1):
                for c in range(min_c, max_c + 1):
                    output_grid[r][c] = color
            continue
            
        # Rule 2: Small Objects (Width < 4) -> Head/Stem Logic
        
        # Identify Head
        best_row = -1
        max_pixels = -1
        
        row_pixel_map = {}
        for r_idx, c_idx in coords:
            if r_idx not in row_pixel_map: row_pixel_map[r_idx] = []
            row_pixel_map[r_idx].append(c_idx)
            
        for r in range(min_r, max_r + 1):
            if r in row_pixel_map:
                px = row_pixel_map[r]
                count = len(px)
                if count > max_pixels:
                    max_pixels = count
                    best_row = r
                elif count == max_pixels:
                    pass
                    
        head_row_idx = best_row
        if head_row_idx == -1: continue 
        
        head_pixels = row_pixel_map[head_row_idx]
        head_min = min(head_pixels)
        head_max = max(head_pixels)
        head_cols_set = set(head_pixels)
        
        # Helper to process a range of rows
        def process_rows(rows_iter):
            current_cols = head_cols_set.copy()
            
            for r in rows_iter:
                # Check boundaries
                if r < 0 or r >= rows: break
                
                target_pixels = row_pixel_map.get(r, [])
                
                should_preserve = False
                
                dist = abs(r - head_row_idx)
                
                if not target_pixels:
                    # Empty
                    if dist == 1:
                        should_preserve = True 
                    else:
                        should_preserve = False
                else:
                    # Has pixels
                    t_min = min(target_pixels)
                    t_max = max(target_pixels)
                    t_len = t_max - t_min + 1
                    is_solid = (len(target_pixels) == t_len)
                    
                    if dist == 1:
                        if is_solid:
                            should_preserve = True
                        else:
                            should_preserve = False 
                    else:
                        # Dist >= 2
                        if t_max < head_min or t_min > head_max:
                            should_preserve = True
                        else:
                            if t_min > head_min and t_max < head_max:
                                should_preserve = True 
                            else:
                                should_preserve = False 
                
                if should_preserve:
                    if not target_pixels:
                        current_cols = set()
                    else:
                        t_min = min(target_pixels)
                        t_max = max(target_pixels)
                        if t_min > head_min and t_max < head_max:
                            current_cols = set(target_pixels)
                        else:
                            current_cols = head_cols_set.copy()
                else:
                    for c in current_cols:
                        output_grid[r][c] = color

        # Define Ranges
        exp_limit = 0
        if bbox_width >= 3: exp_limit = 2
        
        down_range = []
        if head_row_idx <= max_r: 
             limit = max_r
             if head_row_idx == min_r:
                 limit = min(rows - 1, max_r + exp_limit)
             down_range = range(head_row_idx + 1, limit + 1)
             
        up_range = []
        if head_row_idx >= min_r: 
            limit = min_r
            if head_row_idx == max_r:
                limit = max(0, min_r - exp_limit)
            up_range = range(head_row_idx - 1, limit - 1, -1)
            
        process_rows(down_range)
        process_rows(up_range)
        
        # Ensure Head is filled
        for c in range(head_min, head_max + 1):
             output_grid[head_row_idx][c] = color
                    
    return output_grid
