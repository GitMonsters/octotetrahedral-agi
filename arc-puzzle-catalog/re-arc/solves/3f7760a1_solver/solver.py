import numpy as np
import json

def transform(input_grid):
    input_grid = np.array(input_grid)
    output_grid = input_grid.copy()
    rows, cols = input_grid.shape
    
    # Detect background
    vals, counts = np.unique(input_grid, return_counts=True)
    bg_color = vals[np.argmax(counts)]
    
    # Find objects (8-connected)
    objects = []
    visited = np.zeros((rows, cols), dtype=bool)
    
    for r in range(rows):
        for c in range(cols):
            if input_grid[r, c] != bg_color and not visited[r, c]:
                color = input_grid[r, c]
                # BFS
                q = [(r, c)]
                visited[r, c] = True
                obj_coords = []
                while q:
                    curr_r, curr_c = q.pop(0)
                    obj_coords.append((curr_r, curr_c))
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr==0 and dc==0: continue
                            nr, nc = curr_r + dr, curr_c + dc
                            if 0 <= nr < rows and 0 <= nc < cols:
                                if input_grid[nr, nc] == color and not visited[nr, nc]:
                                    visited[nr, nc] = True
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
                    output_grid[r, c] = color
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
                    # Tie break: prefer top-most
                    pass
                    
        head_row_idx = best_row
        if head_row_idx == -1: continue 
        
        head_pixels = row_pixel_map[head_row_idx]
        head_min = min(head_pixels)
        head_max = max(head_pixels)
        
        # Determine Processing Range and Split into Up/Down
        
        # Helper to process a range of rows
        def process_rows(rows_iter):
            head_cols_set = set(head_pixels)
            current_cols = head_cols_set.copy()
            
            for r in rows_iter:
                # Check boundaries
                if r < 0 or r >= rows: break
                
                target_pixels = row_pixel_map.get(r, [])
                
                # Determine Action: Preserve or Overwrite?
                should_preserve = False
                
                dist = abs(r - head_row_idx)
                
                if not target_pixels:
                    # Empty
                    if dist == 1:
                        should_preserve = True # Neighbor Empty -> Stop/Preserve
                    else:
                        should_preserve = False # Distant Empty -> Overwrite
                else:
                    # Has pixels
                    t_min = min(target_pixels)
                    t_max = max(target_pixels)
                    t_len = t_max - t_min + 1
                    is_solid = (len(target_pixels) == t_len)
                    
                    if dist == 1:
                        # Neighbor
                        if is_solid:
                            should_preserve = True
                        else:
                            should_preserve = False # Not Solid -> Overwrite
                    else:
                        # Distant
                        # Disjoint check
                        if t_max < head_min or t_min > head_max:
                            should_preserve = True
                        else:
                            # Overlap check alignment
                            if t_min > head_min and t_max < head_max:
                                should_preserve = True # Center
                            else:
                                should_preserve = False # Edge
                
                if should_preserve:
                    # Update Pattern for NEXT row
                    if not target_pixels:
                        # Empty Preserved -> Pattern becomes Empty (Stop propagation)
                        current_cols = set()
                    else:
                        t_min = min(target_pixels)
                        t_max = max(target_pixels)
                        if t_min > head_min and t_max < head_max:
                            # Center -> Propagate this pattern
                            current_cols = set(target_pixels)
                        else:
                            # Edge -> Reset to Head pattern
                            current_cols = head_cols_set.copy()
                else:
                    # Overwrite with current pattern
                    for c in current_cols:
                        output_grid[r, c] = color
                    # Pattern remains same for next row
                    pass

        # Define Ranges
        up_range = []
        down_range = []
        
        # Expansion Limits
        exp_limit = 0
        if bbox_width >= 3: exp_limit = 2
        
        # Down Range
        if head_row_idx <= max_r: # Head is at or above bottom
             # If Head is Top (min_r), expand Down
             # If Head is Middle, expand Down to max_r (no growth)
             limit = max_r
             if head_row_idx == min_r:
                 limit = min(rows - 1, max_r + exp_limit)
             
             down_range = range(head_row_idx + 1, limit + 1)
             
        # Up Range
        if head_row_idx >= min_r: # Head is at or below top
            limit = min_r
            if head_row_idx == max_r:
                limit = max(0, min_r - exp_limit)
            
            up_range = range(head_row_idx - 1, limit - 1, -1)
            
        # Execute
        process_rows(down_range)
        process_rows(up_range)
        
        # Ensure Head is filled
        for c in range(head_min, head_max + 1):
             output_grid[head_row_idx, c] = color
                    
    return output_grid.tolist()

def check():
    with open('/Users/evanpieser/re_arc_solves/3f7760a1_task.json', 'r') as f:
        data = json.load(f)
    correct = 0
    total = len(data['train'])
    for i, pair in enumerate(data['train']):
        inp = pair['input']
        out = pair['output']
        pred = transform(inp)
        if pred == out:
            print(f"Example {i}: CORRECT")
            correct += 1
        else:
            print(f"Example {i}: INCORRECT")
            p = np.array(pred)
            o = np.array(out)
            diff = (p != o)
            print(f"  Diff count: {np.sum(diff)}")
            # Show diffs
            rows, cols = np.where(diff)
            if len(rows) > 0:
                print(f"  First diff at ({rows[0]},{cols[0]}): Pred {p[rows[0],cols[0]]}, Expected {o[rows[0],cols[0]]}")
                
    print(f"Result: {correct}/{total}")

if __name__ == "__main__":
    check()
