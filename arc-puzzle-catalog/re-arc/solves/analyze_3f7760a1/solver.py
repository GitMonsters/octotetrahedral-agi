import json
import numpy as np

def get_bg_color(grid):
    counts = {}
    rows, cols = grid.shape
    for r in range(rows):
        for c in range(cols):
            val = grid[r, c]
            counts[val] = counts.get(val, 0) + 1
    return max(counts, key=counts.get)

def get_objects(grid):
    bg_color = get_bg_color(grid)
    visited = set()
    objects = []
    rows, cols = grid.shape
    for r in range(rows):
        for c in range(cols):
            color = grid[r, c]
            if color != bg_color and (r, c) not in visited:
                # BFS to find connected component (8-connectivity)
                obj_coords = []
                queue = [(r, c)]
                visited.add((r, c))
                while queue:
                    curr_r, curr_c = queue.pop(0)
                    obj_coords.append((curr_r, curr_c))
                    # Check 8 neighbors
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0: continue
                            nr, nc = curr_r + dr, curr_c + dc
                            if 0 <= nr < rows and 0 <= nc < cols:
                                if grid[nr, nc] == color and (nr, nc) not in visited:
                                    visited.add((nr, nc))
                                    queue.append((nr, nc))
                objects.append({'color': color, 'coords': obj_coords})
    return objects

def analyze_task(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    for i, pair in enumerate(data['train']):
        print(f"--- Example {i} ---")
        input_grid = np.array(pair['input'])
        output_grid = np.array(pair['output'])
        
        # Match objects
        # Note: simplistic matching by color and rough location
        # Better: analyze input objects and see what they become in output
        
        in_objects = get_objects(input_grid)
        
        for obj in in_objects:
            color = obj['color']
            coords = obj['coords']
            rs = [r for r, c in coords]
            cs = [c for r, c in coords]
            min_r, max_r = min(rs), max(rs)
            min_c, max_c = min(cs), max(cs)
            
            print(f"Object Color {color}: Bounds ({min_r}-{max_r}, {min_c}-{max_c})")
            
            # Extract Input and Output Subgrids for this bounding box
            # To see growth, we might need to look slightly outside the bounding box in output
            # But let's start with the input bounding box expanded by 1
            
            check_min_r = max(0, min_r - 1)
            check_max_r = min(input_grid.shape[0] - 1, max_r + 1)
            check_min_c = max(0, min_c - 1)
            check_max_c = min(input_grid.shape[1] - 1, max_c + 1)
            
            print("  Row analysis:")
            width_map = {}
            max_width = 0
            widest_row_idx = -1
            
            # Analyze Input Rows
            row_patterns = {}
            for r in range(min_r, max_r + 1):
                # Get pixels in this row belonging to the object
                row_pixels = sorted([c for cr, c in coords if cr == r])
                width = len(row_pixels)
                row_patterns[r] = row_pixels
                if width > max_width:
                    max_width = width
                    widest_row_idx = r
                elif width == max_width:
                    # If tie, which one?
                    pass
            
            print(f"    Widest Input Row: {widest_row_idx} (Width {max_width})")
            
            # Compare with Output
            # We look at the output grid in the same general area
            # We assume the object doesn't move, just transforms
            
            for r in range(check_min_r, check_max_r + 1):
                # Check for pixels of this color in output
                out_cols = []
                for c in range(check_min_c, check_max_c + 1):
                    if output_grid[r, c] == color:
                        out_cols.append(c)
                
                in_cols = row_patterns.get(r, [])
                
                status = ""
                if not in_cols and not out_cols:
                    continue # Empty row
                
                if in_cols == out_cols:
                    status = "SAME"
                elif not in_cols and out_cols:
                    status = "NEW "
                elif in_cols and not out_cols:
                    status = "GONE"
                else:
                    status = "CHNG"
                    
                print(f"    Row {r}: In {in_cols} -> Out {out_cols} [{status}]")

if __name__ == "__main__":
    analyze_task('/Users/evanpieser/re_arc_solves/3f7760a1_task.json')
