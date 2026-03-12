import json
import sys
from collections import Counter, deque

def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Solve ARC-AGI task 4b6b68e5.
    
    Algorithm:
    1. For each color C that appears >= 4 times (candidate border):
    2.  Find the bounding box of C
    3.  Identify all non-zero, non-C values inside the bounding box (interior special values)
    4.  If any exist, determine fill_color = most common special value
    5.  Flood-fill all 0-regions inside the bounding box with fill_color, and replace special values with fill_color
    """
    
    grid = [row[:] for row in grid]
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0
    
    if height == 0 or width == 0:
        return grid
    
    # Identify candidate border colors
    all_colors = set()
    for row in grid:
        for val in row:
            if val > 0:
                all_colors.add(val)
    
    # Process each color as a potential border
    for border_color in sorted(all_colors):
        # Find all cells with this color
        border_cells = []
        for i in range(height):
            for j in range(width):
                if grid[i][j] == border_color:
                    border_cells.append((i, j))
        
        # Must be a substantial border
        if len(border_cells) < 4:
            continue
        
        # Find bounding box
        min_r = min(b[0] for b in border_cells)
        max_r = max(b[0] for b in border_cells)
        min_c = min(b[1] for b in border_cells)
        max_c = max(b[1] for b in border_cells)
        
        # Find interior special values (non-zero, non-border) inside bounding box
        interior_vals = []
        for i in range(min_r, max_r + 1):
            for j in range(min_c, max_c + 1):
                if grid[i][j] != 0 and grid[i][j] != border_color:
                    interior_vals.append(grid[i][j])
        
        if not interior_vals:
            # No special values to fill with
            continue
        
        # Determine fill color
        fill_color = Counter(interior_vals).most_common(1)[0][0]
        
        # Now fill: replace all 0s and non-border values inside the bounding box
        # But be careful not to cross other border colors
        # Strategy: for each non-border cell in bounding box, if it's reachable from border through non-border cells, fill it
        
        visited = set()
        
        def fill_region(start_r, start_c):
            """Fill connected region starting from (start_r, start_c)"""
            if (start_r, start_c) in visited:
                return
            
            queue = deque([(start_r, start_c)])
            visited.add((start_r, start_c))
            
            while queue:
                r, c = queue.popleft()
                
                # Fill this cell if not a border
                if grid[r][c] != border_color:
                    grid[r][c] = fill_color
                
                # Expand to neighbors (within bounding box and not other borders)
                for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                    nr, nc = r + dr, c + dc
                    if (min_r <= nr <= max_r and min_c <= nc <= max_c and
                        (nr, nc) not in visited):
                        # Can move to 0s or non-border values
                        # Don't move to other borders (cells not equal to border_color or 0 or border_color won't work...)
                        # Actually: can move to any cell that's not a border (grid[nr][nc] != border_color) 
                        # But this includes other borders!
                        # Let's be more careful: only move through 0s and the fill_color itself
                        if grid[nr][nc] == 0 or grid[nr][nc] == fill_color or (grid[nr][nc] != border_color and grid[nr][nc] in interior_vals):
                            visited.add((nr, nc))
                            queue.append((nr, nc))
        
        # Start fill from each cell adjacent to the border
        for i, j in border_cells:
            for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                nr, nc = i + dr, j + dc
                if (min_r <= nr <= max_r and min_c <= nc <= max_c and grid[nr][nc] != border_color):
                    fill_region(nr, nc)
    
    return grid


def main():
    if len(sys.argv) > 1:
        task_path = sys.argv[1]
    else:
        task_path = '/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/4b6b68e5.json'
    
    with open(task_path, 'r') as f:
        task_data = json.load(f)
    
    # Test on training examples
    all_pass = True
    for idx, example in enumerate(task_data['train']):
        input_grid = example['input']
        expected_output = example['output']
        
        predicted_output = solve(input_grid)
        
        if predicted_output == expected_output:
            print(f"Training example {idx}: PASS")
        else:
            print(f"Training example {idx}: FAIL")
            all_pass = False
            # Show first difference
            for i in range(len(expected_output)):
                for j in range(len(expected_output[i])):
                    if predicted_output[i][j] != expected_output[i][j]:
                        print(f"  First diff at ({i}, {j}): got {predicted_output[i][j]}, expected {expected_output[i][j]}")
                        break
                else:
                    continue
                break
    
    if all_pass:
        print("\nAll training examples passed!")
    else:
        print("\nSome training examples failed.")
    
    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
