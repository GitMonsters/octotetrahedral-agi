def solve(grid: list[list[int]]) -> list[list[int]]:
    import copy
    result = copy.deepcopy(grid)
    rows, cols = len(grid), len(grid[0])
    
    # Find the line of 1s - could be horizontal or vertical
    line_row = None
    line_col = None
    
    # Check for horizontal line of 1s
    for r in range(rows):
        if all(grid[r][c] == 1 for c in range(cols)):
            line_row = r
            break
    
    # Check for vertical line of 1s
    for c in range(cols):
        if all(grid[r][c] == 1 for r in range(rows)):
            line_col = c
            break
    
    if line_row is not None or line_col is not None:
        # Find all unique colors in the pattern
        all_colors = set()
        for r in range(rows):
            for c in range(cols):
                val = grid[r][c]
                if val != 0 and val != 1:
                    all_colors.add(val)
        
        if len(all_colors) == 2:
            color_list = sorted(list(all_colors))
            color_map = {color_list[0]: color_list[1], color_list[1]: color_list[0]}
            
            # Apply color transformation everywhere
            for r in range(rows):
                for c in range(cols):
                    if result[r][c] in color_map:
                        result[r][c] = color_map[result[r][c]]
    
    if line_row is not None:
        # Check which side has the pattern for reflection
        pattern_above = any(grid[r][c] != 0 and grid[r][c] != 1 
                           for r in range(line_row) for c in range(cols))
        pattern_below = any(grid[r][c] != 0 and grid[r][c] != 1 
                           for r in range(line_row + 1, rows) for c in range(cols))
        
        if pattern_above and not pattern_below:
            # Reflect pattern from above to below
            for r in range(line_row):
                for c in range(cols):
                    if grid[r][c] != 0 and grid[r][c] != 1:
                        mirror_r = 2 * line_row - r
                        if 0 <= mirror_r < rows:
                            result[mirror_r][c] = grid[r][c]
                            
        elif pattern_below and not pattern_above:
            # Reflect pattern from below to above  
            for r in range(line_row + 1, rows):
                for c in range(cols):
                    if grid[r][c] != 0 and grid[r][c] != 1:
                        mirror_r = 2 * line_row - r
                        if 0 <= mirror_r < rows:
                            result[mirror_r][c] = grid[r][c]
    
    elif line_col is not None:
        # Check which side has the pattern for reflection
        pattern_left = any(grid[r][c] != 0 and grid[r][c] != 1 
                          for r in range(rows) for c in range(line_col))
        pattern_right = any(grid[r][c] != 0 and grid[r][c] != 1 
                           for r in range(rows) for c in range(line_col + 1, cols))
        
        if pattern_right and not pattern_left:
            # Reflect pattern from right to left
            for r in range(rows):
                for c in range(line_col + 1, cols):
                    if grid[r][c] != 0 and grid[r][c] != 1:
                        mirror_c = 2 * line_col - c
                        if 0 <= mirror_c < cols:
                            result[r][mirror_c] = grid[r][c]
                        
        elif pattern_left and not pattern_right:
            # Reflect pattern from left to right
            for r in range(rows):
                for c in range(line_col):
                    if grid[r][c] != 0 and grid[r][c] != 1:
                        mirror_c = 2 * line_col - c
                        if 0 <= mirror_c < cols:
                            result[r][mirror_c] = grid[r][c]
    
    return result

if __name__ == "__main__":
    import json
    with open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/2b01abd0.json") as f:
        task = json.load(f)
    for i, ex in enumerate(task["train"]):
        result = solve(ex["input"])
        status = "PASS" if result == ex["output"] else "FAIL"
        print(f"Train {i}: {status}")
        if status == "FAIL":
            print(f"Expected: {ex['output']}")
            print(f"Got: {result}")