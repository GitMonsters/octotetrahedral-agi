def solve(grid: list[list[int]]) -> list[list[int]]:
    """Find the largest solid rectangular region closest to center and return 3x3 grid filled with that color."""
    
    # Find all colors except 0 (background)
    colors = set()
    for row in grid:
        for cell in row:
            if cell != 0:
                colors.add(cell)
    
    best_area = 0
    best_distance = float('inf')
    best_color = 0
    
    grid_center_r = (len(grid) - 1) / 2  # 9.5 for 20x20 grid
    grid_center_c = (len(grid[0]) - 1) / 2  # 9.5 for 20x20 grid
    
    # For each color, find the largest solid rectangle
    for color in colors:
        # Try all possible top-left corners
        for r1 in range(len(grid)):
            for c1 in range(len(grid[0])):
                if grid[r1][c1] == color:
                    # Try expanding right and down to form rectangles
                    max_c2 = c1
                    # Find max width from this starting point
                    while max_c2 + 1 < len(grid[0]) and grid[r1][max_c2 + 1] == color:
                        max_c2 += 1
                    
                    # For each possible width, find max height
                    for c2 in range(c1, max_c2 + 1):
                        max_r2 = r1
                        # Find max height for this width
                        while max_r2 + 1 < len(grid):
                            # Check if entire row from c1 to c2 is this color
                            valid = True
                            for cc in range(c1, c2 + 1):
                                if grid[max_r2 + 1][cc] != color:
                                    valid = False
                                    break
                            if not valid:
                                break
                            max_r2 += 1
                        
                        # Calculate area and distance to center
                        width = c2 - c1 + 1
                        height = max_r2 - r1 + 1
                        area = width * height
                        
                        # Must be at least 3x3
                        if width >= 3 and height >= 3:
                            # Calculate center of this rectangle
                            rect_center_r = (r1 + max_r2) / 2
                            rect_center_c = (c1 + c2) / 2
                            distance = abs(rect_center_r - grid_center_r) + abs(rect_center_c - grid_center_c)
                            
                            # Update best if: larger area, or same area but closer to center
                            if (area > best_area or 
                                (area == best_area and distance < best_distance)):
                                best_area = area
                                best_distance = distance
                                best_color = color
    
    # Return 3x3 grid filled with the best color
    return [[best_color, best_color, best_color],
            [best_color, best_color, best_color],
            [best_color, best_color, best_color]]

if __name__ == "__main__":
    import json
    with open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/3194b014.json") as f:
        task = json.load(f)
    for i, ex in enumerate(task["train"]):
        result = solve(ex["input"])
        status = "PASS" if result == ex["output"] else "FAIL"
        print(f"Train {i}: {status}")
        if status == "FAIL":
            print(f"  Expected: {ex['output']}")
            print(f"  Got: {result}")