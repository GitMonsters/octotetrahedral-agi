def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    ARC-AGI task 2f0c5170 solver
    
    Pattern: 
    1. Find two rectangular regions (surrounded by 8s)
    2. One has just a colored marker (2, 3, or 1) - the target region
    3. Other has a pattern with 4s and the same marker - the pattern region
    4. Overlay the pattern onto the target, aligning the markers
    """
    
    def find_rectangular_regions(grid):
        regions = []
        rows, cols = len(grid), len(grid[0])
        visited = [[False] * cols for _ in range(rows)]
        
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] != 8 and not visited[r][c]:
                    # Find bounds of rectangular region
                    min_r, max_r = r, r
                    min_c, max_c = c, c
                    
                    # Find rightmost extent
                    while max_c + 1 < cols and grid[r][max_c + 1] != 8:
                        max_c += 1
                    
                    # Find bottom extent
                    while max_r + 1 < rows and grid[max_r + 1][c] != 8:
                        max_r += 1
                    
                    # Verify it's a complete rectangle
                    is_rect = True
                    for rr in range(min_r, max_r + 1):
                        for cc in range(min_c, max_c + 1):
                            if grid[rr][cc] == 8:
                                is_rect = False
                                break
                        if not is_rect:
                            break
                    
                    if is_rect:
                        # Extract region
                        region = []
                        for rr in range(min_r, max_r + 1):
                            row = []
                            for cc in range(min_c, max_c + 1):
                                row.append(grid[rr][cc])
                            region.append(row)
                        
                        regions.append(region)
                        
                        # Mark as visited
                        for rr in range(min_r, max_r + 1):
                            for cc in range(min_c, max_c + 1):
                                visited[rr][cc] = True
        
        return regions
    
    regions = find_rectangular_regions(grid)
    
    if len(regions) < 2:
        return [[0] * 9 for _ in range(9)]
    
    # Find target region (has single non-zero color) and pattern region (has 4s + another color)
    target_region = None
    pattern_region = None
    marker_color = None
    
    for region in regions:
        colors = set()
        for row in region:
            for val in row:
                if val != 0:
                    colors.add(val)
        
        if len(colors) == 1 and 4 not in colors:
            target_region = region
            marker_color = list(colors)[0]
        elif 4 in colors and len(colors) >= 2:
            pattern_region = region
            if marker_color is None:
                marker_color = next(c for c in colors if c != 4)
    
    if not target_region or not pattern_region:
        return [[0] * 9 for _ in range(9)]
    
    # Find marker positions
    target_marker_pos = None
    for r in range(len(target_region)):
        for c in range(len(target_region[0])):
            if target_region[r][c] == marker_color:
                target_marker_pos = (r, c)
                break
        if target_marker_pos:
            break
    
    pattern_marker_pos = None
    for r in range(len(pattern_region)):
        for c in range(len(pattern_region[0])):
            if pattern_region[r][c] == marker_color:
                pattern_marker_pos = (r, c)
                break
        if pattern_marker_pos:
            break
    
    if not target_marker_pos or not pattern_marker_pos:
        return [[0] * 9 for _ in range(9)]
    
    # Calculate offset to align markers
    offset_r = target_marker_pos[0] - pattern_marker_pos[0]
    offset_c = target_marker_pos[1] - pattern_marker_pos[1]
    
    # Create result by copying target region
    result = [row[:] for row in target_region]
    
    # Overlay pattern, aligning markers
    for r in range(len(pattern_region)):
        for c in range(len(pattern_region[0])):
            target_r = r + offset_r
            target_c = c + offset_c
            
            if (0 <= target_r < len(result) and 
                0 <= target_c < len(result[0]) and
                pattern_region[r][c] != 0):
                result[target_r][target_c] = pattern_region[r][c]
    
    return result


if __name__ == "__main__":
    import json
    with open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/2f0c5170.json") as f:
        task = json.load(f)
    for i, ex in enumerate(task["train"]):
        result = solve(ex["input"])
        status = "PASS" if result == ex["output"] else "FAIL"
        print(f"Train {i}: {status}")