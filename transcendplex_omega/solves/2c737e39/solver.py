def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    ARC-AGI task 2c737e39 solver:
    1. Find the main pattern (largest connected component)
    2. Find the lone 5 (single isolated 5 cell)
    3. Remove the lone 5 and copy the pattern (excluding its 5) to align with where the lone 5 was
    """
    import copy
    
    result = copy.deepcopy(grid)
    rows, cols = len(grid), len(grid[0])
    
    # Find all non-zero positions
    non_zero_positions = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                non_zero_positions.append((r, c))
    
    if len(non_zero_positions) == 0:
        return result
    
    # Find connected components using 8-connected flood fill
    visited = set()
    components = []
    
    def flood_fill(start_r, start_c):
        stack = [(start_r, start_c)]
        component = []
        while stack:
            r, c = stack.pop()
            if (r, c) in visited or r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] == 0:
                continue
            visited.add((r, c))
            component.append((r, c))
            # Check 8-connected neighbors
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if (nr, nc) not in visited:
                        stack.append((nr, nc))
        return component
    
    for r, c in non_zero_positions:
        if (r, c) not in visited:
            component = flood_fill(r, c)
            if component:
                components.append(component)
    
    # Find the main pattern (largest component)
    main_pattern = max(components, key=len)
    
    # Find the lone 5 (single cell component with value 5)
    lone_5 = None
    for component in components:
        if len(component) == 1:
            r, c = component[0]
            if grid[r][c] == 5:
                lone_5 = (r, c)
                break
    
    if lone_5 is None:
        return result
    
    # Find the 5 in the main pattern
    pattern_5 = None
    for r, c in main_pattern:
        if grid[r][c] == 5:
            pattern_5 = (r, c)
            break
    
    if pattern_5 is None:
        return result
    
    # Remove the lone 5
    lone_r, lone_c = lone_5
    result[lone_r][lone_c] = 0
    
    # Calculate offset to place pattern so its 5 position aligns with lone 5 position
    pattern_r, pattern_c = pattern_5
    offset_r = lone_r - pattern_r
    offset_c = lone_c - pattern_c
    
    # Copy the main pattern (excluding its 5) to the new location
    for r, c in main_pattern:
        if (r, c) != pattern_5:  # Skip the 5 in the pattern
            new_r = r + offset_r
            new_c = c + offset_c
            if 0 <= new_r < rows and 0 <= new_c < cols:
                result[new_r][new_c] = grid[r][c]
    
    return result

if __name__ == "__main__":
    import json
    with open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/2c737e39.json") as f:
        task = json.load(f)
    for i, ex in enumerate(task["train"]):
        result = solve(ex["input"])
        status = "PASS" if result == ex["output"] else "FAIL"
        print(f"Train {i}: {status}")
        if status == "FAIL":
            print(f"Expected: {ex['output']}")
            print(f"Got: {result}")