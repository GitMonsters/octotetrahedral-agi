def solve(grid: list[list[int]]) -> list[list[int]]:
    import copy
    result = copy.deepcopy(grid)
    rows, cols = len(grid), len(grid[0])
    
    # Find pattern in first row
    pattern_start = None
    pattern_end = None
    for c in range(cols):
        if grid[0][c] != 0:
            if pattern_start is None:
                pattern_start = c
            pattern_end = c
    
    if pattern_start is None:
        return result
    
    # Extract pattern  
    pattern = []
    for c in range(pattern_start, pattern_end + 1):
        pattern.append(grid[0][c])
    
    # Find isolated instances (colors that appear elsewhere in the grid)
    pattern_colors = set(pattern)
    isolated = {}
    for r in range(1, rows):
        for c in range(cols):
            if grid[r][c] in pattern_colors:
                color = grid[r][c]
                if color not in isolated:
                    isolated[color] = []
                isolated[color].append((r, c))
    
    # Color to pattern column mapping
    color_to_col = {}
    for i, color in enumerate(pattern):
        color_to_col[color] = pattern_start + i
    
    max_row = max(r for positions in isolated.values() for r, c in positions) if isolated else 0
    
    # Step 1: Replicate full pattern to ALL rows up to max_row
    for r in range(1, max_row + 1):
        for i, color in enumerate(pattern):
            result[r][pattern_start + i] = color
    
    # Step 2: Handle colors with isolated instances
    for color, positions in isolated.items():
        col = color_to_col[color]
        
        # Create horizontal extensions
        for iso_r, iso_c in positions:
            if iso_c > col:
                # Extend right from pattern column
                for c in range(col, cols):
                    if result[iso_r][c] == 0:
                        result[iso_r][c] = color
            elif iso_c < col:
                # Extend left from isolated position to pattern column
                for c in range(iso_c, col + 1):
                    if result[iso_r][c] == 0:
                        result[iso_r][c] = color
        
        # Clear this color from its pattern column in ALL rows after its first isolated occurrence
        first_isolated_row = min(r for r, c in positions)
        for r in range(first_isolated_row + 1, max_row + 1):
            result[r][col] = 0
    
    # Step 3: Extend middle column down to bottom ONLY if it doesn't have isolated instances
    middle_col = pattern_start + len(pattern)//2
    middle_color = pattern[len(pattern)//2]
    
    # Only extend if the middle color has no isolated instances
    if middle_color not in isolated:
        for r in range(max_row + 1, rows):
            result[r][middle_col] = middle_color
    
    return result

if __name__ == "__main__":
    import json
    with open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/29700607.json") as f:
        task = json.load(f)
    for i, ex in enumerate(task["train"]):
        result = solve(ex["input"])
        status = "PASS" if result == ex["output"] else "FAIL"
        print(f"Train {i}: {status}")