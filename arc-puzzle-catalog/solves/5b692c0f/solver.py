import json

def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    ARC puzzle 5b692c0f solver.
    
    Rule: For each region with 4s, find the axis and make the region
    mirror-symmetric by copying the pattern from one side to the other.
    For vertical axes: copy right pattern to left.
    For horizontal axes: copy bottom pattern to top.
    """
    grid = [row[:] for row in grid]
    
    # Find all cells that will be modified and mark them as visited
    # This prevents reprocessing cells that become non-zero after mirroring
    processed = set()
    
    def get_region(start_i, start_j):
        """Get all cells in a region containing non-zero values."""
        region = set()
        stack = [(start_i, start_j)]
        while stack:
            i, j = stack.pop()
            if (i, j) in region or i < 0 or i >= len(grid) or j < 0 or j >= len(grid[0]):
                continue
            if grid[i][j] == 0:
                continue
            region.add((i, j))
            for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                stack.append((i + di, j + dj))
        return region
    
    # Find all regions and process each
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if grid[i][j] != 0 and (i, j) not in processed:
                region = get_region(i, j)
                # Mark ALL cells in region as processed initially
                processed.update(region)
                
                # Get bounding box and find axes within this region
                rows = [r for r, c in region]
                cols = [c for r, c in region]
                min_row, max_row = min(rows), max(rows)
                min_col, max_col = min(cols), max(cols)
                
                # Mark entire bounding box as processed
                for r in range(min_row, max_row + 1):
                    for c in range(min_col, max_col + 1):
                        processed.add((r, c))
                
                # Count 4s in each row and column within this region
                rows_with_fours = {}
                cols_with_fours = {}
                
                for r, c in region:
                    if grid[r][c] == 4:
                        rows_with_fours[r] = rows_with_fours.get(r, 0) + 1
                        cols_with_fours[c] = cols_with_fours.get(c, 0) + 1
                
                # Pre-mark cells that will be modified by row axis mirroring
                for axis_row, count in rows_with_fours.items():
                    if count >= 3:
                        max_dist = max(axis_row - min_row, max_row - axis_row)
                        for r in range(max(0, axis_row - max_dist), min(len(grid), axis_row + max_dist + 1)):
                            for c in range(min_col, max_col + 1):
                                processed.add((r, c))
                
                # Pre-mark cells that will be modified by column axis mirroring
                for axis_col, count in cols_with_fours.items():
                    if count >= 3:
                        max_dist = max(axis_col, len(grid[0]) - 1 - axis_col)
                        for r in range(min_row, max_row + 1):
                            for c in range(max(0, axis_col - max_dist), min(len(grid[0]), axis_col + max_dist + 1)):
                                processed.add((r, c))
                
                # Apply mirroring for each significant axis
                for axis_col, count in cols_with_fours.items():
                    if count >= 3:
                        # Mirror left and right symmetrically around the axis
                        max_dist = max(axis_col, len(grid[0]) - 1 - axis_col)
                        
                        # Copy right pattern to left (symmetric around axis_col)
                        for r in range(min_row, max_row + 1):
                            for dist in range(1, max_dist + 1):
                                left_col = axis_col - dist
                                right_col = axis_col + dist
                                if left_col >= 0 and right_col < len(grid[r]):
                                    grid[r][left_col] = grid[r][right_col]
                
                for axis_row, count in rows_with_fours.items():
                    if count >= 3:
                        # Mirror top and bottom symmetrically around the axis
                        max_dist = max(axis_row - min_row, max_row - axis_row)
                        
                        # Copy top pattern to bottom (symmetric around axis_row)
                        for c in range(min_col, max_col + 1):
                            for dist in range(1, max_dist + 1):
                                top_row = axis_row - dist
                                bottom_row = axis_row + dist
                                if top_row >= 0 and bottom_row < len(grid):
                                    grid[bottom_row][c] = grid[top_row][c]
    
    return grid


if __name__ == "__main__":
    import sys
    
    # Load the task
    task_path = "~/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/5b692c0f.json".replace("~", "/Users/evanpieser")
    with open(task_path) as f:
        task = json.load(f)
    
    # Test all training examples
    print("Testing training examples:")
    all_pass = True
    for idx, example in enumerate(task['train']):
        result = solve(example['input'])
        expected = example['output']
        
        if result == expected:
            print(f"  Training {idx+1}: PASS")
        else:
            print(f"  Training {idx+1}: FAIL")
            all_pass = False
            
            # Show first difference
            for i in range(len(result)):
                if result[i] != expected[i]:
                    print(f"    First difference at row {i}")
                    print(f"    Got:      {result[i]}")
                    print(f"    Expected: {expected[i]}")
                    break
    
    if all_pass:
        print("\n✓ All training examples passed!")
        sys.exit(0)
    else:
        print("\n✗ Some training examples failed")
        sys.exit(1)
