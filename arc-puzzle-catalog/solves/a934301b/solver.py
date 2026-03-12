"""
ARC Puzzle a934301b Solver

Rule: Remove connected regions (components) that contain 2 or more cells with color 8.
Keep regions with 0 or 1 cell of color 8.

The puzzle is a color_swap task on a 14-15x14 grid with 3 colors (0, 1, 8).
"""

def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Solve ARC puzzle a934301b by removing corrupted regions.
    
    Args:
        grid: Input grid as list of lists of integers
        
    Returns:
        Output grid with regions containing multiple 8s removed (set to 0)
    """
    # Create output grid (copy of input)
    output = [row[:] for row in grid]
    
    # Find all connected regions using BFS
    visited = set()
    
    def get_region(si: int, sj: int) -> set:
        """BFS to find connected component of non-zero values"""
        region = set()
        q = [(si, sj)]
        while q:
            i, j = q.pop(0)
            if (i, j) in visited or (i, j) in region:
                continue
            if i < 0 or i >= len(grid) or j < 0 or j >= len(grid[0]):
                continue
            if grid[i][j] == 0:
                continue
            region.add((i, j))
            for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                q.append((i + di, j + dj))
        return region
    
    # Find all regions
    regions = []
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] != 0 and (i, j) not in visited:
                region = get_region(i, j)
                visited.update(region)
                regions.append(region)
    
    # Remove regions with 2+ eights (corrupted patterns)
    for region in regions:
        eight_count = sum(1 for i, j in region if grid[i][j] == 8)
        if eight_count >= 2:
            # Mark all cells in this region for removal
            for i, j in region:
                output[i][j] = 0
    
    return output


if __name__ == "__main__":
    import json
    
    # Load test data
    with open('/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/a934301b.json') as f:
        data = json.load(f)
    
    # Test on all training examples
    print("Testing ARC puzzle a934301b solver...")
    all_pass = True
    
    for idx, ex in enumerate(data['train']):
        result = solve(ex['input'])
        expected = ex['output']
        
        if result == expected:
            print(f"✓ Training Example {idx + 1}: PASS")
        else:
            print(f"✗ Training Example {idx + 1}: FAIL")
            all_pass = False
    
    if all_pass:
        print("\n✓ ALL TRAINING EXAMPLES PASS!")
    else:
        print("\n✗ SOME TRAINING EXAMPLES FAILED")
        exit(1)
