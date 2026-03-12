"""
ARC Puzzle d931c21c Solver

Rule: For each connected component of 1s containing holes:
1. For each hole, cells that touch the 1-boundary get color 2 or 3
2. Cells in holes that don't touch any 1 stay as 0
3. For the main component (largest interior hole), draw a border of 2
"""

from collections import deque


def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Solve the ARC puzzle d931c21c.
    
    Args:
        grid: Input grid as list of lists
    
    Returns:
        Transformed grid
    """
    if not grid or not grid[0]:
        return grid
    
    height = len(grid)
    width = len(grid[0])
    
    # First, find all exterior 0s (reachable from boundary)
    visited_ext = set()
    
    def bfs_exterior(start_r, start_c):
        """Flood fill from an exterior 0."""
        queue = deque([(start_r, start_c)])
        visited_ext.add((start_r, start_c))
        
        while queue:
            r, c = queue.popleft()
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if (0 <= nr < height and 0 <= nc < width and 
                    (nr, nc) not in visited_ext and grid[nr][nc] == 0):
                    visited_ext.add((nr, nc))
                    queue.append((nr, nc))
    
    # Find exterior 0-component starting from boundary
    for c in range(width):
        if grid[0][c] == 0 and (0, c) not in visited_ext:
            bfs_exterior(0, c)
    for c in range(width):
        if grid[height-1][c] == 0 and (height-1, c) not in visited_ext:
            bfs_exterior(height-1, c)
    for r in range(height):
        if grid[r][0] == 0 and (r, 0) not in visited_ext:
            bfs_exterior(r, 0)
    for r in range(height):
        if grid[r][width-1] == 0 and (r, width-1) not in visited_ext:
            bfs_exterior(r, width-1)
    
    # Check if there are any holes
    has_holes = any((r, c) not in visited_ext 
                    for r in range(height) 
                    for c in range(width) 
                    if grid[r][c] == 0)
    
    # If no holes, return unchanged
    if not has_holes:
        return grid
    
    # Make a copy
    result = [row[:] for row in grid]
    
    # Find all 1-components
    visited_1 = set()
    
    def bfs_1(start_r, start_c):
        queue = deque([(start_r, start_c)])
        visited_1.add((start_r, start_c))
        cells = [(start_r, start_c)]
        
        while queue:
            r, c = queue.popleft()
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if (0 <= nr < height and 0 <= nc < width and 
                    (nr, nc) not in visited_1 and grid[nr][nc] == 1):
                    visited_1.add((nr, nc))
                    queue.append((nr, nc))
                    cells.append((nr, nc))
        return cells
    
    # Process all 1-components
    visited_1.clear()
    for r in range(height):
        for c in range(width):
            if grid[r][c] == 1 and (r, c) not in visited_1:
                component = bfs_1(r, c)
                
                min_r = min(row for row, col in component)
                max_r = max(row for row, col in component)
                min_c = min(col for row, col in component)
                max_c = max(col for row, col in component)
                
                # Find interior 0-components
                visited_interior = set()
                
                def bfs_interior(start_r, start_c):
                    queue = deque([(start_r, start_c)])
                    visited_interior.add((start_r, start_c))
                    cells = [(start_r, start_c)]
                    
                    while queue:
                        cr, cc = queue.popleft()
                        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                            nr, nc = cr + dr, cc + dc
                            if (min_r <= nr <= max_r and min_c <= nc <= max_c and 
                                (nr, nc) not in visited_interior and grid[nr][nc] == 0):
                                visited_interior.add((nr, nc))
                                queue.append((nr, nc))
                                cells.append((nr, nc))
                    return cells
                
                interior_components = []
                for int_r in range(min_r, max_r + 1):
                    for int_c in range(min_c, max_c + 1):
                        if (grid[int_r][int_c] == 0 and 
                            (int_r, int_c) not in visited_interior):
                            int_comp = bfs_interior(int_r, int_c)
                            interior_components.append(int_comp)
                
                # Process each hole
                for int_comp in interior_components:
                    touches_border = any(
                        r == min_r or r == max_r or c == min_c or c == max_c
                        for r, c in int_comp
                    )
                    
                    # Find 0s that don't touch 1s
                    interior_0s = set()
                    for r, c in int_comp:
                        touches_1 = False
                        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                            nr, nc = r + dr, c + dc
                            if (0 <= nr < height and 0 <= nc < width and 
                                grid[nr][nc] == 1):
                                touches_1 = True
                                break
                        if not touches_1:
                            interior_0s.add((r, c))
                    
                    # Color the hole
                    for r, c in int_comp:
                        if (r, c) in interior_0s:
                            result[r][c] = 0
                        elif touches_border:
                            result[r][c] = 2
                        else:
                            result[r][c] = 3
                
                # Check if this component has interior holes (doesn't touch border)
                has_interior_hole = any(
                    not any(r == min_r or r == max_r or c == min_c or c == max_c
                           for r, c in int_comp)
                    for int_comp in interior_components
                )
                
                # Draw border for components with interior holes
                if has_interior_hole:
                    # Draw border in a complete frame around the component
                    for r in range(min_r - 1, max_r + 2):
                        for c in range(min_c - 1, max_c + 2):
                            if (0 <= r < height and 0 <= c < width and 
                                result[r][c] == 0 and not (min_r <= r <= max_r and min_c <= c <= max_c)):
                                # This is a border cell (outside the component bbox)
                                result[r][c] = 2
    
    return result


if __name__ == "__main__":
    import json
    
    with open('/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/d931c21c.json') as f:
        data = json.load(f)
    
    all_pass = True
    
    for idx, example in enumerate(data['train']):
        inp = example['input']
        expected = example['output']
        result = solve(inp)
        
        if result == expected:
            print(f"✓ Training example {idx + 1} PASSED")
        else:
            print(f"✗ Training example {idx + 1} FAILED")
            all_pass = False
            
            for r in range(len(result)):
                for c in range(len(result[0])):
                    if result[r][c] != expected[r][c]:
                        print(f"  ({r},{c}): got {result[r][c]}, expected {expected[r][c]}")
    
    if all_pass:
        print("\n✓ All training examples passed!")
    else:
        print("\n✗ Some examples failed")
