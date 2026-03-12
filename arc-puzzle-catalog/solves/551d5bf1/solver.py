import json

def find_rectangles(grid):
    """Find all rectangles bounded by 1s."""
    rows, cols = len(grid), len(grid[0])
    visited = [[False] * cols for _ in range(rows)]
    rectangles = []
    
    # Find all connected components of 1s
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1 and not visited[r][c]:
                # BFS to find all 1s in this component
                component = []
                queue = [(r, c)]
                visited[r][c] = True
                
                while queue:
                    cr, cc = queue.pop(0)
                    component.append((cr, cc))
                    
                    for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 1 and not visited[nr][nc]:
                            visited[nr][nc] = True
                            queue.append((nr, nc))
                
                if component:
                    # Get bounding box
                    min_r = min(p[0] for p in component)
                    max_r = max(p[0] for p in component)
                    min_c = min(p[1] for p in component)
                    max_c = max(p[1] for p in component)
                    
                    # Check if it forms a rectangle border pattern
                    if is_rectangle_border(grid, min_r, max_r, min_c, max_c, component):
                        rectangles.append({
                            'min_r': min_r,
                            'max_r': max_r,
                            'min_c': min_c,
                            'max_c': max_c,
                            'component': set(component)
                        })
    
    return rectangles

def is_rectangle_border(grid, min_r, max_r, min_c, max_c, component):
    """Check if component forms a rectangular border."""
    if min_r == max_r or min_c == max_c:
        return False  # Too small
    
    comp_set = set(component)
    
    # Count edges: check if we have significant border coverage
    edges_count = 0
    
    # Top and bottom edges
    for c in range(min_c, max_c + 1):
        if (min_r, c) in comp_set:
            edges_count += 1
        if (max_r, c) in comp_set:
            edges_count += 1
    
    # Left and right edges
    for r in range(min_r + 1, max_r):
        if (r, min_c) in comp_set:
            edges_count += 1
        if (r, max_c) in comp_set:
            edges_count += 1
    
    return edges_count >= (2 * (max_r - min_r + max_c - min_c) * 0.5)

def solve(grid):
    """Transform grid according to ARC-AGI task 551d5bf1 rules.
    
    Rules:
    1. Find rectangles outlined by 1s (may have gaps in the border)
    2. Fill their interiors (cells between the 1s) with 8s
    3. For any gap in the border (where 1 is expected but missing):
       - Extend 8s from that gap outward toward the grid edge
       - The extension continues until hitting a boundary (1, edge of grid, or another rectangle)
    """
    output = [row[:] for row in grid]
    rows, cols = len(grid), len(grid[0])
    
    # Find all rectangles
    rectangles = find_rectangles(grid)
    
    for rect in rectangles:
        min_r, max_r, min_c, max_c = rect['min_r'], rect['max_r'], rect['min_c'], rect['max_c']
        comp = rect['component']
        
        # Fill interior with 8
        for r in range(min_r + 1, max_r):
            for c in range(min_c + 1, max_c):
                if output[r][c] == 0:
                    output[r][c] = 8
        
        # Process TOP edge - look for gaps
        for c in range(min_c, max_c + 1):
            if (min_r, c) not in comp:
                # Gap at top edge - fill the gap itself and extend upward
                if output[min_r][c] == 0:
                    output[min_r][c] = 8
                for r in range(min_r - 1, -1, -1):
                    if output[r][c] == 0:
                        output[r][c] = 8
                    else:
                        break
        
        # Process BOTTOM edge - look for gaps
        for c in range(min_c, max_c + 1):
            if (max_r, c) not in comp:
                # Gap at bottom edge - fill the gap itself and extend downward
                if output[max_r][c] == 0:
                    output[max_r][c] = 8
                for r in range(max_r + 1, rows):
                    if output[r][c] == 0:
                        output[r][c] = 8
                    else:
                        break
        
        # Process LEFT edge - look for gaps
        for r in range(min_r, max_r + 1):
            if (r, min_c) not in comp:
                # Gap at left edge - fill the gap itself and extend leftward
                if output[r][min_c] == 0:
                    output[r][min_c] = 8
                for c in range(min_c - 1, -1, -1):
                    if output[r][c] == 0:
                        output[r][c] = 8
                    else:
                        break
        
        # Process RIGHT edge - look for gaps
        for r in range(min_r, max_r + 1):
            if (r, max_c) not in comp:
                # Gap at right edge - fill the gap itself and extend rightward
                if output[r][max_c] == 0:
                    output[r][max_c] = 8
                for c in range(max_c + 1, cols):
                    if output[r][c] == 0:
                        output[r][c] = 8
                    else:
                        break
    
    return output

if __name__ == "__main__":
    with open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/551d5bf1.json") as f:
        task = json.load(f)
    
    print("Testing training examples...")
    all_pass = True
    
    for idx, example in enumerate(task["train"]):
        input_grid = example["input"]
        expected = example["output"]
        result = solve(input_grid)
        
        match = result == expected
        status = "PASS" if match else "FAIL"
        print(f"  Train {idx}: {status}")
        
        if not match:
            all_pass = False
            # Debug: show first difference
            for r in range(len(expected)):
                for c in range(len(expected[0])):
                    if result[r][c] != expected[r][c]:
                        print(f"    Diff at ({r},{c}): got {result[r][c]}, expected {expected[r][c]}")
                        break
                if not all(result[r][c] == expected[r][c] for c in range(len(expected[0]))):
                    break
    
    if all_pass:
        print("\n✓ All training examples PASS")
    else:
        print("\n✗ Some training examples FAIL")
