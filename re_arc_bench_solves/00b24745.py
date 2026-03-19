def transform(grid):
    """
    Pattern: Find rectangular clusters of a 'marker' color. Within each cluster's
    bounding box, replace a specific 'target' background color with 0 (black).
    """
    import copy
    
    rows = len(grid)
    cols = len(grid[0])
    result = copy.deepcopy(grid)
    
    # Count colors
    color_counts = {}
    for r in range(rows):
        for c in range(cols):
            v = grid[r][c]
            color_counts[v] = color_counts.get(v, 0) + 1
    
    # Get the 3 main colors
    colors = sorted(color_counts.keys(), key=lambda x: -color_counts[x])
    if len(colors) < 3:
        return result
    
    # The marker color is the least frequent, target is medium
    marker_color = colors[2]
    target_color = colors[1]  # This will be replaced with 0 inside boxes
    
    # Find connected components of marker color
    visited = [[False]*cols for _ in range(rows)]
    
    def flood_fill(r, c):
        """Find all cells in this connected component"""
        stack = [(r, c)]
        cells = []
        while stack:
            cr, cc = stack.pop()
            if cr < 0 or cr >= rows or cc < 0 or cc >= cols:
                continue
            if visited[cr][cc] or grid[cr][cc] != marker_color:
                continue
            visited[cr][cc] = True
            cells.append((cr, cc))
            # 4-connectivity
            stack.extend([(cr-1, cc), (cr+1, cc), (cr, cc-1), (cr, cc+1)])
        return cells
    
    # Find all components and their bounding boxes
    components = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == marker_color and not visited[r][c]:
                cells = flood_fill(r, c)
                if cells:
                    min_r = min(x[0] for x in cells)
                    max_r = max(x[0] for x in cells)
                    min_c = min(x[1] for x in cells)
                    max_c = max(x[1] for x in cells)
                    components.append((min_r, max_r, min_c, max_c))
    
    # Within each bounding box, replace target_color with 0
    for min_r, max_r, min_c, max_c in components:
        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                if result[r][c] == target_color:
                    result[r][c] = 0
    
    return result


if __name__ == "__main__":
    import json
    
    # Load puzzle data
    with open('/Users/evanpieser/Downloads/re-arc_test_challenges-2026-03-16T22-48-53.json') as f:
        data = json.load(f)
    
    puzzle = data['00b24745']
    
    all_passed = True
    for i, ex in enumerate(puzzle['train']):
        output = transform(ex['input'])
        expected = ex['output']
        passed = output == expected
        print(f"Train {i}: {'PASS' if passed else 'FAIL'}")
        if not passed:
            all_passed = False
            # Show first difference
            for r in range(len(expected)):
                for c in range(len(expected[0])):
                    if output[r][c] != expected[r][c]:
                        print(f"  Diff at ({r},{c}): got {output[r][c]}, expected {expected[r][c]}")
                        break
                else:
                    continue
                break
    
    print(f"\nAll passed: {all_passed}")
