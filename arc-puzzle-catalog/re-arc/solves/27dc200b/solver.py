"""
ARC Puzzle 27dc200b Solver

Pattern: A shape with marker pixels indicating direction to propagate/tile.
The shape is copied repeatedly in the marker's direction until hitting grid edge.
"""

def transform(grid):
    import numpy as np
    grid = np.array(grid)
    rows, cols = grid.shape
    
    # Find background color (most common)
    unique, counts = np.unique(grid, return_counts=True)
    bg_color = unique[np.argmax(counts)]
    
    # Find all non-background colors
    non_bg_colors = [c for c in unique if c != bg_color]
    if len(non_bg_colors) == 0:
        return grid.tolist()
    
    # Find positions for each non-background color
    color_positions = {}
    for color in non_bg_colors:
        positions = list(zip(*np.where(grid == color)))
        color_positions[color] = positions
    
    # Identify main shape color (most pixels) and marker color (fewer pixels)
    # Marker is typically a different color, or isolated pixels of same color
    color_counts = {c: len(pos) for c, pos in color_positions.items()}
    
    if len(non_bg_colors) == 1:
        # Same color for shape and marker - need to separate
        shape_color = non_bg_colors[0]
        marker_color = shape_color
    else:
        # Different colors - marker has fewer pixels
        sorted_colors = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)
        shape_color = sorted_colors[0][0]
        marker_color = sorted_colors[1][0]
    
    # Get shape and marker positions
    shape_positions = set(color_positions[shape_color])
    marker_positions = set(color_positions[marker_color]) if marker_color != shape_color else set()
    
    # If same color, find isolated pixels (not part of main connected component)
    if marker_color == shape_color:
        # Find main connected component
        all_pos = list(shape_positions)
        visited = set()
        components = []
        
        def flood_fill(start):
            component = set()
            stack = [start]
            while stack:
                pos = stack.pop()
                if pos in visited or pos not in shape_positions:
                    continue
                visited.add(pos)
                component.add(pos)
                r, c = pos
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                    nr, nc = r+dr, c+dc
                    if (nr, nc) in shape_positions and (nr, nc) not in visited:
                        stack.append((nr, nc))
            return component
        
        for pos in all_pos:
            if pos not in visited:
                comp = flood_fill(pos)
                components.append(comp)
        
        # Largest component is shape, rest are markers
        components.sort(key=len, reverse=True)
        shape_positions = components[0]
        marker_positions = set()
        for comp in components[1:]:
            marker_positions.update(comp)
    
    if not marker_positions:
        return grid.tolist()
    
    # Get shape bounding box
    shape_rows = [p[0] for p in shape_positions]
    shape_cols = [p[1] for p in shape_positions]
    shape_min_r, shape_max_r = min(shape_rows), max(shape_rows)
    shape_min_c, shape_max_c = min(shape_cols), max(shape_cols)
    shape_height = shape_max_r - shape_min_r + 1
    shape_width = shape_max_c - shape_min_c + 1
    
    # Determine direction from marker position relative to shape
    marker_r = sum(p[0] for p in marker_positions) / len(marker_positions)
    marker_c = sum(p[1] for p in marker_positions) / len(marker_positions)
    shape_center_r = (shape_min_r + shape_max_r) / 2
    shape_center_c = (shape_min_c + shape_max_c) / 2
    
    # Direction vector (normalized to -1, 0, or 1)
    dr = 1 if marker_r > shape_center_r else (-1 if marker_r < shape_center_r else 0)
    dc = 1 if marker_c > shape_center_c else (-1 if marker_c < shape_center_c else 0)
    
    # Create output grid
    output = grid.copy()
    
    # Remove marker pixels from shape pattern (we'll use shape_color for tiling)
    # Extract shape pattern relative to its bounding box
    shape_pattern = {}
    for r, c in shape_positions:
        shape_pattern[(r - shape_min_r, c - shape_min_c)] = grid[r, c]
    
    # Propagate shape in direction until out of bounds
    step = 1
    while True:
        new_min_r = shape_min_r + dr * shape_height * step
        new_min_c = shape_min_c + dc * shape_width * step
        
        # Check if any part of new shape would be in bounds
        in_bounds = False
        for (rel_r, rel_c), color in shape_pattern.items():
            new_r = new_min_r + rel_r
            new_c = new_min_c + rel_c
            if 0 <= new_r < rows and 0 <= new_c < cols:
                in_bounds = True
                break
        
        if not in_bounds:
            break
        
        # Place the shape copy (use marker_color for new copies)
        tile_color = marker_color if marker_color != shape_color else shape_color
        for (rel_r, rel_c), _ in shape_pattern.items():
            new_r = new_min_r + rel_r
            new_c = new_min_c + rel_c
            if 0 <= new_r < rows and 0 <= new_c < cols:
                output[new_r, new_c] = tile_color
        
        step += 1
    
    return output.tolist()


if __name__ == "__main__":
    import json
    
    # Load task data
    with open('/Users/evanpieser/Downloads/re-arc_test_challenges-2026-03-16T22-48-53.json') as f:
        data = json.load(f)
    
    task = data['27dc200b']
    
    print("Testing on ALL training examples:")
    all_passed = True
    for i, example in enumerate(task['train']):
        input_grid = example['input']
        expected = example['output']
        result = transform(input_grid)
        
        passed = result == expected
        all_passed = all_passed and passed
        print(f"\nTrain {i}: {'✓ PASS' if passed else '✗ FAIL'}")
        
        if not passed:
            import numpy as np
            result_arr = np.array(result)
            expected_arr = np.array(expected)
            diff = result_arr != expected_arr
            diff_count = np.sum(diff)
            print(f"  Differences: {diff_count} cells")
            if diff_count < 20:
                diff_pos = list(zip(*np.where(diff)))
                for r, c in diff_pos[:10]:
                    print(f"    ({r},{c}): got {result_arr[r,c]}, expected {expected_arr[r,c]}")
    
    print(f"\n{'='*50}")
    print(f"Overall: {'ALL PASSED ✓' if all_passed else 'SOME FAILED ✗'}")
