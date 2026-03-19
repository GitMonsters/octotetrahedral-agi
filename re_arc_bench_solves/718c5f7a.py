def transform(grid):
    """
    ARC Puzzle 718c5f7a: Find marker pair, draw L-shaped crosshair.
    - Horizontal markers: extend right, then vertical line up
    - Vertical markers: extend up, then horizontal line right
    """
    import copy
    grid = [list(row) for row in grid]
    rows, cols = len(grid), len(grid[0])
    output = copy.deepcopy(grid)
    
    # Find marker: pair of adjacent same-colored cells that are rare
    # Count color frequencies to identify marker vs background/noise
    color_counts = {}
    for r in range(rows):
        for c in range(cols):
            color_counts[grid[r][c]] = color_counts.get(grid[r][c], 0) + 1
    
    # Find adjacent pairs of same color
    marker_positions = []
    marker_color = None
    orientation = None  # 'h' for horizontal, 'v' for vertical
    
    for r in range(rows):
        for c in range(cols):
            color = grid[r][c]
            # Check horizontal adjacency
            if c + 1 < cols and grid[r][c + 1] == color:
                # Check if this could be a marker (relatively rare color)
                if color_counts[color] <= 4:  # Marker should be rare
                    marker_positions = [(r, c), (r, c + 1)]
                    marker_color = color
                    orientation = 'h'
                    break
            # Check vertical adjacency
            if r + 1 < rows and grid[r + 1][c] == color:
                if color_counts[color] <= 4:
                    marker_positions = [(r, c), (r + 1, c)]
                    marker_color = color
                    orientation = 'v'
                    break
        if marker_positions:
            break
    
    if not marker_positions:
        return output
    
    # Determine the noise color (second most common after background)
    sorted_colors = sorted(color_counts.items(), key=lambda x: -x[1])
    background = sorted_colors[0][0]
    noise_color = sorted_colors[1][0] if len(sorted_colors) > 1 else background
    
    # The line color should be the noise color (what we draw with)
    line_color = noise_color if noise_color != marker_color else marker_color
    
    if orientation == 'h':
        # Horizontal markers: extend right, then draw vertical line up
        r, c = marker_positions[0]  # leftmost marker
        c_right = marker_positions[1][1]  # rightmost marker col
        
        # Find how far right to extend (to edge or based on grid)
        # The vertical line goes up from the endpoint
        # Extension seems to go to position that makes sense geometrically
        
        # Calculate extension: extend right from marker until we reach a good column
        # Based on examples, extend to roughly mirror position or edge
        ext_col = min(c_right + (c_right - c + 1) * 2, cols - 1)
        
        # Actually, looking at train 1: markers at col 3,4, line goes to col 8
        # That's col 4 + 4 = col 8. Try doubling the distance.
        ext_col = min(c_right + (c_right - c + 1) + (rows - r - 1) // 2, cols - 1)
        
        # Simpler: the vertical line column seems to be positioned based on row
        # Train 1: marker at row 10, vertical line at col 8
        ext_col = c_right + (rows - r - 1) // 2
        if ext_col >= cols:
            ext_col = cols - 1
        
        # Draw horizontal line from marker to ext_col
        for cc in range(c_right + 1, ext_col + 1):
            output[r][cc] = marker_color
        
        # Draw vertical line going up from ext_col
        for rr in range(0, r):
            output[rr][ext_col] = marker_color
            
    elif orientation == 'v':
        # Vertical markers: extend up, then draw horizontal line right
        r_top = marker_positions[0][0]  # top marker row
        r_bot = marker_positions[1][0]  # bottom marker row  
        c = marker_positions[0][1]
        
        # Calculate extension upward
        # From train 3: markers at rows 13-14, vertical line up to row 4
        ext_row = max(r_top - (r_bot - r_top + 1) - (cols - c - 1) // 2, 0)
        
        # Simpler approach: extend up based on column position
        ext_row = max(r_top - (cols - c) // 2, 0)
        
        # Draw vertical line from marker going up
        for rr in range(ext_row, r_top):
            output[rr][c] = line_color
        
        # Draw horizontal line going right from ext_row
        for cc in range(c + 1, cols):
            output[ext_row][cc] = line_color
    
    return output


if __name__ == "__main__":
    import json
    
    with open('/Users/evanpieser/Downloads/re-arc_test_challenges-2026-03-16T22-48-53.json') as f:
        data = json.load(f)
    
    task = data['718c5f7a']
    
    print("Testing on training examples:")
    for i, ex in enumerate(task['train']):
        inp = ex['input']
        expected = ex['output']
        result = transform(inp)
        
        match = result == expected
        print(f"\nTrain {i}: {'PASS' if match else 'FAIL'}")
        
        if not match:
            print("Expected:")
            for row in expected[:5]:
                print(row[:15], "...")
            print("Got:")
            for row in result[:5]:
                print(row[:15], "...")
