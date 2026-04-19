def transform(grid):
    rows, cols = len(grid), len(grid[0])
    
    layers = []
    top, left, bottom, right = 0, 0, rows - 1, cols - 1
    
    while top <= bottom and left <= right:
        color = grid[top][left]
        t = 0
        while top + t <= bottom - t and left + t <= right - t:
            all_same = True
            for c in range(left + t, right - t + 1):
                if grid[top + t][c] != color or grid[bottom - t][c] != color:
                    all_same = False
                    break
            if not all_same:
                break
            for r in range(top + t, bottom - t + 1):
                if grid[r][left + t] != color or grid[r][right - t] != color:
                    all_same = False
                    break
            if not all_same:
                break
            t += 1
        
        if t == 0:
            layers.append((1, color))
            break
        layers.append((t, color))
        top += t; left += t; bottom -= t; right -= t
    
    if not layers:
        return [row[:] for row in grid]
    
    n = len(layers)
    colors = [c for _, c in layers]
    thicknesses = [t for t, _ in layers]
    
    # Check for 3-layer palindrome special case
    if n == 3 and colors[0] == colors[2] and thicknesses[0] <= 2 * (thicknesses[1] + thicknesses[2]):
        # Reversal: frame goes to outside and inside, outer fills middle
        new_layers = [(thicknesses[1], colors[1]), 
                      (thicknesses[0], colors[0]), 
                      (thicknesses[2], colors[1])]
    else:
        # Standard rotation: each group gets the next-inner color
        new_colors = [colors[-1]] + colors[:-1]
        new_layers = [(thicknesses[i], new_colors[i]) for i in range(n)]
        
        # Merge consecutive same-color layers
        merged = [list(new_layers[0])]
        for i in range(1, len(new_layers)):
            if new_layers[i][1] == merged[-1][1]:
                merged[-1][0] += new_layers[i][0]
            else:
                merged.append(list(new_layers[i]))
        new_layers = [tuple(m) for m in merged]
    
    layers = new_layers
    
    # Rebuild grid
    result = [[0] * cols for _ in range(rows)]
    top, left, bottom, right = 0, 0, rows - 1, cols - 1
    
    for i, (t, color) in enumerate(layers):
        if i == len(layers) - 1:
            for r in range(top, bottom + 1):
                for c in range(left, right + 1):
                    result[r][c] = color
            break
        for d in range(t):
            for c in range(left + d, right - d + 1):
                result[top + d][c] = color
                result[bottom - d][c] = color
            for r in range(top + d, bottom - d + 1):
                result[r][left + d] = color
                result[r][right - d] = color
        top += t; left += t; bottom -= t; right -= t
    
    return result
