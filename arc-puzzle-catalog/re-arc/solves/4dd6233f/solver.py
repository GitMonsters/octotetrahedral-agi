def transform(input_grid):
    from collections import Counter
    
    rows = len(input_grid)
    cols = len(input_grid[0])
    
    # Find background (most common value)
    all_vals = [v for row in input_grid for v in row]
    bg = Counter(all_vals).most_common(1)[0][0]
    
    # Extract bars: rows with non-bg prefix
    bars = []  # (row, color, length)
    for r in range(rows):
        length = 0
        color = None
        for c in range(cols):
            if input_grid[r][c] != bg:
                if color is None:
                    color = input_grid[r][c]
                length += 1
            else:
                break
        if length > 0:
            bars.append((r, color, length))
    
    # Get distinct lengths, sorted descending
    distinct_lengths = sorted(set(l for _, _, l in bars), reverse=True)
    
    # Assign colors: longest -> 6, 2nd longest -> 0, rest -> 1
    length_to_color = {}
    for i, l in enumerate(distinct_lengths):
        if i == 0:
            length_to_color[l] = 6
        elif i == 1:
            length_to_color[l] = 0
        else:
            length_to_color[l] = 1
    
    # Build output
    output = [row[:] for row in input_grid]
    for r, old_color, length in bars:
        new_color = length_to_color[length]
        for c in range(length):
            output[r][c] = new_color
    
    return output
