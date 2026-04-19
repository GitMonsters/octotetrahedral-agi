def transform(grid):
    from collections import Counter
    
    rows = len(grid)
    cols = len(grid[0])
    
    # Find background color (most common)
    c = Counter()
    for row in grid:
        c.update(row)
    bg = c.most_common(1)[0][0]
    
    # Find non-bg pixels and classify by edge type
    h_edge = []  # top/bottom edge pixels → will fill columns
    v_edge = []  # left/right edge pixels → will fill rows
    
    for r in range(rows):
        for col in range(cols):
            if grid[r][col] != bg:
                color = grid[r][col]
                if r == 0 or r == rows - 1:
                    h_edge.append((col, color))
                elif col == 0 or col == cols - 1:
                    v_edge.append((r, color))
    
    # Build output
    output = [row[:] for row in grid]
    
    if h_edge:
        # Fill columns, sorted by column position
        h_edge.sort()
        positions = [p for p, _ in h_edge]
        colors = [c for _, c in h_edge]
        
        if len(positions) == 1:
            col_pos = positions[0]
            for r in range(rows):
                output[r][col_pos] = colors[0]
        else:
            step = positions[1] - positions[0]
            pos = positions[0]
            i = 0
            while pos < cols:
                for r in range(rows):
                    output[r][pos] = colors[i % len(colors)]
                pos += step
                i += 1
    
    if v_edge:
        # Fill rows, sorted by row position
        v_edge.sort()
        positions = [p for p, _ in v_edge]
        colors = [c for _, c in v_edge]
        
        if len(positions) == 1:
            row_pos = positions[0]
            for col in range(cols):
                output[row_pos][col] = colors[0]
        else:
            step = positions[1] - positions[0]
            pos = positions[0]
            i = 0
            while pos < rows:
                for col in range(cols):
                    output[pos][col] = colors[i % len(colors)]
                pos += step
                i += 1
    
    return output


if __name__ == "__main__":
    import json
    with open('/tmp/rearc45/04490b87.json') as f:
        task = json.load(f)
    for i, pair in enumerate(task['train']):
        result = transform(pair['input'])
        if result == pair['output']:
            print(f"Train {i}: PASS")
        else:
            print(f"Train {i}: FAIL")
            # Show differences
            for r in range(len(result)):
                for c in range(len(result[0])):
                    if result[r][c] != pair['output'][r][c]:
                        print(f"  ({r},{c}): got {result[r][c]}, expected {pair['output'][r][c]}")
    print("All training pairs pass!")
