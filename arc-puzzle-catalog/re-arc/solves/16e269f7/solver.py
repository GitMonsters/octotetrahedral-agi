"""
Solver for ARC puzzle 16e269f7

Pattern: Scattered colored pixels form 3 anti-diagonal rows.
Each anti-diagonal row (constant r-c value) becomes one row of the 3x3 output.
Within each row, pixels are sorted by position along the diagonal (r+c).
"""

def transform(grid):
    rows, cols = len(grid), len(grid[0])
    
    # Find background color (most common)
    from collections import Counter
    flat = [c for row in grid for c in row]
    bg = Counter(flat).most_common(1)[0][0]
    
    # Find all non-background pixels with their positions
    pixels = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg:
                pixels.append((r, c, grid[r][c]))
    
    if not pixels:
        return [[bg]*3 for _ in range(3)]
    
    # Group pixels by their anti-diagonal index (r - c)
    # This determines which row they belong to in the output
    diag_groups = {}
    for r, c, val in pixels:
        key = r - c
        if key not in diag_groups:
            diag_groups[key] = []
        diag_groups[key].append((r, c, val))
    
    # Sort groups by their diagonal index to get row order
    sorted_keys = sorted(diag_groups.keys())
    
    # We expect 3 groups for a 3x3 output
    # Merge into 3 groups if there are more
    if len(sorted_keys) > 3:
        # Group into 3 clusters based on diagonal index ranges
        min_key, max_key = min(sorted_keys), max(sorted_keys)
        range_size = (max_key - min_key) / 3
        
        clustered = {0: [], 1: [], 2: []}
        for key in sorted_keys:
            cluster_idx = min(2, int((key - min_key) / range_size)) if range_size > 0 else 0
            clustered[cluster_idx].extend(diag_groups[key])
        
        groups = [clustered[i] for i in range(3)]
    else:
        groups = [diag_groups[k] for k in sorted_keys]
        # Pad to 3 groups if needed
        while len(groups) < 3:
            groups.append([])
    
    # Build output: each group becomes a row
    # Within each group, sort by (r + c) to get column order
    output = []
    for group in groups:
        if not group:
            output.append([bg, bg, bg])
            continue
            
        # Sort by r+c (position along anti-diagonal)
        sorted_pixels = sorted(group, key=lambda x: x[0] + x[1])
        
        if len(sorted_pixels) >= 3:
            # Cluster into 3 columns
            min_pos = sorted_pixels[0][0] + sorted_pixels[0][1]
            max_pos = sorted_pixels[-1][0] + sorted_pixels[-1][1]
            range_size = (max_pos - min_pos) / 3 if max_pos > min_pos else 1
            
            row = [bg, bg, bg]
            for r, c, val in sorted_pixels:
                pos = r + c
                col_idx = min(2, int((pos - min_pos) / range_size)) if range_size > 0 else 0
                row[col_idx] = val
            output.append(row)
        else:
            # Spread pixels across 3 columns
            row = [bg, bg, bg]
            for i, (r, c, val) in enumerate(sorted_pixels):
                row[i] = val
            output.append(row)
    
    return output


if __name__ == "__main__":
    import json
    
    with open('/Users/evanpieser/Downloads/re-arc_test_challenges-2026-03-16T22-48-53.json') as f:
        data = json.load(f)
    
    task = data['16e269f7']
    
    print("Testing on training examples:")
    all_pass = True
    for i, ex in enumerate(task['train']):
        result = transform(ex['input'])
        expected = ex['output']
        match = result == expected
        all_pass = all_pass and match
        print(f"\nTrain {i}: {'PASS' if match else 'FAIL'}")
        print(f"Expected: {expected}")
        print(f"Got:      {result}")
    
    print(f"\n{'='*50}")
    print(f"Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
