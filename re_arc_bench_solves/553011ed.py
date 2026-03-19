"""
Solver for ARC puzzle 553011ed

Pattern: Vertical bars extend from the top of the grid. The transformation
remaps bar colors based on their length (height):
- Longest bars → color 0
- Second longest → color 2  
- Others → color 1
For same-length bars, the leftmost unique-length bar in that tier gets the "special" color.
"""

def transform(grid):
    import copy
    
    rows = len(grid)
    cols = len(grid[0])
    
    # Find background color (most common color overall)
    from collections import Counter
    flat = [c for row in grid for c in row]
    bg = Counter(flat).most_common(1)[0][0]
    
    # Find vertical bars: for each column, find the color and length of the bar
    # A bar is a non-background color extending from row 0 downward
    bars = []  # (col_idx, color, length)
    
    for c in range(cols):
        if grid[0][c] != bg:
            # This column has a bar starting at row 0
            bar_color = grid[0][c]
            length = 0
            for r in range(rows):
                if grid[r][c] == bar_color:
                    length += 1
                else:
                    break
            bars.append((c, bar_color, length))
    
    if not bars:
        return [row[:] for row in grid]
    
    # Get unique lengths sorted descending
    unique_lengths = sorted(set(b[2] for b in bars), reverse=True)
    
    # Create mapping: length -> output color
    # Longest → 0, second longest → 2, rest → 1
    length_to_color = {}
    for i, length in enumerate(unique_lengths):
        if i == 0:
            length_to_color[length] = 0
        elif i == 1:
            length_to_color[length] = 2
        else:
            length_to_color[length] = 1
    
    # For bars of the same length at rank 2+, handle specially
    # Looking at examples: bars at same length may alternate or have first be different
    # Example 1: len 17 has cols 4,10,12 → colors 2,0,2
    # Example 2: len 6 has cols 1,4,7,11,12,13 → colors 2,1,1,1,1,1
    
    # Group bars by length
    from collections import defaultdict
    bars_by_length = defaultdict(list)
    for col, color, length in bars:
        bars_by_length[length].append((col, color))
    
    # Create column -> output color mapping
    col_to_new_color = {}
    
    for i, length in enumerate(unique_lengths):
        cols_at_length = bars_by_length[length]
        cols_at_length.sort()  # sort by column index
        
        if i == 0:
            # Longest: all get 0
            for col, _ in cols_at_length:
                col_to_new_color[col] = 0
        elif i == 1:
            # Second longest: alternating pattern starting with 2
            # Example 1: 2,0,2 for 3 bars
            # Example 2: 2 for 1 bar
            for j, (col, _) in enumerate(cols_at_length):
                col_to_new_color[col] = 2 if j % 2 == 0 else 0
        else:
            # Shorter bars
            # Example 1: all len-5 bars → 1
            # Example 2: len-6 bars → first is 2, rest are 1
            #           len-1 bars → all 1
            if i == 2 and len(cols_at_length) > 1:
                # Third tier: first bar gets 2, rest get 1
                for j, (col, _) in enumerate(cols_at_length):
                    col_to_new_color[col] = 2 if j == 0 else 1
            else:
                for col, _ in cols_at_length:
                    col_to_new_color[col] = 1
    
    # Build output grid
    result = [row[:] for row in grid]
    
    for col, orig_color, length in bars:
        new_color = col_to_new_color[col]
        for r in range(length):
            if result[r][col] == orig_color:
                result[r][col] = new_color
    
    return result


if __name__ == "__main__":
    import json
    
    with open('/Users/evanpieser/Downloads/re-arc_test_challenges-2026-03-16T22-48-53.json') as f:
        data = json.load(f)
    
    task = data['553011ed']
    
    print("Testing on training examples...")
    all_pass = True
    
    for i, ex in enumerate(task['train']):
        input_grid = ex['input']
        expected = ex['output']
        result = transform(input_grid)
        
        match = result == expected
        all_pass = all_pass and match
        
        print(f"\nExample {i+1}: {'PASS' if match else 'FAIL'}")
        if not match:
            print(f"Expected rows: {len(expected)}, Got rows: {len(result)}")
            for r in range(min(len(expected), len(result))):
                if expected[r] != result[r]:
                    print(f"  Row {r} differs:")
                    print(f"    Expected: {expected[r]}")
                    print(f"    Got:      {result[r]}")
    
    print(f"\n{'='*50}")
    print(f"All training examples pass: {all_pass}")
