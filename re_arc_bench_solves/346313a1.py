"""
Solver for ARC puzzle 346313a1

Pattern: The grid has a tiled structure with markers at regular intervals.
The markers form a 5x5 tile pattern (positions 0,1,2,3,4 repeating).
Markers appear at position 1 and 6, 11, 16, 21... (i.e., 1 + 5*k).
The output is 4x4, capturing which unique color appears at each logical cell
formed by these marker positions across the grid.
"""

from collections import Counter

def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    
    # Find the background color (most common)
    flat = [c for row in grid for c in row]
    bg = Counter(flat).most_common(1)[0][0]
    
    # Find all non-background marker positions
    markers = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg:
                markers.append((r, c, grid[r][c]))
    
    if not markers:
        return [[bg] * 4 for _ in range(4)]
    
    # Get unique row and column positions of markers
    marker_rows = sorted(set(r for r, c, v in markers))
    marker_cols = sorted(set(c for r, c, v in markers))
    
    # Find the period (distance between consecutive marker positions)
    def find_period(positions):
        if len(positions) < 2:
            return positions[0] if positions else 5
        diffs = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
        # Period is the GCD of differences or the minimum difference
        min_diff = min(diffs) if diffs else 5
        return min_diff
    
    row_period = find_period(marker_rows)
    col_period = find_period(marker_cols)
    
    # Map each marker to its position in the 4x4 logical grid
    # We use modulo of the marker index
    output = [[bg] * 4 for _ in range(4)]
    
    # Create mapping from marker positions to logical grid positions
    row_to_idx = {r: i % 4 for i, r in enumerate(marker_rows)}
    col_to_idx = {c: j % 4 for j, c in enumerate(marker_cols)}
    
    # Fill in the output grid with marker colors
    for r, c, v in markers:
        ri = row_to_idx.get(r, -1)
        ci = col_to_idx.get(c, -1)
        if 0 <= ri < 4 and 0 <= ci < 4:
            output[ri][ci] = v
    
    return output


if __name__ == "__main__":
    import json
    
    with open('/Users/evanpieser/Downloads/re-arc_test_challenges-2026-03-16T22-48-53.json') as f:
        data = json.load(f)
    
    task = data['346313a1']
    
    print("Testing on all training examples:")
    all_passed = True
    for i, ex in enumerate(task['train']):
        inp = ex['input']
        expected = ex['output']
        result = transform(inp)
        
        passed = result == expected
        all_passed = all_passed and passed
        
        print(f"\n--- Example {i+1}: {'PASS' if passed else 'FAIL'} ---")
        if not passed:
            print("Expected:")
            for row in expected:
                print(row)
            print("Got:")
            for row in result:
                print(row)
    
    print(f"\n{'='*40}")
    print(f"Overall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
