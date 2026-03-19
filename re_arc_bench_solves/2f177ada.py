"""
ARC Puzzle 2f177ada Solver

Pattern: Create nested rectangular frames at even depths (0, 2, 4, ...) filled with 6.
- Depth 0: outermost border (row 0, last row, col 0, last col)
- Depth 2: next frame inward
- etc.
Cells at odd depths keep their original values.
"""

def transform(grid: list[list[int]]) -> list[list[int]]:
    H = len(grid)
    W = len(grid[0])
    
    # Create output grid as copy of input
    output = [row[:] for row in grid]
    
    # For each cell, determine its depth and whether it's on a frame
    for r in range(H):
        for c in range(W):
            # Distance from each edge
            dist_top = r
            dist_bottom = H - 1 - r
            dist_left = c
            dist_right = W - 1 - c
            
            # The depth is the minimum distance to any edge
            depth = min(dist_top, dist_bottom, dist_left, dist_right)
            
            # Check if this cell is on the frame at an even depth
            # A cell is on a frame at depth d if its min distance is exactly d
            # and it lies on the perimeter of the rectangle at that depth
            
            # We fill with 6 if the cell is on any even-depth frame
            # Cell is on frame at depth d if: depth == d AND 
            # (r == d OR r == H-1-d OR c == d OR c == W-1-c)
            
            # Simpler: cell at (r,c) is on an even frame if depth is even
            # OR if it's on the perimeter line of an even depth
            
            # Actually: for nested frames at 0, 2, 4, ...
            # Check each even frame depth and see if this cell is on that frame's perimeter
            for d in range(0, min(H, W) // 2 + 1, 2):  # even depths: 0, 2, 4, ...
                # Check if cell is on the frame at depth d
                # Frame at depth d spans: rows [d, H-1-d], cols [d, W-1-d]
                if d > min(H-1-d, W-1-d):  # frame doesn't exist
                    break
                    
                # Check if on this frame's perimeter
                on_top = (r == d) and (d <= c <= W - 1 - d)
                on_bottom = (r == H - 1 - d) and (d <= c <= W - 1 - d)
                on_left = (c == d) and (d <= r <= H - 1 - d)
                on_right = (c == W - 1 - d) and (d <= r <= H - 1 - d)
                
                if on_top or on_bottom or on_left or on_right:
                    output[r][c] = 6
                    break
    
    return output


if __name__ == "__main__":
    import json
    
    # Load the task
    with open('/Users/evanpieser/Downloads/re-arc_test_challenges-2026-03-16T22-48-53.json') as f:
        data = json.load(f)
    
    task = data['2f177ada']
    
    # Test on all training examples
    print("Testing on all training examples:\n")
    all_passed = True
    
    for i, ex in enumerate(task['train']):
        input_grid = ex['input']
        expected_output = ex['output']
        actual_output = transform(input_grid)
        
        passed = actual_output == expected_output
        all_passed = all_passed and passed
        
        print(f"Example {i}: {'PASS' if passed else 'FAIL'}")
        
        if not passed:
            print(f"  Input size: {len(input_grid)}x{len(input_grid[0])}")
            print(f"  Expected size: {len(expected_output)}x{len(expected_output[0])}")
            print(f"  Actual size: {len(actual_output)}x{len(actual_output[0])}")
            # Find first difference
            for r in range(len(expected_output)):
                for c in range(len(expected_output[0])):
                    if actual_output[r][c] != expected_output[r][c]:
                        print(f"  First diff at ({r},{c}): expected {expected_output[r][c]}, got {actual_output[r][c]}")
                        break
                else:
                    continue
                break
    
    print(f"\n{'All tests PASSED!' if all_passed else 'Some tests FAILED!'}")
