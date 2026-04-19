"""
ARC Puzzle 0625447b Solver

Pattern: The grid contains rectangular regions filled with a "masking" color.
The transformation fills these regions by creating point symmetry (180° rotation)
around the center of the grid.
"""

def transform(grid):
    import numpy as np
    grid = np.array(grid)
    H, W = grid.shape
    output = grid.copy()
    
    # For each cell, check if it's part of a masked region
    # by seeing if the point-symmetric counterpart differs
    # The masked regions are large rectangular areas of uniform color
    
    # Find potential masking colors by looking for colors that form
    # large contiguous rectangular blocks
    from collections import Counter
    
    # Count occurrences of each color
    color_counts = Counter(grid.flatten())
    
    # Create a mask of cells that need to be filled
    # A cell needs filling if it's part of a large uniform region
    # and differs from its point-symmetric counterpart
    
    mask = np.zeros_like(grid, dtype=bool)
    
    for r in range(H):
        for c in range(W):
            # Point symmetric position
            sym_r = H - 1 - r
            sym_c = W - 1 - c
            
            val = grid[r, c]
            sym_val = grid[sym_r, sym_c]
            
            # If values differ, one of them is in a masked region
            if val != sym_val:
                # The one that's part of a larger uniform block is the mask
                # Check local uniformity
                def count_same_neighbors(row, col, value, g):
                    count = 0
                    for dr in range(-2, 3):
                        for dc in range(-2, 3):
                            nr, nc = row + dr, col + dc
                            if 0 <= nr < H and 0 <= nc < W:
                                if g[nr, nc] == value:
                                    count += 1
                    return count
                
                local_val = count_same_neighbors(r, c, val, grid)
                local_sym = count_same_neighbors(sym_r, sym_c, sym_val, grid)
                
                # The one with more uniform neighbors is likely the mask
                if local_val > local_sym:
                    mask[r, c] = True
                elif local_sym > local_val:
                    mask[sym_r, sym_c] = True
                else:
                    # Tie-breaker: use color frequency (masking color often more frequent)
                    if color_counts[val] > color_counts[sym_val]:
                        mask[r, c] = True
                    else:
                        mask[sym_r, sym_c] = True
    
    # Fill masked regions using point symmetry
    for r in range(H):
        for c in range(W):
            if mask[r, c]:
                sym_r = H - 1 - r
                sym_c = W - 1 - c
                output[r, c] = grid[sym_r, sym_c]
    
    return output.tolist()


if __name__ == "__main__":
    import json
    
    # Load task
    with open('/Users/evanpieser/Downloads/re-arc_test_challenges-2026-03-16T22-48-53.json') as f:
        data = json.load(f)
    
    task = data['0625447b']
    
    print("Testing on all training examples:\n")
    all_passed = True
    
    for i, ex in enumerate(task['train']):
        input_grid = ex['input']
        expected = ex['output']
        result = transform(input_grid)
        
        passed = result == expected
        all_passed = all_passed and passed
        
        print(f"Train {i}: {'PASS' if passed else 'FAIL'}")
        
        if not passed:
            import numpy as np
            result_arr = np.array(result)
            expected_arr = np.array(expected)
            diff = result_arr != expected_arr
            print(f"  Differences at {np.sum(diff)} positions")
    
    print(f"\n{'All tests passed!' if all_passed else 'Some tests failed.'}")
