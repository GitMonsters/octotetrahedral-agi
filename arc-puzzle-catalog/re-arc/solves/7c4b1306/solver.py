def transform(grid):
    """
    ARC puzzle 7c4b1306:
    1. Scale up each cell by factor N (where N is determined by input structure)
    2. Draw diagonal lines of 1s where different non-9 colored corners meet
    """
    import numpy as np
    grid = np.array(grid)
    h, w = grid.shape
    
    # Find scale factor: count contiguous 9s in the interior
    # The scale is based on the rectangular 9-region dimensions
    # Find the 9-filled rectangle bounds
    nines = np.argwhere(grid == 9)
    if len(nines) > 0:
        min_r, min_c = nines.min(axis=0)
        max_r, max_c = nines.max(axis=0)
        # Scale factor is roughly the 9-region extent
        nine_h = max_r - min_r + 1
        nine_w = max_c - min_c + 1
        scale = max(nine_h, nine_w)
    else:
        scale = 3
    
    # Scale up the grid
    out_h, out_w = h * scale, w * scale
    output = np.zeros((out_h, out_w), dtype=int)
    
    for r in range(h):
        for c in range(w):
            output[r*scale:(r+1)*scale, c*scale:(c+1)*scale] = grid[r, c]
    
    # Find corners where different non-9 colors meet and draw diagonals
    for r in range(h - 1):
        for c in range(w - 1):
            # Get 2x2 neighborhood
            tl = grid[r, c]
            tr = grid[r, c + 1]
            bl = grid[r + 1, c]
            br = grid[r + 1, c + 1]
            
            # Check diagonal pairs for non-9 matching colors
            # If TL == BR and they're non-9, and TR/BL are different (and non-9 or 9)
            if tl != 9 and br != 9 and tl == br and tl != tr and tl != bl:
                # Draw diagonal from corner going outward both ways
                # TR-BL diagonal with 1s
                for d in range(scale):
                    # Upper-right to lower-left diagonal
                    dr, dc = (r + 1) * scale - 1 - d, (c + 1) * scale + d
                    if 0 <= dr < out_h and 0 <= dc < out_w:
                        output[dr, dc] = 1
                    dr, dc = (r + 1) * scale + d, (c + 1) * scale - 1 - d
                    if 0 <= dr < out_h and 0 <= dc < out_w:
                        output[dr, dc] = 1
                        
            if tr != 9 and bl != 9 and tr == bl and tr != tl and tr != br:
                # Draw TL-BR diagonal with 1s
                for d in range(scale):
                    dr, dc = (r + 1) * scale - 1 - d, (c + 1) * scale - 1 - d
                    if 0 <= dr < out_h and 0 <= dc < out_w:
                        output[dr, dc] = 1
                    dr, dc = (r + 1) * scale + d, (c + 1) * scale + d
                    if 0 <= dr < out_h and 0 <= dc < out_w:
                        output[dr, dc] = 1
    
    return output.tolist()

# Test
if __name__ == "__main__":
    import json
    data = json.load(open('/Users/evanpieser/Downloads/re-arc_test_challenges-2026-03-16T22-48-53.json'))
    puzzle = data['7c4b1306']
    
    all_pass = True
    for i, ex in enumerate(puzzle['train']):
        result = transform(ex['input'])
        expected = ex['output']
        if result == expected:
            print(f"Train {i+1}: PASS")
        else:
            print(f"Train {i+1}: FAIL")
            print(f"  Expected shape: {len(expected)}x{len(expected[0])}")
            print(f"  Got shape: {len(result)}x{len(result[0])}")
            all_pass = False
    
    print(f"\nAll pass: {all_pass}")
