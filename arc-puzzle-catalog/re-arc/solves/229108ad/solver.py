def transform(grid):
    """
    ARC puzzle 229108ad:
    1. Find the clean rectangular region (only background color 1, or background + sparse markers)
    2. Extract it and project markers: if marker at (r,c), entire row r and col c become marker color
    """
    import numpy as np
    grid = np.array(grid)
    h, w = grid.shape
    
    # Find the largest rectangular region containing ONLY value 1 (pure background)
    # OR value 1 + one marker color
    best_rect = None
    best_area = 0
    best_marker = None
    
    for r1 in range(h):
        for r2 in range(r1 + 2, h):  # min height 3
            for c1 in range(w):
                for c2 in range(c1 + 2, w):  # min width 3
                    region = grid[r1:r2+1, c1:c2+1]
                    unique, counts = np.unique(region, return_counts=True)
                    area = (r2 - r1 + 1) * (c2 - c1 + 1)
                    
                    # Case 1: Pure region of 1s only
                    if len(unique) == 1 and unique[0] == 1:
                        if area > best_area:
                            best_area = area
                            best_rect = (r1, r2, c1, c2)
                            best_marker = None
                    
                    # Case 2: Region with 1 (background) + one marker color
                    elif len(unique) == 2 and 1 in unique:
                        marker = unique[unique != 1][0]
                        bg_count = counts[unique == 1][0]
                        marker_count = counts[unique == marker][0]
                        
                        # Markers should be sparse (< 30% of region)
                        if marker_count / area < 0.3:
                            if area > best_area:
                                best_area = area
                                best_rect = (r1, r2, c1, c2)
                                best_marker = marker
    
    if best_rect is None:
        return grid.tolist()
    
    r1, r2, c1, c2 = best_rect
    region = grid[r1:r2+1, c1:c2+1].copy()
    reg_h, reg_w = region.shape
    
    # If no markers, just return background
    if best_marker is None:
        return [[1] * reg_w for _ in range(reg_h)]
    
    # Project markers: entire row and column become marker color
    output = np.ones((reg_h, reg_w), dtype=int)
    
    marker_rows = set()
    marker_cols = set()
    for r in range(reg_h):
        for c in range(reg_w):
            if region[r, c] == best_marker:
                marker_rows.add(r)
                marker_cols.add(c)
    
    for r in marker_rows:
        output[r, :] = best_marker
    for c in marker_cols:
        output[:, c] = best_marker
    
    return output.tolist()


if __name__ == "__main__":
    import json
    
    with open('/Users/evanpieser/Downloads/re-arc_test_challenges-2026-03-16T22-48-53.json') as f:
        data = json.load(f)
    
    task = data['229108ad']
    
    for i, ex in enumerate(task['train']):
        inp = ex['input']
        expected = ex['output']
        result = transform(inp)
        
        match = result == expected
        print(f"Train {i}: {'PASS' if match else 'FAIL'}")
        if not match:
            print(f"  Expected shape: {len(expected)}x{len(expected[0])}")
            print(f"  Got shape: {len(result)}x{len(result[0])}")
            print(f"  Expected: {expected[:3]}...")
            print(f"  Got: {result[:3]}...")
