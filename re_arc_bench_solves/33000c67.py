def transform(grid):
    """
    ARC puzzle 33000c67:
    - Grid divided into regions by separator lines
    - Find the region surrounded by the most clean (background-only) neighbors
    - Keep only that region's content, clear all others to background
    """
    import numpy as np
    grid = np.array(grid)
    h, w = grid.shape
    
    # Find separator color (forms complete rows/cols)
    sep_color = None
    for r in range(h):
        if len(set(grid[r])) == 1:
            sep_color = grid[r, 0]
            break
    if sep_color is None:
        for c in range(w):
            if len(set(grid[:, c])) == 1:
                sep_color = grid[0, c]
                break
    
    # Find separator rows and columns
    sep_rows = [r for r in range(h) if all(grid[r, c] == sep_color for c in range(w))]
    sep_cols = [c for c in range(w) if all(grid[r, c] == sep_color for r in range(h))]
    
    # Create region boundaries
    row_bounds = [-1] + sep_rows + [h]
    col_bounds = [-1] + sep_cols + [w]
    
    # Find background color (most common non-separator)
    flat = grid.flatten()
    counts = {}
    for v in flat:
        if v != sep_color:
            counts[v] = counts.get(v, 0) + 1
    bg_color = max(counts, key=counts.get) if counts else 0
    
    # Extract regions
    regions = []
    for i in range(len(row_bounds) - 1):
        for j in range(len(col_bounds) - 1):
            r1, r2 = row_bounds[i] + 1, row_bounds[i + 1]
            c1, c2 = col_bounds[j] + 1, col_bounds[j + 1]
            if r1 < r2 and c1 < c2:
                regions.append((i, j, r1, r2, c1, c2))
    
    # Check if region is clean (only background)
    def is_clean(r1, r2, c1, c2):
        for r in range(r1, r2):
            for c in range(c1, c2):
                if grid[r, c] != bg_color:
                    return False
        return True
    
    # Build region grid for neighbor lookup
    n_rows = len(row_bounds) - 1
    n_cols = len(col_bounds) - 1
    region_map = {}
    for reg in regions:
        i, j = reg[0], reg[1]
        region_map[(i, j)] = reg
    
    clean_status = {}
    for reg in regions:
        i, j, r1, r2, c1, c2 = reg
        clean_status[(i, j)] = is_clean(r1, r2, c1, c2)
    
    # Count clean neighbors for each non-clean region
    def count_clean_neighbors(i, j):
        count = 0
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if (ni, nj) in clean_status and clean_status[(ni, nj)]:
                count += 1
        return count
    
    # Find region with most clean neighbors (among non-clean regions)
    best_region = None
    best_score = -1
    for reg in regions:
        i, j, r1, r2, c1, c2 = reg
        if not clean_status[(i, j)]:  # Has content
            score = count_clean_neighbors(i, j)
            if score > best_score:
                best_score = score
                best_region = reg
    
    # If tie or no clear winner, use alternate criteria:
    # Find region with unique/rare color as tiebreaker
    if best_region is None or best_score == 0:
        # Find rarest non-bg, non-sep color
        color_counts = {}
        for reg in regions:
            i, j, r1, r2, c1, c2 = reg
            for r in range(r1, r2):
                for c in range(c1, c2):
                    v = grid[r, c]
                    if v != bg_color and v != sep_color:
                        color_counts[v] = color_counts.get(v, 0) + 1
        
        if color_counts:
            rarest_color = min(color_counts, key=color_counts.get)
            for reg in regions:
                i, j, r1, r2, c1, c2 = reg
                for r in range(r1, r2):
                    for c in range(c1, c2):
                        if grid[r, c] == rarest_color:
                            best_region = reg
                            break
                    if best_region == reg:
                        break
    
    # Build output: preserve only best_region, clear others
    output = np.full_like(grid, bg_color)
    
    # Copy separator lines
    for r in sep_rows:
        output[r, :] = sep_color
    for c in sep_cols:
        output[:, c] = sep_color
    
    # Copy content from best region
    if best_region:
        _, _, r1, r2, c1, c2 = best_region
        output[r1:r2, c1:c2] = grid[r1:r2, c1:c2]
    
    return output.tolist()


if __name__ == "__main__":
    import json
    with open('/Users/evanpieser/Downloads/re-arc_test_challenges-2026-03-16T22-48-53.json') as f:
        data = json.load(f)
    task = data['33000c67']
    
    print("Testing on training examples:")
    all_pass = True
    for i, ex in enumerate(task['train']):
        result = transform(ex['input'])
        expected = ex['output']
        match = result == expected
        all_pass = all_pass and match
        print(f"  Train {i}: {'✓ PASS' if match else '✗ FAIL'}")
        if not match:
            import numpy as np
            r = np.array(result)
            e = np.array(expected)
            diff = np.where(r != e)
            print(f"    Differences at {len(diff[0])} positions")
    
    print(f"\nOverall: {'ALL PASSED' if all_pass else 'SOME FAILED'}")
