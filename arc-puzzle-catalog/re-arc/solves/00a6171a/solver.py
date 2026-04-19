from collections import Counter


def transform(grid: list[list[int]]) -> list[list[int]]:
    """Complete the foreground pattern to have 180° rotational symmetry.
    
    New positions needed for symmetry are marked with color 9.
    """
    rows = len(grid)
    cols = len(grid[0])
    
    # Determine background (most common color)
    all_vals = [v for row in grid for v in row]
    bg = Counter(all_vals).most_common(1)[0][0]
    
    # Find foreground positions
    fg_set = set()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg:
                fg_set.add((r, c))
    
    if not fg_set:
        return [row[:] for row in grid]
    
    # Find optimal center of 180° rotation by maximizing symmetric pairs
    # Search over all half-integer centers (encoded as 2x integers)
    best_center = None
    best_pairs = -1
    
    # Narrow search range to area around the foreground
    fg_rows = [r for r, c in fg_set]
    fg_cols = [c for r, c in fg_set]
    min_fr, max_fr = min(fg_rows), max(fg_rows)
    min_fc, max_fc = min(fg_cols), max(fg_cols)
    
    # The center must be such that all rotations stay in-bounds.
    # Search within a reasonable range around the foreground.
    for cr2 in range(2 * min_fr, 2 * max_fr + 1):
        for cc2 in range(2 * min_fc, 2 * max_fc + 1):
            pairs = 0
            all_in_bounds = True
            
            for r, c in fg_set:
                rot_r = cr2 - r
                rot_c = cc2 - c
                
                if rot_r < 0 or rot_r >= rows or rot_c < 0 or rot_c >= cols:
                    all_in_bounds = False
                    break
                
                if (rot_r, rot_c) in fg_set:
                    pairs += 1
            
            if all_in_bounds and pairs > best_pairs:
                best_pairs = pairs
                best_center = (cr2, cc2)
    
    if best_center is None:
        return [row[:] for row in grid]
    
    cr2, cc2 = best_center
    
    # Build output: copy input, then add 9s for asymmetric positions
    out = [row[:] for row in grid]
    for r, c in fg_set:
        rot_r = cr2 - r
        rot_c = cc2 - c
        if 0 <= rot_r < rows and 0 <= rot_c < cols:
            if (rot_r, rot_c) not in fg_set:
                out[rot_r][rot_c] = 9
    
    return out
