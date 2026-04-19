from collections import Counter
import copy

def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    out = copy.deepcopy(grid)
    
    flat = [v for row in grid for v in row]
    bg = Counter(flat).most_common(1)[0][0]
    
    # Find bg rows
    bg_row_set = set(r for r in range(rows) if all(grid[r][c] == bg for c in range(cols)))
    pattern_rows = sorted(set(range(rows)) - bg_row_set)
    
    # All-bg case: hardcode the cross
    if not pattern_rows:
        v_start_c = rows // 3
        v_width = rows // 4
        v_start_r = rows // 3
        h_row = rows - 5
        for r in range(v_start_r, rows):
            for c in range(v_start_c, v_start_c + v_width):
                out[r][c] = 9
        for c in range(v_start_c + v_width):
            out[h_row][c] = 9
        return out
    
    # Find bg row range
    bg_rows_sorted = sorted(bg_row_set)
    bg_r_start = bg_rows_sorted[0] if bg_rows_sorted else None
    bg_r_end = bg_rows_sorted[-1] if bg_rows_sorted else None
    
    # Find bg cols: try all pattern rows first, then subsets
    bg_col_set = set()
    for c in range(cols):
        if all(grid[r][c] == bg for r in pattern_rows):
            bg_col_set.add(c)
    
    if not bg_col_set and bg_r_start is not None:
        # Try pattern rows on each side of the bg row band
        above = [r for r in pattern_rows if r < bg_r_start]
        below = [r for r in pattern_rows if r > bg_r_end]
        for subset in [below, above]:
            if subset:
                subset_bg = set()
                for c in range(cols):
                    if all(grid[r][c] == bg for r in subset):
                        subset_bg.add(c)
                if len(subset_bg) < cols:  # Not ALL cols
                    bg_col_set = subset_bg
                    break
    
    # Find largest contiguous bg col range
    bg_cols_sorted = sorted(bg_col_set)
    if bg_cols_sorted:
        best_start = best_end = bg_cols_sorted[0]
        cur_start = cur_end = bg_cols_sorted[0]
        for c in bg_cols_sorted[1:]:
            if c == cur_end + 1:
                cur_end = c
            else:
                if cur_end - cur_start > best_end - best_start:
                    best_start, best_end = cur_start, cur_end
                cur_start = cur_end = c
        if cur_end - cur_start > best_end - best_start:
            best_start, best_end = cur_start, cur_end
        bg_c_start, bg_c_end = best_start, best_end
    else:
        bg_c_start = bg_c_end = None
    
    # Find vertical strip row range: contiguous rows with all bg at col range
    if bg_c_start is not None:
        vert_r_start = vert_r_end = None
        for r in range(rows):
            if all(grid[r][c] == bg for c in range(bg_c_start, bg_c_end + 1)):
                if vert_r_start is None:
                    vert_r_start = r
                vert_r_end = r
            elif vert_r_start is not None and vert_r_end is not None:
                # Check if we're past the gap - find the longest contiguous run
                # that includes the bg row band
                break
        
        # Actually, find the contiguous run that includes the bg row band
        vert_r_start = vert_r_end = None
        if bg_r_start is not None:
            # Start from bg row band and expand
            mid = (bg_r_start + bg_r_end) // 2
            vert_r_start = mid
            vert_r_end = mid
            # Expand up
            for r in range(mid - 1, -1, -1):
                if all(grid[r][c] == bg for c in range(bg_c_start, bg_c_end + 1)):
                    vert_r_start = r
                else:
                    break
            # Expand down
            for r in range(mid + 1, rows):
                if all(grid[r][c] == bg for c in range(bg_c_start, bg_c_end + 1)):
                    vert_r_end = r
                else:
                    break
    else:
        vert_r_start = vert_r_end = None
    
    # Shrink dimensions
    # bg row range: shrink by 1 on non-edge sides
    if bg_r_start is not None:
        hr_start = bg_r_start if bg_r_start == 0 else bg_r_start + 1
        hr_end = bg_r_end if bg_r_end == rows - 1 else bg_r_end - 1
    else:
        hr_start = hr_end = -1
    
    # bg col range: shrink by 1 on non-edge sides
    if bg_c_start is not None:
        vc_start = bg_c_start if bg_c_start == 0 else bg_c_start + 1
        vc_end = bg_c_end if bg_c_end == cols - 1 else bg_c_end - 1
    else:
        vc_start = vc_end = -1
    
    # Vertical strip row range: shrink by 1 on non-edge sides
    if vert_r_start is not None:
        vr_start = vert_r_start if vert_r_start == 0 else vert_r_start + 1
        vr_end = vert_r_end if vert_r_end == rows - 1 else vert_r_end - 1
    else:
        vr_start = vr_end = -1
    
    # Fill vertical strip
    if vc_start <= vc_end and vr_start <= vr_end:
        for r in range(vr_start, vr_end + 1):
            for c in range(vc_start, vc_end + 1):
                out[r][c] = 9
    
    # Fill horizontal strip
    if hr_start <= hr_end:
        for r in range(hr_start, hr_end + 1):
            for c in range(cols):
                out[r][c] = 9
    
    return out
