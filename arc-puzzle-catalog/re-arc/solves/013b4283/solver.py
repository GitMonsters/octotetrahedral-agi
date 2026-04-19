"""
Solver for ARC task 013b4283

Pattern: Project edge markers onto block boundaries or create intersections.

Two cases:
1. Interior block of 5s exists: Project edge markers onto the block boundaries
   with reflection symmetry when span doesn't cover the block center.
2. No interior block: Create intersections where row markers meet column markers,
   with special handling for edge rows that fill the full column span.
"""
from collections import Counter


def transform(grid):
    H, W = len(grid), len(grid[0])
    
    flat = [c for row in grid for c in row]
    bg = Counter(flat).most_common(1)[0][0]
    
    # Find interior 5s (the main block)
    block_cells = [(r, c) for r in range(H) for c in range(W) 
                   if grid[r][c] == 5 and 0 < r < H-1 and 0 < c < W-1]
    
    out = [row[:] for row in grid]
    
    if block_cells:
        # CASE 1: Interior block exists
        br_min, br_max = min(r for r, c in block_cells), max(r for r, c in block_cells)
        bc_min, bc_max = min(c for r, c in block_cells), max(c for r, c in block_cells)
        bc_center = (bc_min + bc_max) / 2.0
        br_center = (br_min + br_max) / 2.0
        
        # Only count non-bg AND non-5 values as markers
        top_markers = [c for c in range(W) if grid[0][c] != bg and grid[0][c] != 5]
        bot_markers = [c for c in range(W) if grid[H-1][c] != bg and grid[H-1][c] != 5]
        left_markers = [r for r in range(H) if grid[r][0] != bg and grid[r][0] != 5]
        right_markers = [r for r in range(H) if grid[r][W-1] != bg and grid[r][W-1] != 5]
        
        # Top edge: full span between markers (with extension if single marker)
        if top_markers:
            span_min, span_max = min(top_markers), max(top_markers)
            if len(top_markers) == 1 and span_min > bc_min:
                span_min -= 1
            for c in range(span_min, span_max + 1):
                if bc_min <= c <= bc_max:
                    out[br_min][c] = grid[0][c]
        
        # Bottom edge: reflection if span doesn't contain center
        if bot_markers:
            span_min, span_max = min(bot_markers), max(bot_markers)
            if span_min <= bc_center <= span_max:
                affected_cols = set(range(span_min, span_max + 1))
            else:
                affected_cols = set(range(span_min, span_max + 1))
                for c in range(span_min, span_max + 1):
                    reflected = int(2 * bc_center - c)
                    affected_cols.add(reflected)
            for c in affected_cols:
                if bc_min <= c <= bc_max:
                    out[br_max][c] = grid[H-1][c]
        
        # Left edge: markers + extension (one cell toward block top)
        if left_markers:
            affected_rows = set(left_markers)
            topmost = min(left_markers)
            if topmost > br_min:
                affected_rows.add(topmost - 1)
            for r in affected_rows:
                if br_min <= r <= br_max:
                    out[r][bc_min] = grid[r][0]
        
        # Right edge: just markers
        if right_markers:
            for r in right_markers:
                if br_min <= r <= br_max:
                    out[r][bc_max] = grid[r][W-1]
    
    else:
        # CASE 2: No interior block - create intersections
        left_markers = {r: grid[r][0] for r in range(H) if grid[r][0] != bg}
        right_markers = {r: grid[r][W-1] for r in range(H) if grid[r][W-1] != bg}
        top_markers = {c: grid[0][c] for c in range(W) if grid[0][c] != bg}
        bottom_markers = {c: grid[H-1][c] for c in range(W) if grid[H-1][c] != bg}
        
        row_marker_rows = sorted(set(left_markers.keys()) | set(right_markers.keys()))
        col_marker_cols = sorted(set(top_markers.keys()) | set(bottom_markers.keys()))
        
        if not row_marker_rows or not col_marker_cols:
            return out
        
        first_row = min(row_marker_rows)
        last_row = max(row_marker_rows)
        
        # Column span is from min col to max col + 1
        col_span_min = min(col_marker_cols)
        col_span_max = max(col_marker_cols) + 1
        
        for r in row_marker_rows:
            left_val = left_markers.get(r, bg)
            right_val = right_markers.get(r, bg)
            has_both = (left_val != bg and right_val != bg)
            is_edge_row = (r == first_row or r == last_row)
            
            if has_both or is_edge_row:
                cols_to_fill = range(col_span_min, col_span_max + 1)
            elif left_val != bg:
                cols_to_fill = [col_span_min]
            else:
                cols_to_fill = [col_span_max]
            
            for c in cols_to_fill:
                top_val = top_markers.get(c, bg)
                bottom_val = bottom_markers.get(c, bg)
                
                row_vals = [v for v in [left_val, right_val] if v != bg]
                col_vals = [v for v in [top_val, bottom_val] if v != bg]
                
                # Determine value based on priority rules
                if col_vals and all(v == 5 for v in col_vals):
                    out[r][c] = 5
                elif row_vals and len(set(row_vals)) == 1:
                    row_val = row_vals[0]
                    if not col_vals or row_val in col_vals:
                        out[r][c] = row_val
                    else:
                        edges = [(c, left_val), (W-1-c, right_val), (r, top_val), (H-1-r, bottom_val)]
                        valid = [(d, v) for d, v in edges if v != bg]
                        valid.sort(key=lambda x: x[0])
                        out[r][c] = valid[0][1]
                elif col_vals:
                    edges = [(c, left_val), (W-1-c, right_val), (r, top_val), (H-1-r, bottom_val)]
                    valid = [(d, v) for d, v in edges if v != bg]
                    valid.sort(key=lambda x: x[0])
                    out[r][c] = valid[0][1]
                elif row_vals:
                    out[r][c] = row_vals[0]
    
    return out
