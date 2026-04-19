def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    
    # Find the background color (most common)
    color_count = {}
    for r in range(rows):
        for c in range(cols):
            color_count[grid[r][c]] = color_count.get(grid[r][c], 0) + 1
    background = max(color_count, key=color_count.get)
    
    # Find all non-background colors
    other_colors = [c for c in color_count if c != background]
    
    # Find bounding boxes for each non-background color
    def find_bbox(color):
        min_r, max_r = rows, -1
        min_c, max_c = cols, -1
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == color:
                    min_r = min(min_r, r)
                    max_r = max(max_r, r)
                    min_c = min(min_c, c)
                    max_c = max(max_c, c)
        if max_r == -1:
            return None
        return (min_r, max_r, min_c, max_c)
    
    bboxes = {c: find_bbox(c) for c in other_colors if find_bbox(c)}
    
    # Identify frame and connector:
    # The connector's bbox contains the frame's bbox range in one dimension (extends beyond)
    # The frame's bbox is more compact
    frame_color = None
    connector_color = None
    
    if len(other_colors) == 2:
        c1, c2 = other_colors
        bb1, bb2 = bboxes[c1], bboxes[c2]
        
        # Check if c1's rows contain c2's rows (c1 is connector, c2 is frame)
        c1_contains_c2_rows = bb1[0] <= bb2[0] and bb1[1] >= bb2[1]
        c2_contains_c1_rows = bb2[0] <= bb1[0] and bb2[1] >= bb1[1]
        
        if c1_contains_c2_rows and not c2_contains_c1_rows:
            connector_color, frame_color = c1, c2
        elif c2_contains_c1_rows and not c1_contains_c2_rows:
            connector_color, frame_color = c2, c1
        else:
            # Check columns
            c1_contains_c2_cols = bb1[2] <= bb2[2] and bb1[3] >= bb2[3]
            c2_contains_c1_cols = bb2[2] <= bb1[2] and bb2[3] >= bb1[3]
            if c1_contains_c2_cols and not c2_contains_c1_cols:
                connector_color, frame_color = c1, c2
            elif c2_contains_c1_cols and not c1_contains_c2_cols:
                connector_color, frame_color = c2, c1
    
    if frame_color is None or connector_color is None:
        return grid
    
    # Get frame bbox
    frame_bb = bboxes[frame_color]
    fr_min_r, fr_max_r, fr_min_c, fr_max_c = frame_bb
    
    # Determine if connector extends vertically or horizontally beyond frame
    conn_bb = bboxes[connector_color]
    cn_min_r, cn_max_r, cn_min_c, cn_max_c = conn_bb
    
    extends_vertically = cn_min_r < fr_min_r or cn_max_r > fr_max_r
    
    def segment_indices(positions):
        """Group consecutive positions into segments"""
        if not positions:
            return []
        sorted_pos = sorted(positions)
        segments = []
        current = [sorted_pos[0]]
        for p in sorted_pos[1:]:
            if p == current[-1] + 1:
                current.append(p)
            else:
                segments.append(current)
                current = [p]
        segments.append(current)
        return segments
    
    if extends_vertically:
        # Find connector positions by row
        conn_rows = set()
        for r in range(rows):
            if any(grid[r][c] == connector_color for c in range(cols)):
                conn_rows.add(r)
        
        segments = segment_indices(conn_rows)
        
        # Find slot position in frame (where connector passes through frame)
        slot_rows = sorted([r for r in range(fr_min_r, fr_max_r + 1) 
                           if any(grid[r][c] == connector_color for c in range(fr_min_c, fr_max_c + 1))])
        slot_rel_start = slot_rows[0] - fr_min_r
        
        # Find which segment contains the current slot
        current_seg_idx = None
        for i, seg in enumerate(segments):
            if slot_rows[0] in seg:
                current_seg_idx = i
                break
        
        # Move to next segment
        if current_seg_idx is not None and current_seg_idx + 1 < len(segments):
            next_seg = segments[current_seg_idx + 1]
            target_row = next_seg[0]
            new_fr_min_r = target_row - slot_rel_start
            shift_r = new_fr_min_r - fr_min_r
        else:
            if current_seg_idx is not None and current_seg_idx > 0:
                prev_seg = segments[current_seg_idx - 1]
                target_row = prev_seg[0]
                new_fr_min_r = target_row - slot_rel_start
                shift_r = new_fr_min_r - fr_min_r
            else:
                shift_r = 0
        shift_c = 0
        
    else:
        # Horizontal connector
        conn_cols = set()
        for c in range(cols):
            if any(grid[r][c] == connector_color for r in range(rows)):
                conn_cols.add(c)
        
        segments = segment_indices(conn_cols)
        
        slot_cols = sorted([c for c in range(fr_min_c, fr_max_c + 1)
                           if any(grid[r][c] == connector_color for r in range(fr_min_r, fr_max_r + 1))])
        slot_rel_start = slot_cols[0] - fr_min_c
        
        current_seg_idx = None
        for i, seg in enumerate(segments):
            if slot_cols[0] in seg:
                current_seg_idx = i
                break
        
        if current_seg_idx is not None and current_seg_idx + 1 < len(segments):
            next_seg = segments[current_seg_idx + 1]
            target_col = next_seg[0]
            new_fr_min_c = target_col - slot_rel_start
            shift_c = new_fr_min_c - fr_min_c
        else:
            if current_seg_idx is not None and current_seg_idx > 0:
                prev_seg = segments[current_seg_idx - 1]
                target_col = prev_seg[0]
                new_fr_min_c = target_col - slot_rel_start
                shift_c = new_fr_min_c - fr_min_c
            else:
                shift_c = 0
        shift_r = 0
    
    # Build output
    output = [[background for _ in range(cols)] for _ in range(rows)]
    
    # Copy ALL connector positions (they stay in place)
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == connector_color:
                output[r][c] = connector_color
    
    # Place frame at new position (frame color only, connector inside frame was already placed)
    for r in range(fr_min_r, fr_max_r + 1):
        for c in range(fr_min_c, fr_max_c + 1):
            new_r = r + shift_r
            new_c = c + shift_c
            if 0 <= new_r < rows and 0 <= new_c < cols:
                cell = grid[r][c]
                if cell == frame_color:
                    output[new_r][new_c] = cell
    
    return output
