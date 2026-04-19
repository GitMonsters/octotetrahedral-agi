import numpy as np
from scipy import ndimage

def transform(grid):
    """
    ARC puzzle 20d46a17 solver.
    
    Algorithm:
    1. Split the input grid either vertically (if width is even) or horizontally (if height is even)
       - If both are even, split along the longer dimension
    2. Determine which grid to use as base:
       - Use the grid with FEWER marked cells as the base (output)
       - Use the grid with MORE marked cells as the source of patterns
    3. For each marked component in the base grid, find a matching marked component in the source grid
    4. Match by shared color value, or by spatial overlap (for multi-cell components)
    5. Extract the source component's shape/pattern and place it in the output
       - For single-cell components: align by the shared color position
       - For multi-cell components: place border color adjacent to the component
    """
    h, w = grid.shape
    
    # Determine split direction
    w_even = w % 2 == 0
    h_even = h % 2 == 0
    
    if w_even and not h_even:
        split_w = True
    elif h_even and not w_even:
        split_w = False
    elif w_even and h_even:
        # Both even: split along longer dimension
        split_w = (w >= h)
    else:
        # Neither even (shouldn't happen in valid inputs)
        split_w = True
    
    if split_w:
        mid_w = w // 2
        left = grid[:, :mid_w]
        right = grid[:, mid_w:]
    else:
        mid_h = h // 2
        left = grid[:mid_h, :]
        right = grid[mid_h:, :]
    
    left_bg = np.bincount(left.flatten()).argmax()
    right_bg = np.bincount(right.flatten()).argmax()
    
    # Determine which grid to use as base (use the one with fewer marked cells)
    left_marked_count = np.sum(left != left_bg)
    right_marked_count = np.sum(right != right_bg)
    
    if right_marked_count < left_marked_count:
        # Use right as base, left as source
        base = right
        source = left
        base_bg = right_bg
        source_bg = left_bg
    else:
        # Use left as base, right as source
        base = left
        source = right
        base_bg = left_bg
        source_bg = right_bg
    
    output = base.copy()
    
    base_marked = (base != base_bg)
    base_labels, base_count = ndimage.label(base_marked)
    
    source_marked = (source != source_bg)
    source_labels, source_count = ndimage.label(source_marked)
    
    # Build component info for base
    base_comps = {}
    for bc_id in range(1, base_count + 1):
        bc_mask = (base_labels == bc_id)
        bc_pos = np.argwhere(bc_mask)
        bc_r_min, bc_c_min = bc_pos.min(axis=0)
        bc_r_max, bc_c_max = bc_pos.max(axis=0)
        bc_values = set(np.unique(base[bc_mask]))
        
        base_comps[bc_id] = {
            'mask': bc_mask,
            'pos': bc_pos,
            'r_min': bc_r_min, 'r_max': bc_r_max,
            'c_min': bc_c_min, 'c_max': bc_c_max,
            'values': bc_values,
            'is_single': len(bc_pos) == 1,
            'h_span': bc_r_max - bc_r_min + 1,
            'w_span': bc_c_max - bc_c_min + 1
        }
    
    # Build component info for source
    source_comps = {}
    for sc_id in range(1, source_count + 1):
        sc_mask = (source_labels == sc_id)
        sc_pos = np.argwhere(sc_mask)
        sc_r_min, sc_c_min = sc_pos.min(axis=0)
        sc_r_max, sc_c_max = sc_pos.max(axis=0)
        sc_values = set(np.unique(source[sc_mask]))
        shape_h = sc_r_max - sc_r_min + 1
        shape_w = sc_c_max - sc_c_min + 1
        shape = source[sc_r_min:sc_r_max+1, sc_c_min:sc_c_max+1].copy()
        
        source_comps[sc_id] = {
            'mask': sc_mask,
            'pos': sc_pos,
            'r_min': sc_r_min, 'r_max': sc_r_max,
            'c_min': sc_c_min, 'c_max': sc_c_max,
            'values': sc_values,
            'shape_h': shape_h,
            'shape_w': shape_w,
            'shape': shape
        }
    
    # Match components and apply transformations
    for bc_id, bc_info in base_comps.items():
        bc_values = bc_info['values']
        
        matched_sc_ids = []
        
        # Try color match first
        for sc_id, sc_info in source_comps.items():
            sc_values = sc_info['values']
            shared = (bc_values - {base_bg}) & (sc_values - {source_bg})
            
            if len(shared) > 0:
                matched_sc_ids.append((sc_id, 'color', shared))
        
        # Fallback to spatial match if no color match
        if len(matched_sc_ids) == 0 and not bc_info['is_single']:
            for sc_id, sc_info in source_comps.items():
                row_overlap = not (bc_info['r_max'] < sc_info['r_min'] or sc_info['r_max'] < bc_info['r_min'])
                col_overlap = not (bc_info['c_max'] < sc_info['c_min'] or sc_info['c_max'] < bc_info['c_min'])
                
                if row_overlap or col_overlap:
                    matched_sc_ids.append((sc_id, 'spatial', None))
        
        for sc_id, match_type, shared in matched_sc_ids:
            sc_info = source_comps[sc_id]
            shape = sc_info['shape']
            shape_h = sc_info['shape_h']
            shape_w = sc_info['shape_w']
            
            # Determine border color (most common non-bg, non-shared color)
            sc_values = sc_info['values']
            if shared:
                border_colors = sc_values - {source_bg} - shared
            else:
                border_colors = sc_values - {source_bg}
            
            if len(border_colors) > 0:
                color_list = []
                for val in border_colors:
                    color_list.extend([val] * np.sum(shape == val))
                border_color = np.bincount(color_list).argmax()
            else:
                border_color = list(sc_values - {source_bg})[0] if len(sc_values - {source_bg}) > 0 else base_bg
            
            if bc_info['is_single'] and shared:
                # Single-cell with color match: align by shared color position in source shape
                bc_r, bc_c = bc_info['pos'][0]
                shared_color = list(shared)[0]
                
                shared_positions = []
                for dr in range(shape_h):
                    for dc in range(shape_w):
                        if shape[dr, dc] == shared_color:
                            shared_positions.append((dr, dc))
                
                if len(shared_positions) > 0:
                    shared_r_in_shape, shared_c_in_shape = shared_positions[0]
                    
                    out_r_start = bc_r - shared_r_in_shape
                    out_c_start = bc_c - shared_c_in_shape
                    
                    for dr in range(shape_h):
                        for dc in range(shape_w):
                            out_r = out_r_start + dr
                            out_c = out_c_start + dc
                            
                            if 0 <= out_r < output.shape[0] and 0 <= out_c < output.shape[1]:
                                if shape[dr, dc] != source_bg:
                                    output[out_r, out_c] = shape[dr, dc]
            else:
                # Multi-cell: place border color adjacent to component
                if bc_info['h_span'] > bc_info['w_span']:
                    # Vertical component - place left/right borders
                    for r in range(bc_info['r_min'], bc_info['r_max'] + 1):
                        if bc_info['c_min'] - 1 >= 0:
                            output[r, bc_info['c_min'] - 1] = border_color
                        if bc_info['c_max'] + 1 < output.shape[1]:
                            output[r, bc_info['c_max'] + 1] = border_color
                else:
                    # Horizontal component - place top/bottom borders
                    for c in range(bc_info['c_min'], bc_info['c_max'] + 1):
                        if bc_info['r_min'] - 1 >= 0:
                            output[bc_info['r_min'] - 1, c] = border_color
                        if bc_info['r_max'] + 1 < output.shape[0]:
                            output[bc_info['r_max'] + 1, c] = border_color
    
    return output
