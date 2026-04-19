import numpy as np
from collections import Counter

def get_intruders(grid, bg_color):
    height, width = grid.shape
    intruders = []
    for r_idx, row in enumerate(grid):
        if np.all(row == bg_color):
            intruders.append({'row': r_idx, 'color': None, 'length': 0, 'is_intruder': False})
            continue
            
        mask = row != bg_color
        if not np.any(mask):
            intruders.append({'row': r_idx, 'color': None, 'length': 0, 'is_intruder': False})
            continue
            
        length = np.sum(mask)
        color = row[mask][0]
        intruders.append({'row': r_idx, 'color': int(color), 'length': int(length), 'is_intruder': True})
    return intruders

def solve_grid(grid, bg_color):
    height, width = grid.shape
    grid = np.array(grid, dtype=int)
    
    intruders = get_intruders(grid, bg_color)
    
    # 1. Collect Stats
    all_lengths = []
    all_colors = set()
    
    for x in intruders:
        if x['is_intruder']:
            l = x['length']
            c = x['color']
            all_lengths.append(l)
            all_colors.add(c)
            
    # 2. Target Colors (Exclude 7, 9)
    excluded_colors = [7, 9]
    target_colors = sorted([c for c in all_colors if c not in excluded_colors])
    
    # 3. Target Lengths (Use ALL unique lengths)
    # We DO NOT filter lengths even if they come from excluded colors
    unique_lengths = sorted(list(set(all_lengths)), reverse=True)
            
    # 4. Build Map
    len_to_params = {}
    num_targets = len(target_colors)
    num_lengths = len(unique_lengths)
    
    if num_targets > 0:
        for i in range(num_lengths):
            l = unique_lengths[i]
            
            if i < num_targets - 1:
                c = target_colors[i]
                len_to_params[l] = (c, l)
            else:
                # Tail merge
                c = target_colors[-1]
                tail = unique_lengths[num_targets-1:]
                min_l = min(tail)
                len_to_params[l] = (c, min_l)
                
    # 5. Generate Initial Output
    output_rows_info = []
    for x in intruders:
        info = {'color': bg_color, 'length': 0, 'is_bg': True}
        if x['is_intruder']:
            l = x['length']
            if l in len_to_params:
                c, out_l = len_to_params[l]
                info = {'color': c, 'length': out_l, 'is_bg': False}
        output_rows_info.append(info)
        
    # 6. Apply Fill Rules
    for i in range(height):
        # Only fill gaps
        if not output_rows_info[i]['is_bg']:
            continue
            
        prev_output = output_rows_info[i-1] if i > 0 else None
        next_input = intruders[i+1] if i < height - 1 else None
        
        if not prev_output or prev_output['is_bg']:
            continue
            
        should_fill = False
        consume_next = False
        
        # Priority 1: Prev is Full Width
        # Must also have a next intruder (don't fill into void)
        if prev_output['length'] == width:
            if next_input and next_input['is_intruder']:
                should_fill = True
            
        # Priority 2: Next is Intruder
        elif next_input and next_input['is_intruder']:
            # If Next is Excluded Color -> Fill and Consume
            if next_input['color'] in excluded_colors:
                should_fill = True
                consume_next = True
            
            # If Next Mapped Color matches Prev Color -> Fill
            else:
                l = next_input['length']
                if l in len_to_params:
                    mapped_c, _ = len_to_params[l]
                    if mapped_c == prev_output['color']:
                        should_fill = True
                        
        if should_fill:
            output_rows_info[i] = {'color': prev_output['color'], 'length': prev_output['length'], 'is_bg': False}
            if consume_next and i < height - 1:
                # Clear next row
                output_rows_info[i+1] = {'color': bg_color, 'length': 0, 'is_bg': True}

    # 7. Draw
    output = np.full((height, width), bg_color, dtype=int)
    for r_idx, info in enumerate(output_rows_info):
        if not info['is_bg']:
            l = info['length']
            c = info['color']
            # Right aligned
            output[r_idx, width-l:] = c
            
    return output

def transform(input_grid):
    input_grid = np.array(input_grid)
    # Detect bg
    first_col = input_grid[:, 0]
    bg_color = Counter(first_col).most_common(1)[0][0]
    
    return solve_grid(input_grid, bg_color).tolist()
