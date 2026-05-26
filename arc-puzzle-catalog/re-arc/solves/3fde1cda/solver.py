"""
ARC-AGI Task 3fde1cda Solver - Version 2

Complete understanding of the pattern:
1. Find 4 corner markers forming a rectangle -> defines crop region
2. Crop to that region
3. Find "legend" outside crop: vertical or horizontal sequence of colors
4. Find rectangular blocks in input (may not match all legend colors)
5. Create blocks in output for ALL legend colors, stacked vertically
   - Use the template shape/position from existing input blocks
   - Stack in legend order
"""

import numpy as np
from collections import Counter


def transform(grid):
    """Transform the input grid according to task 3fde1cda rules."""
    inp = np.array(grid)
    
    # Step 1: Identify background color (most common)
    bg_color = Counter(inp.flatten()).most_common(1)[0][0]
    
    # Step 2: Find corner markers (largest bounding box)
    corner_color = None
    crop_r1, crop_r2, crop_c1, crop_c2 = None, None, None, None
    max_area = 0
    
    for potential_corner_color in set(inp.flatten()) - {bg_color}:
        positions = [(r, c) for r in range(inp.shape[0]) for c in range(inp.shape[1]) 
                     if inp[r, c] == potential_corner_color]
        
        row_counts = Counter(p[0] for p in positions)
        col_counts = Counter(p[1] for p in positions)
        
        corner_rows = [r for r, count in row_counts.items() if count == 2]
        corner_cols = [c for c, count in col_counts.items() if count == 2]
        
        if len(corner_rows) >= 2 and len(corner_cols) >= 2:
            corner_rows = sorted(corner_rows)
            corner_cols = sorted(corner_cols)
            r1, r2 = corner_rows[0], corner_rows[-1]
            c1, c2 = corner_cols[0], corner_cols[-1]
            area = (r2 - r1 + 1) * (c2 - c1 + 1)
            
            if area > max_area:
                max_area = area
                corner_color = potential_corner_color
                crop_r1, crop_r2 = r1, r2
                crop_c1, crop_c2 = c1, c2
    
    if corner_color is None:
        return grid
    
    # Step 3: Create cropped output
    out_height = crop_r2 - crop_r1 + 1
    out_width = crop_c2 - crop_c1 + 1
    output = np.full((out_height, out_width), bg_color, dtype=inp.dtype)
    
    # Place corner markers
    output[0, 0] = corner_color
    output[0, out_width - 1] = corner_color
    output[out_height - 1, 0] = corner_color
    output[out_height - 1, out_width - 1] = corner_color
    
    # Step 4: Find legend (isolated colors outside crop region)
    # The legend cells are isolated colors outside the crop, but NOT the 4 corner markers
    corner_positions = {(crop_r1, crop_c1), (crop_r1, crop_c2), (crop_r2, crop_c1), (crop_r2, crop_c2)}
    
    legend_candidates = []
    for r in range(inp.shape[0]):
        for c in range(inp.shape[1]):
            # Outside crop region
            if r < crop_r1 or r > crop_r2 or c < crop_c1 or c > crop_c2:
                # Skip if this is a corner marker position
                if (r, c) in corner_positions:
                    continue
                    
                color = inp[r, c]
                if color != bg_color:
                    # Check if isolated (single cell or part of small line)
                    same_neighbors = 0
                    for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < inp.shape[0] and 0 <= nc < inp.shape[1]:
                            if inp[nr, nc] == color:
                                same_neighbors += 1
                    # Include if isolated or part of small sequence
                    if same_neighbors <= 1:
                        legend_candidates.append((r, c, color))
    
    # Sort to get legend sequence (try vertical first, then horizontal)
    if legend_candidates:
        # Check if they form a vertical or horizontal line
        rows = [p[0] for p in legend_candidates]
        cols = [p[1] for p in legend_candidates]
        
        # If rows are more varied, it's vertical; if cols are more varied, it's horizontal
        if max(rows) - min(rows) >= max(cols) - min(cols):
            # Vertical legend
            legend_candidates.sort(key=lambda x: (x[0], x[1]))
        else:
            # Horizontal legend
            legend_candidates.sort(key=lambda x: (x[1], x[0]))
        
        legend_sequence = [p[2] for p in legend_candidates]
    else:
        legend_sequence = []
    
    # Step 5: Find rectangular blocks in input using connected components
    def find_dense_rectangles(color):
        """Find connected rectangular regions of a given color."""
        locs = set(zip(*np.where(inp == color)))
        if len(locs) < 4:
            return []
        
        rectangles = []
        visited = set()
        
        for start_r, start_c in locs:
            if (start_r, start_c) in visited:
                continue
            
            # Find connected component
            cells_in_rect = set()
            cells_in_rect.add((start_r, start_c))
            to_check = [(start_r, start_c)]
            
            while to_check:
                r, c = to_check.pop()
                for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                    nr, nc = r + dr, c + dc
                    if (nr, nc) in locs and (nr, nc) not in cells_in_rect:
                        cells_in_rect.add((nr, nc))
                        to_check.append((nr, nc))
            
            # Check if forms a dense rectangle
            if cells_in_rect:
                rs = [r for r, c in cells_in_rect]
                cs = [c for r, c in cells_in_rect]
                min_r, max_r = min(rs), max(rs)
                min_c, max_c = min(cs), max(cs)
                expected = (max_r - min_r + 1) * (max_c - min_c + 1)
                
                # Only include if reasonably rectangular and not just a single cell
                if len(cells_in_rect) >= expected * 0.8 and expected >= 4:
                    rectangles.append((min_r, max_r, min_c, max_c))
                    visited.update(cells_in_rect)
        
        return rectangles
    
    input_blocks = []
    for color in set(inp.flatten()) - {bg_color, corner_color}:
        rects = find_dense_rectangles(color)
        for min_r, max_r, min_c, max_c in rects:
            # Check if block is within or intersects crop region
            if not (max_r < crop_r1 or min_r > crop_r2 or max_c < crop_c1 or min_c > crop_c2):
                input_blocks.append({
                    'color': color,
                    'r1': min_r, 'r2': max_r,
                    'c1': min_c, 'c2': max_c,
                    'height': max_r - min_r + 1,
                    'width': max_c - min_c + 1
                })
    
    # Sort blocks by position
    input_blocks.sort(key=lambda b: (b['r1'], b['c1']))
    
    # Step 6: Determine transformation strategy
    if legend_sequence and input_blocks:
        # Check if we have a legend with more colors than input blocks
        # This indicates we need to create stacked blocks from the legend
        
        input_block_colors = set(b['color'] for b in input_blocks)
        legend_has_extra_colors = len(set(legend_sequence) - input_block_colors) > 0
        
        if legend_has_extra_colors or len(legend_sequence) >= len(input_blocks):
            # Strategy A: Use legend to create stacked blocks
            # Get template from first input block
            template = input_blocks[0]
            block_height = template['height']
            block_width = template['width']
            
            # Determine starting position in output
            # Translate the block's position to output coordinates
            start_row = template['r1'] - crop_r1
            start_col = template['c1'] - crop_c1
            
            # Stack blocks vertically for each legend color
            current_row = start_row
            for color in legend_sequence:
                for r in range(current_row, min(current_row + block_height, out_height)):
                    for c in range(start_col, min(start_col + block_width, out_width)):
                        if 0 <= r < out_height and 0 <= c < out_width:
                            output[r, c] = color
                current_row += block_height
        else:
            # Strategy B: Just copy/reposition existing blocks
            for block in input_blocks:
                # Translate to output coordinates
                out_r1 = max(0, block['r1'] - crop_r1)
                out_r2 = min(out_height - 1, block['r2'] - crop_r1)
                out_c1 = max(0, block['c1'] - crop_c1)
                out_c2 = min(out_width - 1, block['c2'] - crop_c1)
                
                # Fill block in output
                for r in range(out_r1, out_r2 + 1):
                    for c in range(out_c1, out_c2 + 1):
                        output[r, c] = block['color']
    elif input_blocks:
        # No legend, just copy blocks
        for block in input_blocks:
            out_r1 = max(0, block['r1'] - crop_r1)
            out_r2 = min(out_height - 1, block['r2'] - crop_r1)
            out_c1 = max(0, block['c1'] - crop_c1)
            out_c2 = min(out_width - 1, block['c2'] - crop_c1)
            
            for r in range(out_r1, out_r2 + 1):
                for c in range(out_c1, out_c2 + 1):
                    output[r, c] = block['color']
    
    return output.tolist()
