import numpy as np

def transform(grid):
    grid = np.array(grid)
    output = grid.copy()
    rows, cols = grid.shape
    
    # Find background (most common)
    values, counts = np.unique(grid, return_counts=True)
    bg = values[np.argmax(counts)]
    
    # Find the template - look for the fill color (6) pattern
    # Template has fill_color connecting block_color to anchor_color
    non_bg = [v for v in values if v != bg]
    
    # Find 2x2 blocks and single points for each color
    def find_2x2_blocks(g, color):
        blocks = []
        for r in range(rows - 1):
            for c in range(cols - 1):
                if (g[r,c] == color and g[r+1,c] == color and 
                    g[r,c+1] == color and g[r+1,c+1] == color):
                    blocks.append((r, c))
        return blocks
    
    def find_singles(g, color, exclude_2x2=None):
        singles = []
        exclude = set()
        if exclude_2x2:
            for r, c in exclude_2x2:
                exclude.update([(r,c), (r+1,c), (r,c+1), (r+1,c+1)])
        for r in range(rows):
            for c in range(cols):
                if g[r, c] == color and (r, c) not in exclude:
                    singles.append((r, c))
        return singles
    
    # Identify colors by their role
    fill_color = None
    block_color = None
    anchor_color = None
    
    for color in non_bg:
        blocks = find_2x2_blocks(grid, color)
        singles = find_singles(grid, color, blocks)
        if len(blocks) > 0 and len(singles) > 0:
            # This is the block color (has both 2x2 and singles but singles are part of template)
            block_color = color
        elif len(blocks) == 0 and len(singles) > 0:
            # Could be anchor or fill - fill color has specific L-shape pattern
            # Check if any cell has 3+ neighbors of same color (fill pattern)
            has_pattern = False
            for r, c in singles:
                count = 0
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == color:
                        count += 1
                if count >= 2:
                    has_pattern = True
                    break
            if has_pattern:
                fill_color = color
            else:
                anchor_color = color
    
    if fill_color is None or anchor_color is None or block_color is None:
        return output.tolist()
    
    # Find 2x2 blocks and single anchors
    blocks_2x2 = find_2x2_blocks(grid, block_color)
    anchors = find_singles(grid, anchor_color)
    
    # Extract template relative positions from the input
    # Find the template origin - where fill, block, and anchor meet
    fill_positions = find_singles(grid, fill_color)
    
    # Template: relative to anchor, where are fill and block?
    template_fill = []
    template_block = None
    
    for ar, ac in anchors:
        for fr, fc in fill_positions:
            if abs(fr - ar) <= 3 and abs(fc - ac) <= 3:
                template_fill.append((fr - ar, fc - ac))
    
    # Find block position in template
    for ar, ac in anchors:
        for br, bc in blocks_2x2:
            # Check if this block is close to anchor (part of template)
            for dr in range(2):
                for dc in range(2):
                    if abs((br+dr) - ar) <= 3 and abs((bc+dc) - ac) <= 3:
                        # This block is part of template
                        template_block = (br - ar, bc - ac)
                        break
    
    # If we found a template, extract it properly
    if len(template_fill) > 0:
        # Use the template to stamp at each anchor
        for ar, ac in anchors:
            # Check if this anchor already has fill neighbors (is template origin)
            is_origin = False
            for fr, fc in fill_positions:
                if abs(fr - ar) <= 2 and abs(fc - ac) <= 2:
                    is_origin = True
                    break
            
            if is_origin:
                continue  # Skip the original template
            
            # Find nearest 2x2 block
            min_dist = float('inf')
            nearest_block = None
            for br, bc in blocks_2x2:
                dist = abs(br - ar) + abs(bc - ac)
                if dist < min_dist:
                    min_dist = dist
                    nearest_block = (br, bc)
            
            if nearest_block is None:
                continue
            
            br, bc = nearest_block
            # Direction from anchor to block
            dr = 1 if br > ar else (-1 if br < ar else 0)
            dc = 1 if bc > ac else (-1 if bc < ac else 0)
            
            # Stamp L-shaped fill pattern from anchor toward block
            # Pattern: fills go from anchor toward block in L-shape
            if dr != 0 and dc != 0:
                # Diagonal - L-shape
                for i in range(1, 3):
                    nr, nc = ar + i * dr, ac
                    if 0 <= nr < rows and 0 <= nc < cols and output[nr, nc] == bg:
                        output[nr, nc] = fill_color
                    nr, nc = ar, ac + i * dc
                    if 0 <= nr < rows and 0 <= nc < cols and output[nr, nc] == bg:
                        output[nr, nc] = fill_color
                # Corner
                nr, nc = ar + dr, ac + dc
                if 0 <= nr < rows and 0 <= nc < cols and output[nr, nc] == bg:
                    output[nr, nc] = fill_color
            elif dr != 0:
                for i in range(1, 3):
                    nr = ar + i * dr
                    if 0 <= nr < rows:
                        if output[nr, ac] == bg:
                            output[nr, ac] = fill_color
                        if ac > 0 and output[nr, ac-1] == bg:
                            output[nr, ac-1] = fill_color
                        if ac < cols-1 and output[nr, ac+1] == bg:
                            output[nr, ac+1] = fill_color
            else:
                for i in range(1, 3):
                    nc = ac + i * dc
                    if 0 <= nc < cols:
                        if output[ar, nc] == bg:
                            output[ar, nc] = fill_color
                        if ar > 0 and output[ar-1, nc] == bg:
                            output[ar-1, nc] = fill_color
                        if ar < rows-1 and output[ar+1, nc] == bg:
                            output[ar+1, nc] = fill_color
    
    # For each 2x2 block, extend with fill toward anchors
    for br, bc in blocks_2x2:
        for ar, ac in anchors:
            # Direction
            dr = 1 if ar > br else (-1 if ar < br else 0) 
            dc = 1 if ac > bc else (-1 if ac < bc else 0)
            
            # Add 2x2 fill blocks adjacent to the 2x2 block in that direction
            if dr != 0 and dc != 0:
                # Diagonal
                for roff in range(2):
                    for coff in range(2):
                        nr = br + roff + 2*dr if dr > 0 else br + roff + dr - 1
                        nc = bc + coff + 2*dc if dc > 0 else bc + coff + dc - 1
                        if dr > 0:
                            nr = br + 2 + roff
                        else:
                            nr = br - 2 + roff
                        if dc > 0:
                            nc = bc + 2 + coff
                        else:
                            nc = bc - 2 + coff
                        if 0 <= nr < rows and 0 <= nc < cols and output[nr, nc] == bg:
                            output[nr, nc] = fill_color
            elif dr != 0:
                start_r = br + 2 if dr > 0 else br - 2
                for roff in range(2):
                    for coff in range(2):
                        nr, nc = start_r + roff, bc + coff
                        if 0 <= nr < rows and 0 <= nc < cols and output[nr, nc] == bg:
                            output[nr, nc] = fill_color
            elif dc != 0:
                start_c = bc + 2 if dc > 0 else bc - 2
                for roff in range(2):
                    for coff in range(2):
                        nr, nc = br + roff, start_c + coff
                        if 0 <= nr < rows and 0 <= nc < cols and output[nr, nc] == bg:
                            output[nr, nc] = fill_color
    
    return output.tolist()
