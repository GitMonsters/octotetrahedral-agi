def transform(grid):
    """Transform ARC puzzle 102c6899 - Mark dividing lines."""
    import numpy as np
    
    grid = np.array(grid)
    output = grid.copy()
    
    def find_blocks(indices):
        if not indices:
            return []
        blocks = [[indices[0]]]
        for idx in indices[1:]:
            if idx == blocks[-1][-1] + 1:
                blocks[-1].append(idx)
            else:
                blocks.append([idx])
        return blocks
    
    def interior(block):
        """Interior of a block (excluding boundaries)."""
        if block[0] == 0:
            return block[:-1]  # If touches edge 0, exclude last
        elif len(block) < 2:
            return []
        else:
            return block[1:-1]  # Otherwise exclude both first and last
    
    # Find global uniform rows/cols
    u_rows = [r for r in range(grid.shape[0]) if len(set(grid[r])) == 1]
    u_cols = [c for c in range(grid.shape[1]) if len(set(grid[:, c])) == 1]
    
    r_blocks = find_blocks(u_rows)
    c_blocks = find_blocks(u_cols)
    
    # Mark interior of global blocks FULLY
    for block in r_blocks:
        for r in interior(block):
            output[r, :] = 5
    
    for block in c_blocks:
        for c in interior(block):
            output[:, c] = 5
    
    # Find sections created by col blocks
    if c_blocks:
        sections = []
        prev = 0
        for block in c_blocks:
            if block[0] > prev:
                sections.append((prev, block[0] - 1))
            prev = block[-1] + 1
        if prev < grid.shape[1]:
            sections.append((prev, grid.shape[1] - 1))
        
        for start_c, end_c in sections:
            if start_c > end_c:
                continue
            
            section = grid[:, start_c:end_c+1]
            s_u_rows = [r for r in range(section.shape[0]) if len(set(section[r, :])) == 1]
            s_r_blocks = find_blocks(s_u_rows)
            
            for sblock in s_r_blocks:
                interior_rows = interior(sblock)
                
                if start_c == 0:
                    # Before first col_block
                    end_mark_col = c_blocks[0][-1]
                    for r in interior_rows:
                        output[r, :end_mark_col] = 5
                elif any(block[-1] < start_c for block in c_blocks):
                    # After a col_block
                    prev_block = [b for b in c_blocks if b[-1] < start_c][-1]
                    start_mark_col = prev_block[-1]
                    for r in interior_rows:
                        output[r, start_mark_col:] = 5
    
    # Find sections created by row blocks
    if r_blocks:
        sections = []
        prev = 0
        for block in r_blocks:
            if block[0] > prev:
                sections.append((prev, block[0] - 1))
            prev = block[-1] + 1
        if prev < grid.shape[0]:
            sections.append((prev, grid.shape[0] - 1))
        
        for start_r, end_r in sections:
            if start_r > end_r:
                continue
            
            section = grid[start_r:end_r+1, :]
            s_u_cols = [c for c in range(section.shape[1]) if len(set(section[:, c])) == 1]
            s_c_blocks = find_blocks(s_u_cols)
            
            for sblock in s_c_blocks:
                interior_cols = interior(sblock)
                
                if start_r == 0:
                    # Before first row_block
                    end_mark_row = r_blocks[0][-1]
                    for c in interior_cols:
                        output[:end_mark_row, c] = 5
                elif any(block[-1] < start_r for block in r_blocks):
                    # After a row_block
                    prev_block = [b for b in r_blocks if b[-1] < start_r][-1]
                    start_mark_row = prev_block[-1]
                    for c in interior_cols:
                        output[start_mark_row:, c] = 5
    
    return output.tolist()
