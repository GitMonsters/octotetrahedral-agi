def transform(grid):
    """Transform ARC puzzle 102c6899 - Mark dividing lines in a grid."""
    import numpy as np
    
    grid = np.array(grid)
    output = grid.copy()
    
    def find_contiguous_blocks(indices):
        """Group consecutive indices into blocks."""
        if not indices:
            return []
        blocks = [[indices[0]]]
        for idx in indices[1:]:
            if idx == blocks[-1][-1] + 1:
                blocks[-1].append(idx)
            else:
                blocks.append([idx])
        return blocks
    
    def get_marked(block):
        """Get indices to mark: skip first if not at edge, skip last."""
        if not block:
            return []
        if block[0] == 0:
            return block[:-1]  # Touch edge: mark all except last
        else:
            return block[1:-1]  # Interior: mark all except first and last
    
    # Find uniform rows and cols
    uniform_rows = [r for r in range(grid.shape[0]) if len(set(grid[r])) == 1]
    uniform_cols = [c for c in range(grid.shape[1]) if len(set(grid[:, c])) == 1]
    
    row_blocks = find_contiguous_blocks(uniform_rows)
    col_blocks = find_contiguous_blocks(uniform_cols)
    
    # Collect rows and cols to mark
    full_rows = set()
    full_cols = set()
    partial_cols = {}  # col -> start_row
    partial_rows = {}  # row -> start_col
    
    # Mark from global blocks
    for block in row_blocks:
        full_rows.update(get_marked(block))
    
    for block in col_blocks:
        full_cols.update(get_marked(block))
    
    # Find implicit blocks in sections created by row dividers
    if row_blocks:
        row_boundaries = set()
        for block in row_blocks:
            row_boundaries.add(block[0])
            row_boundaries.add(block[-1] + 1)
        row_boundaries = sorted(row_boundaries)
        
        # Create row ranges (sections)
        row_ranges = []
        for i in range(len(row_boundaries) - 1):
            row_ranges.append((row_boundaries[i], row_boundaries[i+1] - 1))
        if row_boundaries[-1] < grid.shape[0]:
            row_ranges.append((row_boundaries[-1], grid.shape[0] - 1))
        
        for start_row, end_row in row_ranges:
            if start_row > end_row or start_row >= grid.shape[0]:
                continue
            
            # Is this section AFTER a row block?
            is_after_row_block = any(start_row == rb[-1] + 1 for rb in row_blocks)
            
            section = grid[start_row:end_row+1, :]
            section_uniform_cols = [c for c in range(section.shape[1]) if len(set(section[:, c])) == 1]
            section_col_blocks = find_contiguous_blocks(section_uniform_cols)
            
            for block in section_col_blocks:
                interior = get_marked(block)
                if is_after_row_block:
                    # Mark only from start_row downward
                    for col in interior:
                        if col not in partial_cols:
                            partial_cols[col] = start_row
                else:
                    full_cols.update(interior)
    
    # Find implicit blocks in sections created by col dividers
    if col_blocks:
        col_boundaries = set()
        for block in col_blocks:
            col_boundaries.add(block[0])
            col_boundaries.add(block[-1] + 1)
        col_boundaries = sorted(col_boundaries)
        
        # Create col ranges (sections)
        col_ranges = []
        for i in range(len(col_boundaries) - 1):
            col_ranges.append((col_boundaries[i], col_boundaries[i+1] - 1))
        if col_boundaries[-1] < grid.shape[1]:
            col_ranges.append((col_boundaries[-1], grid.shape[1] - 1))
        
        for start_col, end_col in col_ranges:
            if start_col > end_col or start_col >= grid.shape[1]:
                continue
            
            # Is this section AFTER a col block?
            is_after_col_block = any(start_col == cb[-1] + 1 for cb in col_blocks)
            
            section = grid[:, start_col:end_col+1]
            section_uniform_rows = [r for r in range(section.shape[0]) if len(set(section[r, :])) == 1]
            section_row_blocks = find_contiguous_blocks(section_uniform_rows)
            
            for block in section_row_blocks:
                interior = get_marked(block)
                if is_after_col_block:
                    # Mark only from start_col rightward
                    for row in interior:
                        if row not in partial_rows:
                            partial_rows[row] = start_col
                else:
                    full_rows.update(interior)
    
    # Apply all marks
    for r in full_rows:
        output[r, :] = 5
    
    for c in full_cols:
        output[:, c] = 5
    
    for c, start_row in partial_cols.items():
        if c not in full_cols:
            output[start_row:, c] = 5
    
    for r, start_col in partial_rows.items():
        if r not in full_rows:
            output[r, start_col:] = 5
    
    return output.tolist()
