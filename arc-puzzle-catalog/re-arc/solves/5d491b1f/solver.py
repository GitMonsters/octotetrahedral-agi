import numpy as np

def transform(grid):
    grid = np.array(grid)
    h, w = grid.shape
    
    # 1. Find the separator "cross"
    sep_row = -1
    sep_col = -1
    sep_color = -1
    
    # Search for a row and column that intersect and have the same color everywhere
    for r in range(h):
        # Check if row is uniform
        if len(np.unique(grid[r])) == 1:
            color = grid[r][0]
            # Check if there is a corresponding uniform column with the same color
            for c in range(w):
                if grid[r][c] == color:
                    if len(np.unique(grid[:, c])) == 1:
                        sep_row = r
                        sep_col = c
                        sep_color = color
                        break
        if sep_row != -1:
            break
            
    if sep_row == -1:
        return grid.tolist()

    # 2. Define Quadrants
    # Q1: Top-Left (0)
    # Q2: Top-Right (1)
    # Q3: Bottom-Left (2)
    # Q4: Bottom-Right (3)
    
    q1 = grid[:sep_row, :sep_col]
    q2 = grid[:sep_row, sep_col+1:]
    q3 = grid[sep_row+1:, :sep_col]
    q4 = grid[sep_row+1:, sep_col+1:]
    
    quadrants = [q1, q2, q3, q4]
    
    # 3. Identify Target (Largest area)
    areas = [q.size for q in quadrants]
    target_idx = int(np.argmax(areas))
    target_grid = quadrants[target_idx]
    
    # 4. Identify Control (Diagonal opposite)
    control_idx = 3 - target_idx
    control_grid = quadrants[control_idx]
    
    # 5. Identify Background Color from Stripes
    stripe_indices = [i for i in range(4) if i != target_idx and i != control_idx]
    
    bg_color = -1
    for idx in stripe_indices:
        q = quadrants[idx]
        if q.size > 0:
            bg_color = q.flatten()[0]
            break
            
    # 6. Transform
    th, tw = target_grid.shape
    ch, cw = control_grid.shape
    
    if ch == 0 or cw == 0:
        return target_grid.tolist()
        
    rh = th // ch
    rw = tw // cw
    
    if rh == 0: rh = 1
    if rw == 0: rw = 1
    
    output_grid = np.copy(target_grid)
    
    for r in range(th):
        for c in range(tw):
            current_color = target_grid[r, c]
            if current_color != bg_color:
                # Map based on region
                cr = r // rh
                cc = c // rw
                
                # Boundary check
                if cr >= ch: cr = ch - 1
                if cc >= cw: cc = cw - 1
                
                new_color = control_grid[cr, cc]
                output_grid[r, c] = new_color
                
    return output_grid.tolist()
