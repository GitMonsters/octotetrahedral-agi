def transform(input_grid):
    """The grid is a repeating wallpaper tile pattern with some cells overwritten
    by a single 'noise' color (forming blobs). Restore the underlying periodic tile.
    
    The tile has parameters (th, tw, shift) such that:
      output[r][c] = tile[r % th][(c - shift * (r // th)) % tw]
    """
    rows = len(input_grid)
    cols = len(input_grid[0])
    
    from collections import Counter
    color_counts = Counter()
    for row in input_grid:
        for val in row:
            color_counts[val] += 1
    
    all_colors = list(color_counts.keys())
    
    best_output = None
    best_area = float('inf')
    
    for nc in all_colors:
        # Find smallest horizontal period ignoring noise-colored cells
        tw = None
        for candidate_tw in range(1, cols):
            valid = True
            for r in range(rows):
                for c in range(cols - candidate_tw):
                    v1 = input_grid[r][c]
                    v2 = input_grid[r][c + candidate_tw]
                    if v1 != nc and v2 != nc and v1 != v2:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                tw = candidate_tw
                break
        
        if tw is None:
            continue
        
        # Search for tile height (th) and diagonal shift
        for th in range(1, rows):
            if th * tw >= best_area:
                break
            for shift in range(tw):
                tile = [[None] * tw for _ in range(th)]
                consistent = True
                
                for r in range(rows):
                    if not consistent:
                        break
                    for c in range(cols):
                        val = input_grid[r][c]
                        if val == nc:
                            continue
                        tr = r % th
                        tc = (c - shift * (r // th)) % tw
                        if tile[tr][tc] is None:
                            tile[tr][tc] = val
                        elif tile[tr][tc] != val:
                            consistent = False
                            break
                
                if not consistent:
                    continue
                
                # Fill remaining None positions with nc
                for tr in range(th):
                    for tc in range(tw):
                        if tile[tr][tc] is None:
                            tile[tr][tc] = nc
                
                # Count noise cells that would be changed
                changes = 0
                for r in range(rows):
                    for c in range(cols):
                        if input_grid[r][c] == nc:
                            tr = r % th
                            tc = (c - shift * (r // th)) % tw
                            if tile[tr][tc] != nc:
                                changes += 1
                
                if changes > 0 and th * tw < best_area:
                    output = []
                    for r in range(rows):
                        row_data = []
                        for c in range(cols):
                            tr = r % th
                            tc = (c - shift * (r // th)) % tw
                            row_data.append(tile[tr][tc])
                        output.append(row_data)
                    best_output = output
                    best_area = th * tw
    
    return best_output if best_output else input_grid
