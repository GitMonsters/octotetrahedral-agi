import collections

def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    
    # Identify background color (most frequent)
    flat = [x for r in grid for x in r]
    if not flat: return grid
    bg_color = max(set(flat), key=flat.count)
    
    output_grid = [row[:] for row in grid]
    
    # Three-layer flower patterns around a center point
    patterns = {
        'inner_square': [(-1, -1), (-1, 1), (1, -1), (1, 1)],
        'diamond': [(-2, 0), (2, 0), (0, -2), (0, 2)],
        'outer_square': [(-2, -2), (-2, 2), (2, -2), (2, 2)]
    }

    fills = {}
    
    # Find all non-background cells as potential flower centers
    for r in range(rows):
        for c in range(cols):
            center_val = grid[r][c]
            if center_val == bg_color:
                continue
            
            layer_data = {}
            active_count = 0
            
            # Analyze each pattern layer
            for pname, offsets in patterns.items():
                positions = []
                values = []
                valid = True
                for dr, dc in offsets:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        positions.append((nr, nc))
                        values.append(grid[nr][nc])
                    else:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                counts = collections.Counter([v for v in values if v != bg_color])
                if not counts:
                    continue
                
                target_color, count = counts.most_common(1)[0]
                bg_count = sum(1 for v in values if v == bg_color)
                
                # Layer is active if at least 2 cells have the same non-bg color
                if count >= 2:
                    active_count += 1
                    layer_data[pname] = {
                        'positions': positions,
                        'values': values,
                        'target': target_color,
                        'count': count,
                        'bg_count': bg_count
                    }
            
            # Only process centers with all 3 layers active (strong flower pattern)
            if active_count < 3:
                continue
            
            # Check if outer_square is already complete (no bg cells)
            # If so, the inner layers shouldn't produce new fills
            outer_complete = False
            if 'outer_square' in layer_data and layer_data['outer_square']['bg_count'] == 0:
                outer_complete = True
            
            # Generate fills from active layers
            for pname, data in layer_data.items():
                # Skip inner_square and diamond if outer is complete
                # This prevents filling inside already-complete patterns
                if outer_complete and pname in ['inner_square', 'diamond']:
                    continue
                
                for pos, val in zip(data['positions'], data['values']):
                    if val == bg_color:
                        fills[pos] = data['target']
    
    # Apply all fills
    for (r, c), color in fills.items():
        output_grid[r][c] = color
    
    return output_grid
