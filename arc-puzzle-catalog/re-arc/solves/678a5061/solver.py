def count_colors(grid):
    """Count occurrences of each color in the grid."""
    counts = {}
    for row in grid:
        for cell in row:
            counts[cell] = counts.get(cell, 0) + 1
    return counts


def get_most_common_color(grid):
    """Return the most common color in the grid (background)."""
    counts = count_colors(grid)
    return max(counts.keys(), key=lambda c: counts[c])


def find_non_background_cells(grid, background):
    """Find all cells that are not the background color."""
    cells = {}
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if grid[r][c] != background:
                color = grid[r][c]
                if color not in cells:
                    cells[color] = []
                cells[color].append((r, c))
    return cells


def check_rectangle_corners(cells):
    """
    Check if exactly 4 cells form the corners of an axis-aligned rectangle.
    Returns (is_rectangle, r_min, r_max, c_min, c_max) or (False, None, None, None, None)
    """
    if len(cells) != 4:
        return False, None, None, None, None
    
    rows = sorted(set(r for r, c in cells))
    cols = sorted(set(c for r, c in cells))
    
    if len(rows) != 2 or len(cols) != 2:
        return False, None, None, None, None
    
    r_min, r_max = rows[0], rows[1]
    c_min, c_max = cols[0], cols[1]
    
    expected_corners = {(r_min, c_min), (r_min, c_max), (r_max, c_min), (r_max, c_max)}
    if set(cells) == expected_corners:
        # Check that interior is non-empty
        if r_max > r_min + 1 and c_max > c_min + 1:
            return True, r_min, r_max, c_min, c_max
    
    return False, None, None, None, None


def transform(grid):
    """Transform the grid according to the rules."""
    background = get_most_common_color(grid)
    non_bg_cells = find_non_background_cells(grid, background)
    
    # Check if any color forms rectangle corners
    corner_color = None
    r_min, r_max, c_min, c_max = None, None, None, None
    
    for color, cells in non_bg_cells.items():
        is_rect, rm, rM, cm, cM = check_rectangle_corners(cells)
        if is_rect:
            corner_color = color
            r_min, r_max, c_min, c_max = rm, rM, cm, cM
            break
    
    if corner_color is not None:
        # Case 1: Rectangle corners found
        # Crop the exclusive interior and map
        interior = []
        for r in range(r_min + 1, r_max):
            row = []
            for c in range(c_min + 1, c_max):
                cell = grid[r][c]
                if cell == background:
                    row.append(background)
                else:
                    row.append(corner_color)
            interior.append(row)
        return interior
    else:
        # Case 2: No rectangle corners
        # Crop the bounding box of all non-background cells and map all to background
        all_non_bg = []
        for cells in non_bg_cells.values():
            all_non_bg.extend(cells)
        
        if not all_non_bg:
            return grid
        
        rows = [r for r, c in all_non_bg]
        cols = [c for r, c in all_non_bg]
        r_min, r_max = min(rows), max(rows)
        c_min, c_max = min(cols), max(cols)
        
        # Crop and map everything to background if it's not background, otherwise keep background
        cropped = []
        for r in range(r_min, r_max + 1):
            row = []
            for c in range(c_min, c_max + 1):
                cell = grid[r][c]
                if cell == background:
                    row.append(background)
                else:
                    row.append(background)
            cropped.append(row)
        return cropped
