def transform(grid: list[list[int]]) -> list[list[int]]:
    """
    Analyze the pattern in the grid and return a 1x1 grid with a single value.
    
    The rule appears to be related to the shape/pattern formed by the non-background color:
    - If the pattern forms an "L" shape or similar asymmetric pattern, return 8
    - If the pattern forms a symmetric cross/plus shape, return 5
    - If the pattern forms a checkerboard-like symmetric pattern, return 1
    """
    if not grid or not grid[0]:
        return [[0]]
    
    rows = len(grid)
    cols = len(grid[0])
    
    # Find the background color (most common color)
    color_counts = {}
    for r in range(rows):
        for c in range(cols):
            color = grid[r][c]
            color_counts[color] = color_counts.get(color, 0) + 1
    
    background_color = max(color_counts, key=color_counts.get)
    
    # Find all non-background cells
    non_bg_cells = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != background_color:
                non_bg_cells.append((r, c))
    
    if not non_bg_cells:
        return [[0]]
    
    # Get bounding box of the pattern
    min_r = min(cell[0] for cell in non_bg_cells)
    max_r = max(cell[0] for cell in non_bg_cells)
    min_c = min(cell[1] for cell in non_bg_cells)
    max_c = max(cell[1] for cell in non_bg_cells)
    
    height = max_r - min_r + 1
    width = max_c - min_c + 1
    
    # Normalize positions relative to bounding box
    normalized = set((r - min_r, c - min_c) for r, c in non_bg_cells)
    
    # Check for horizontal symmetry
    def is_h_symmetric(cells, h, w):
        for r, c in cells:
            mirror_c = w - 1 - c
            if (r, mirror_c) not in cells:
                return False
        return True
    
    # Check for vertical symmetry
    def is_v_symmetric(cells, h, w):
        for r, c in cells:
            mirror_r = h - 1 - r
            if (mirror_r, c) not in cells:
                return False
        return True
    
    h_sym = is_h_symmetric(normalized, height, width)
    v_sym = is_v_symmetric(normalized, height, width)
    
    # Check for point symmetry (180 degree rotation)
    def is_point_symmetric(cells, h, w):
        for r, c in cells:
            rot_r = h - 1 - r
            rot_c = w - 1 - c
            if (rot_r, rot_c) not in cells:
                return False
        return True
    
    point_sym = is_point_symmetric(normalized, height, width)
    
    # Check if it's a cross/plus shape (symmetric both ways)
    if h_sym and v_sym:
        # Fully symmetric in both directions
        # Need to distinguish between checkerboard (return 1) and cross (return 5)
        
        # Check if the pattern has a "checkerboard" structure
        # by seeing if there are distinct rectangular blocks
        
        # For checkerboard pattern: alternating filled/unfilled regions
        # For cross: a connected cross shape
        
        # Count how many "blocks" there are
        # A cross typically has cells along center rows/columns
        
        center_r = (height - 1) / 2
        center_c = (width - 1) / 2
        
        # Check if this looks like a plus/cross shape
        # A cross has more cells along center row/column
        cells_on_center_row = sum(1 for r, c in normalized if abs(r - center_r) < 1)
        cells_on_center_col = sum(1 for r, c in normalized if abs(c - center_c) < 1)
        
        total_cells = len(normalized)
        
        # For a cross shape, most cells are on center row/col
        # For checkerboard, cells are spread out in blocks
        
        # Check if pattern forms rectangular blocks in corners
        # Checkerboard typically has filled corners and center, empty in between
        
        # Let's check the structure more carefully
        # Divide into quadrants and check symmetry
        
        # Alternative: check if the pattern has "holes" - areas where
        # background shows through in a symmetric way
        
        # For checkerboard: there are rectangular "holes"
        # For cross: the "holes" are in the corners
        
        # Check corners of bounding box
        corner_filled = 0
        if (0, 0) in normalized:
            corner_filled += 1
        if (0, width-1) in normalized:
            corner_filled += 1
        if (height-1, 0) in normalized:
            corner_filled += 1
        if (height-1, width-1) in normalized:
            corner_filled += 1
        
        # Cross shape: corners are NOT filled
        # Checkerboard: corners ARE filled
        if corner_filled == 0:
            return [[5]]  # Cross shape
        else:
            return [[1]]  # Checkerboard-like
    
    # Not fully symmetric - return 8
    return [[8]]