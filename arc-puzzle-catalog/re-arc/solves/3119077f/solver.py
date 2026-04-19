def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return grid
    
    rows = len(grid)
    cols = len(grid[0])
    
    # Find the dividing lines (rows and columns that are entirely one non-3 color)
    divider_row = -1
    divider_col = -1
    divider_color = -1
    
    # Find horizontal divider (a row that's entirely one non-3 color)
    for r in range(rows):
        first_val = grid[r][0]
        if first_val != 3 and all(grid[r][c] == first_val for c in range(cols)):
            divider_row = r
            divider_color = first_val
            break
    
    # Find vertical divider (a column that's entirely one non-3 color, same color as horizontal)
    for c in range(cols):
        first_val = grid[0][c]
        if first_val != 3 and all(grid[r][c] == first_val for r in range(rows)):
            divider_col = c
            break
    
    # Create output grid as a copy
    output = [row[:] for row in grid]
    
    # Identify the four quadrants
    # Top-left: rows 0 to divider_row-1, cols 0 to divider_col-1
    # Top-right: rows 0 to divider_row-1, cols divider_col+1 to cols-1
    # Bottom-left: rows divider_row+1 to rows-1, cols 0 to divider_col-1
    # Bottom-right: rows divider_row+1 to rows-1, cols divider_col+1 to cols-1
    
    # Find which quadrant has the non-3, non-divider content (the source)
    quadrants = [
        (0, divider_row, 0, divider_col, "TL"),  # top-left
        (0, divider_row, divider_col + 1, cols, "TR"),  # top-right
        (divider_row + 1, rows, 0, divider_col, "BL"),  # bottom-left
        (divider_row + 1, rows, divider_col + 1, cols, "BR")  # bottom-right
    ]
    
    def count_non_background(r_start, r_end, c_start, c_end):
        count = 0
        for r in range(r_start, r_end):
            for c in range(c_start, c_end):
                if grid[r][c] != 3 and grid[r][c] != divider_color:
                    count += 1
        return count
    
    # Find the source quadrant (the one with content)
    source_quadrant = None
    max_count = 0
    for r_start, r_end, c_start, c_end, name in quadrants:
        cnt = count_non_background(r_start, r_end, c_start, c_end)
        if cnt > max_count:
            max_count = cnt
            source_quadrant = (r_start, r_end, c_start, c_end, name)
    
    if source_quadrant is None:
        return output
    
    src_r_start, src_r_end, src_c_start, src_c_end, src_name = source_quadrant
    
    # Extract the source pattern
    src_height = src_r_end - src_r_start
    src_width = src_c_end - src_c_start
    
    # Copy the source pattern to all other quadrants
    for dst_r_start, dst_r_end, dst_c_start, dst_c_end, dst_name in quadrants:
        dst_height = dst_r_end - dst_r_start
        dst_width = dst_c_end - dst_c_start
        
        for r in range(min(src_height, dst_height)):
            for c in range(min(src_width, dst_width)):
                src_r = src_r_start + r
                src_c = src_c_start + c
                dst_r = dst_r_start + r
                dst_c = dst_c_start + c
                
                output[dst_r][dst_c] = grid[src_r][src_c]
    
    return output