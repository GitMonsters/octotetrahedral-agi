import json
import numpy as np

def transform(grid):
    grid = np.array(grid)
    h, w = grid.shape
    
    # Find the background color (most common)
    unique, counts = np.unique(grid, return_counts=True)
    bg_color = unique[np.argmax(counts)]
    
    # Find all non-background colors
    non_bg_colors = [c for c in unique if c != bg_color]
    
    # Find the marker rectangle (solid rectangle of one color)
    marker_color = None
    marker_bounds = None
    shape_color = None
    
    for color in non_bg_colors:
        mask = (grid == color)
        if not mask.any():
            continue
        rows, cols = np.where(mask)
        r_min, r_max = rows.min(), rows.max()
        c_min, c_max = cols.min(), cols.max()
        
        # Check if it's a solid rectangle
        region = grid[r_min:r_max+1, c_min:c_max+1]
        if np.all(region == color):
            marker_color = color
            marker_bounds = (r_min, r_max, c_min, c_max)
        else:
            shape_color = color
    
    if marker_color is None or shape_color is None:
        return grid.tolist()
    
    m_r_min, m_r_max, m_c_min, m_c_max = marker_bounds
    marker_height = m_r_max - m_r_min + 1
    
    # Find shape bounds (including marker pixels as part of overall object)
    combined_mask = (grid == shape_color) | (grid == marker_color)
    combined_rows, combined_cols = np.where(combined_mask)
    shape_r_min = combined_rows.min()
    shape_r_max = combined_rows.max()
    shape_c_min = combined_cols.min()
    shape_c_max = combined_cols.max()
    
    # Create output grid - start with input
    output = grid.copy()
    
    # Replace marker region with reflected content from opposite side of shape
    # The shape (excluding marker) should be symmetric
    
    # If marker is at top of shape, reflect bottom to top
    # If marker is at bottom of shape, reflect top to bottom
    
    shape_height = shape_r_max - shape_r_min + 1
    
    if m_r_min == shape_r_min:  # Marker at top
        # Reflect from bottom: row (shape_r_max - i) -> row (shape_r_min + i)
        for i in range(marker_height):
            src_row = shape_r_max - i
            dst_row = shape_r_min + i
            for c in range(shape_c_min, shape_c_max + 1):
                if grid[src_row, c] == shape_color:
                    output[dst_row, c] = shape_color
                elif grid[dst_row, c] == marker_color:
                    output[dst_row, c] = bg_color
    else:  # Marker at bottom
        # Reflect from top: row (shape_r_min + i) -> row (shape_r_max - i)
        for i in range(marker_height):
            src_row = shape_r_min + i
            dst_row = shape_r_max - i
            for c in range(shape_c_min, shape_c_max + 1):
                if grid[src_row, c] == shape_color:
                    output[dst_row, c] = shape_color
                elif grid[dst_row, c] == marker_color:
                    output[dst_row, c] = bg_color
    
    return output.tolist()


if __name__ == "__main__":
    with open('/Users/evanpieser/Downloads/re-arc_test_challenges-2026-03-16T22-48-53.json') as f:
        data = json.load(f)
    
    task = data['30d2496a']
    
    all_pass = True
    for i, ex in enumerate(task['train']):
        result = transform(ex['input'])
        expected = ex['output']
        if result == expected:
            print(f"Train {i}: PASS")
        else:
            print(f"Train {i}: FAIL")
            all_pass = False
    
    print(f"\nAll pass: {all_pass}")
