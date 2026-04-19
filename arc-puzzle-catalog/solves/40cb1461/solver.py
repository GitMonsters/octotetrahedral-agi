import numpy as np
from collections import Counter

def transform(grid):
    """
    Transformation rule:
    1. Find a large rectangular region with one dominant color (background)
    2. Trim edges where density drops below threshold
    3. Identify rare marker pixels within the trimmed region
    4. Create output same size as trimmed rectangle:
       - Rows containing markers: fill entirely with marker color
       - Other rows: pattern [bg, marker, marker, marker, bg, ...]
    """
    grid = np.array(grid)
    h, w = grid.shape
    
    # Find initial rectangular region with balanced score
    candidates = []
    
    for r1 in range(h):
        for c1 in range(w):
            for r2 in range(r1+4, min(h, r1+20)):
                for c2 in range(c1+4, min(w, c1+25)):
                    region = grid[r1:r2+1, c1:c2+1]
                    counter = Counter(region.flatten())
                    most_common = counter.most_common(1)[0]
                    color, count = most_common
                    
                    density = count / region.size
                    
                    if density > 0.82:
                        score = region.size * (density ** 2)
                        candidates.append((score, r1, r2, c1, c2, color, density))
    
    if not candidates:
        return grid.tolist()
    
    candidates.sort(reverse=True)
    _, r1, r2, c1, c2, bg_color, _ = candidates[0]
    
    region = grid[r1:r2+1, c1:c2+1]
    
    # Trim edges where density is low
    # Trim right columns
    while c2 > c1:
        col = grid[r1:r2+1, c2]
        density = np.sum(col == bg_color) / len(col)
        if density < 0.70:
            c2 -= 1
        else:
            break
    
    # Trim bottom rows
    while r2 > r1:
        row = grid[r2, c1:c2+1]
        density = np.sum(row == bg_color) / len(row)
        if density < 0.70:
            r2 -= 1
        else:
            break
    
    # Trim left columns
    while c1 < c2:
        col = grid[r1:r2+1, c1]
        density = np.sum(col == bg_color) / len(col)
        if density < 0.70:
            c1 += 1
        else:
            break
    
    # Trim top rows
    while r1 < r2:
        row = grid[r1, c1:c2+1]
        density = np.sum(row == bg_color) / len(row)
        if density < 0.70:
            r1 += 1
        else:
            break
    
    # Extract trimmed region
    region = grid[r1:r2+1, c1:c2+1]
    rect_h, rect_w = region.shape
    
    # Find marker color (rare color in the region)
    counter = Counter(region.flatten())
    marker_color = None
    
    for color, count in counter.items():
        if color != bg_color and 1 <= count <= 5:
            marker_color = color
            break
    
    # If no marker found, output is all background color
    if marker_color is None:
        output = np.full((rect_h, rect_w), bg_color, dtype=int)
        return output.tolist()
    
    # Find which rows have markers
    marker_rows = set()
    for r in range(rect_h):
        if np.any(region[r] == marker_color):
            marker_rows.add(r)
    
    # Create output
    output = np.full((rect_h, rect_w), bg_color, dtype=int)
    
    for r in range(rect_h):
        if r in marker_rows:
            # Row with marker: fill entirely with marker color
            output[r, :] = marker_color
        else:
            # Row without marker: pattern with marker in middle columns
            # Based on examples: middle columns (1 to width-2) are marker color
            if rect_w >= 5:
                # For 5-6 width: cols 1,2,3 are marker
                # For larger: adapt
                if rect_w <= 6:
                    output[r, 1:min(4, rect_w-1)] = marker_color
                else:
                    # General case: middle ~60% columns
                    mid_start = 1
                    mid_end = rect_w - 1
                    if rect_w >= 12:
                        # For width 12+: middle 60-80%
                        mid_start = rect_w // 5
                        mid_end = rect_w - mid_start
                    output[r, mid_start:mid_end] = marker_color
    
    return output.tolist()

