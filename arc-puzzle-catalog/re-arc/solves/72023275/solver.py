import numpy as np
from collections import Counter

def transform(grid):
    output = np.array(grid, dtype=int)
    h, w = output.shape
    
    bg = Counter(output.flatten()).most_common(1)[0][0]
    
    # SPECIAL ROWS: rows with 2+ continuous cells of same color at the end
    special_rows = {}
    for r in range(h):
        row = output[r]
        non_bg_idx = np.where(row != bg)[0]
        if len(non_bg_idx) >= 2:
            last_color = row[non_bg_idx[-1]]
            count_at_end = 0
            for i in range(len(non_bg_idx)-1, -1, -1):
                if row[non_bg_idx[i]] == last_color:
                    count_at_end += 1
                else:
                    break
            
            if count_at_end >= 2:
                special_rows[r] = last_color
    
    # PATTERN COLUMNS: have values in non-special rows
    pattern_cols = {}
    for c in range(w):
        col = output[:, c]
        non_bg_idx = np.where(col != bg)[0]
        if len(non_bg_idx) > 0:
            non_special_idx = [r for r in non_bg_idx if r not in special_rows]
            if non_special_idx:
                non_special_colors = col[non_special_idx]
                color = Counter(non_special_colors).most_common(1)[0][0]
                pattern_cols[c] = color
    
    # FILL special rows completely
    for r, color in special_rows.items():
        for c in range(w):
            output[r, c] = color
    
    # FILL pattern columns for non-special rows  
    for c, color in pattern_cols.items():
        for r in range(h):
            if r not in special_rows:
                output[r, c] = color
    
    # MARK INTERSECTIONS with 9
    for r in special_rows:
        for c in pattern_cols:
            output[r, c] = 9
    
    return output.tolist()
