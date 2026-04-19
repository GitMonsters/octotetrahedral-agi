"""
ARC-AGI Task 3fde1cda Solver
"""

import numpy as np
from collections import Counter

def solve(task_input):
    grid = np.array(task_input)
    h, w = grid.shape
    
    bg_color = Counter(grid.flatten()).most_common(1)[0][0]
    
    # Find corners
    corners = None
    best_area = 0
    
    for marker_color in set(grid.flatten()):
        if marker_color == bg_color:
            continue
        
        positions = list(map(tuple, np.argwhere(grid == marker_color).tolist()))
        if len(positions) < 4:
            continue
        
        rows = sorted(set(p[0] for p in positions))
        cols = sorted(set(p[1] for p in positions))
        
        for i, r1 in enumerate(rows):
            for r2 in rows[i+1:]:
                for j, c1 in enumerate(cols):
                    for c2 in cols[j+1:]:
                        if ((r1,c1) in positions and (r1,c2) in positions and 
                            (r2,c1) in positions and (r2,c2) in positions):
                            area = (r2-r1) * (c2-c1)
                            if area > best_area and (r2-r1) >= 5 and (c2-c1) >= 5:
                                best_area = area
                                corners = (r1, r2, c1, c2, marker_color)
    
    if not corners:
        return task_input
    
    r1, r2, c1, c2, marker_color = corners
    cropped = grid[r1:r2+1, c1:c2+1].copy()
    crop_h, crop_w = cropped.shape
    
    crop_bg = Counter(cropped.flatten()).most_common(1)[0][0]
    
    # Find legend (don't exclude marker_color - it might also be used as data!)
    legend = []
    for r in range(h):
        for c in range(w):
            if r1 <= r <= r2 and c1 <= c <= c2:
                continue
            
            color = int(grid[r, c])
            # Only exclude bg_color and crop_bg from legend
            if color != bg_color and color != crop_bg:
                if color not in legend:
                    legend.append(color)
    
    marker_positions = {(0, 0), (0, crop_w-1), (crop_h-1, 0), (crop_h-1, crop_w-1)}
    
    # Find blocks
    blocks = []
    seen = set()
    
    for r in range(crop_h):
        for c in range(crop_w):
            if (r, c) in seen or cropped[r, c] == crop_bg:
                continue
            
            # Skip marker ONLY at corner positions
            if cropped[r, c] == marker_color and (r, c) in marker_positions:
                continue
            
            color = int(cropped[r, c])
            
            # Flood fill
            to_visit = [(r, c)]
            block_cells = set()
            
            while to_visit:
                rr, cc = to_visit.pop()
                if (rr, cc) in block_cells:
                    continue
                if rr < 0 or rr >= crop_h or cc < 0 or cc >= crop_w:
                    continue
                if cropped[rr, cc] == marker_color and (rr, cc) in marker_positions:
                    continue
                if cropped[rr, cc] != color:
                    continue
                
                block_cells.add((rr, cc))
                seen.add((rr, cc))
                
                for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                    to_visit.append((rr+dr, cc+dc))
            
            if block_cells:
                min_r = min(p[0] for p in block_cells)
                max_r = max(p[0] for p in block_cells)
                min_c = min(p[1] for p in block_cells)
                max_c = max(p[1] for p in block_cells)
                
                # Verify rectangle
                is_rect = True
                for rr in range(min_r, max_r+1):
                    for cc in range(min_c, max_c+1):
                        if cropped[rr, cc] != color:
                            is_rect = False
                            break
                    if not is_rect:
                        break
                
                if is_rect:
                    blocks.append({
                        'color': color,
                        'r1': min_r, 'r2': max_r,
                        'c1': min_c, 'c2': max_c
                    })
    
    # Apply transformation
    if len(legend) > 0 and len(blocks) > 0:
        for block in blocks:
            block_color = block['color']
            
            if block_color in legend:
                color_idx = legend.index(block_color)
                missing_colors = legend[:color_idx]
                
                if missing_colors:
                    block_height = block['r2'] - block['r1'] + 1
                    
                    for i, missing_color in enumerate(missing_colors):
                        new_r1 = block['r1'] - block_height * (len(missing_colors) - i)
                        new_r2 = new_r1 + block_height - 1
                        new_c1 = block['c1']
                        new_c2 = block['c2']
                        
                        if new_r1 >= 0 and new_r2 < crop_h:
                            for rr in range(new_r1, new_r2 + 1):
                                for cc in range(new_c1, new_c2 + 1):
                                    if cc < crop_w:
                                        cropped[rr, cc] = missing_color
    
    return cropped.tolist()


if __name__ == "__main__":
    import json
    
    with open('/tmp/rearc45/3fde1cda.json', 'r') as f:
        data = json.load(f)
    
    print("="*90)
    print("FINAL TEST - ALL TRAINING PAIRS")
    print("="*90)
    
    all_pass = True
    for idx, pair in enumerate(data['train']):
        predicted = solve(pair['input'])
        expected = pair['output']
        
        match = np.array_equal(predicted, expected)
        status = "✓ PASS" if match else "✗ FAIL"
        all_pass = all_pass and match
        
        print(f"Train {idx}: {status}")
    
    print(f"\n{'='*90}")
    if all_pass:
        print("✓✓✓ SUCCESS! ALL 3 TRAINING PAIRS PASS! ✓✓✓")
    else:
        print("FAILED - STILL HAVE ERRORS")
    print("="*90)

transform = solve
