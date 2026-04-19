def transform(grid):
    """
    Transform bordered rectangles: invert border and inner colors,
    and extend the original border color to adjacent background cells.
    Prefers smaller rectangles (3x3) over larger ones, but will accept larger
    rectangles if they have perfect border/inner percentages.
    """
    import numpy as np
    
    inp = np.array(grid)
    out = inp.copy()
    h, w = inp.shape
    
    # Detect background color (most common)
    from collections import Counter
    flat = inp.flatten()
    bg = Counter(flat).most_common(1)[0][0]
    
    # Find bordered rectangles (3x3 and larger)
    def find_bordered_rectangles():
        candidates = []
        for r1 in range(h):
            for c1 in range(w):
                for r2 in range(r1 + 2, h):
                    for c2 in range(c1 + 2, w):
                        rect = inp[r1:r2+1, c1:c2+1]
                        rh, rw = rect.shape
                        
                        border_cells = []
                        inner_cells = []
                        
                        for i in range(rh):
                            for j in range(rw):
                                if i == 0 or i == rh-1 or j == 0 or j == rw-1:
                                    border_cells.append(rect[i, j])
                                else:
                                    inner_cells.append(rect[i, j])
                        
                        if not border_cells or not inner_cells:
                            continue
                        
                        border_color = max(set(border_cells), key=lambda x: border_cells.count(x))
                        inner_color = max(set(inner_cells), key=lambda x: inner_cells.count(x))
                        
                        border_pct = border_cells.count(border_color) / len(border_cells)
                        inner_pct = inner_cells.count(inner_color) / len(inner_cells)
                        
                        # Require non-background border and distinct inner color
                        if (border_pct >= 0.6 and inner_pct >= 0.4 and 
                            border_color != inner_color and border_color != bg):
                            candidates.append({
                                'r1': r1, 'r2': r2, 'c1': c1, 'c2': c2,
                                'border_color': border_color,
                                'inner_color': inner_color,
                                'area': (r2-r1+1) * (c2-c1+1),
                                'size': (r2-r1+1, c2-c1+1),
                                'quality': border_pct + inner_pct,
                                'border_pct': border_pct,
                                'inner_pct': inner_pct
                            })
        return candidates
    
    def select_non_overlapping(candidates):
        # Prefer LARGER rectangles with high quality over smaller ones
        candidates.sort(key=lambda x: (-x['quality'], -max(x['size']), -x['area']))
        
        selected = []
        for cand in candidates:
            overlap = False
            for sel in selected:
                if not (cand['r2'] < sel['r1'] or cand['r1'] > sel['r2'] or
                        cand['c2'] < sel['c1'] or cand['c1'] > sel['c2']):
                    overlap = True
                    break
            if not overlap:
                selected.append(cand)
        return selected
    
    candidates = find_bordered_rectangles()
    selected = select_non_overlapping(candidates)
    
    # Apply transformation
    for rect_info in selected:
        r1, r2 = rect_info['r1'], rect_info['r2']
        c1, c2 = rect_info['c1'], rect_info['c2']
        orig_border_color = rect_info['border_color']
        orig_inner_color = rect_info['inner_color']
        
        # Invert the rectangle
        rect = inp[r1:r2+1, c1:c2+1]
        rh, rw = rect.shape
        
        inverted = rect.copy()
        for i in range(rh):
            for j in range(rw):
                if i == 0 or i == rh-1 or j == 0 or j == rw-1:
                    if inverted[i, j] == orig_border_color:
                        inverted[i, j] = orig_inner_color
                else:
                    if inverted[i, j] == orig_inner_color:
                        inverted[i, j] = orig_border_color
        
        out[r1:r2+1, c1:c2+1] = inverted
        
        # Extend original border color to adjacent background
        if r1 > 0:
            for c in range(c1, c2+1):
                if out[r1-1, c] == bg:
                    out[r1-1, c] = orig_border_color
        
        if r2 < h-1:
            for c in range(c1, c2+1):
                if out[r2+1, c] == bg:
                    out[r2+1, c] = orig_border_color
        
        if c1 > 0:
            for r in range(r1, r2+1):
                if out[r, c1-1] == bg:
                    out[r, c1-1] = orig_border_color
        
        if c2 < w-1:
            for r in range(r1, r2+1):
                if out[r, c2+1] == bg:
                    out[r, c2+1] = orig_border_color
    
    return out.tolist()


if __name__ == "__main__":
    import json
    
    with open("/tmp/re_arc_tasks_fresh/551131f7.json") as f:
        data = json.load(f)
    
    for i, ex in enumerate(data['train']):
        inp = ex['input']
        expected = ex['output']
        result = transform(inp)
        
        if result == expected:
            print(f"Train {i}: PASS")
        else:
            print(f"Train {i}: FAIL")
            print("Expected:")
            for row in expected:
                print(row)
            print("Got:")
            for row in result:
                print(row)
