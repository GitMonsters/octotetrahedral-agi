from collections import Counter


def transform(grid):
    """
    Rule: Find two non-background connected objects. The smaller object's cropped
    bbox is the template (multicolored). The larger object's cropped bbox is a 
    monochrome mask at an integer scale of the smaller bbox. Partition the large 
    bbox into equal blocks matching the small bbox size; keep each template cell 
    iff the corresponding large block contains the large object's color, 
    otherwise replace with background.
    """
    rows, cols = len(grid), len(grid[0])
    
    # Find background color (most common)
    flat = [c for row in grid for c in row]
    bg = Counter(flat).most_common(1)[0][0]
    
    # Find all non-background cells
    non_bg_cells = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] != bg]
    if not non_bg_cells:
        return grid
    
    # Find connected components using flood fill
    visited = set()
    components = []
    
    def flood_fill(start_r, start_c):
        stack = [(start_r, start_c)]
        component = []
        while stack:
            r, c = stack.pop()
            if (r, c) in visited:
                continue
            if r < 0 or r >= rows or c < 0 or c >= cols:
                continue
            if grid[r][c] == bg:
                continue
            visited.add((r, c))
            component.append((r, c, grid[r][c]))
            # 4-connectivity
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (nr, nc) not in visited:
                    stack.append((nr, nc))
        return component
    
    for r, c in non_bg_cells:
        if (r, c) not in visited:
            comp = flood_fill(r, c)
            if comp:
                components.append(comp)
    
    if len(components) < 2:
        return grid
    
    # Determine bounding boxes and sizes
    def get_bbox(comp):
        rs = [r for r, c, v in comp]
        cs = [c for r, c, v in comp]
        return min(rs), max(rs), min(cs), max(cs)
    
    def bbox_area(comp):
        r1, r2, c1, c2 = get_bbox(comp)
        return (r2 - r1 + 1) * (c2 - c1 + 1)
    
    def is_monochrome(comp):
        colors = set(v for r, c, v in comp)
        return len(colors) == 1
    
    def get_colors(comp):
        return set(v for r, c, v in comp)
    
    # Separate into monochrome (mask candidate) and multicolor (template candidate)
    monochrome_comps = [c for c in components if is_monochrome(c)]
    multicolor_comps = [c for c in components if not is_monochrome(c)]
    
    # Template is the multicolored one; mask uses monochrome components
    if multicolor_comps and monochrome_comps:
        # Template = multicolored component (smallest by area if multiple)
        small_comp = min(multicolor_comps, key=bbox_area)
        
        # Mask = ALL monochrome cells combined (they form the mask pattern)
        # Get the mask color from the largest monochrome component
        largest_mono = max(monochrome_comps, key=bbox_area)
        mask_color = list(get_colors(largest_mono))[0]
        
        # Combine all monochrome components with the mask color
        mask_cells = set()
        for comp in monochrome_comps:
            comp_color = list(get_colors(comp))[0]
            if comp_color == mask_color:
                for r, c, v in comp:
                    mask_cells.add((r, c))
        
        # Get overall bbox of all mask cells
        if mask_cells:
            mask_rs = [r for r, c in mask_cells]
            mask_cs = [c for r, c in mask_cells]
            lr1, lr2, lc1, lc2 = min(mask_rs), max(mask_rs), min(mask_cs), max(mask_cs)
        else:
            return grid
    else:
        # Fallback: sort all by bbox area
        components.sort(key=bbox_area)
        small_comp = components[0]
        largest = components[-1]
        
        mask_cells = set((r, c) for r, c, v in largest)
        lr1, lr2, lc1, lc2 = get_bbox(largest)
    
    # Get template bbox
    sr1, sr2, sc1, sc2 = get_bbox(small_comp)
    
    small_h = sr2 - sr1 + 1
    small_w = sc2 - sc1 + 1
    large_h = lr2 - lr1 + 1
    large_w = lc2 - lc1 + 1
    
    # Determine scale factor
    scale_h = large_h // small_h
    scale_w = large_w // small_w
    
    if scale_h == 0:
        scale_h = 1
    if scale_w == 0:
        scale_w = 1
    
    # Extract template (cropped small object region from grid)
    template = []
    for r in range(sr1, sr2 + 1):
        row = []
        for c in range(sc1, sc2 + 1):
            row.append(grid[r][c])
        template.append(row)
    
    # For each cell in template, check if corresponding block in large contains mask color
    output = []
    for ti in range(small_h):
        row = []
        for tj in range(small_w):
            # Corresponding block in large object bbox
            block_r_start = lr1 + ti * scale_h
            block_r_end = lr1 + (ti + 1) * scale_h
            block_c_start = lc1 + tj * scale_w
            block_c_end = lc1 + (tj + 1) * scale_w
            
            # Check if any cell in this block contains the mask
            has_mask = False
            for br in range(block_r_start, block_r_end):
                for bc in range(block_c_start, block_c_end):
                    if (br, bc) in mask_cells:
                        has_mask = True
                        break
                if has_mask:
                    break
            
            if has_mask:
                row.append(template[ti][tj])
            else:
                row.append(bg)
        output.append(row)
    
    return output
