import sys

def transform(input_grid):
    rows = len(input_grid)
    cols = len(input_grid[0])
    
    # 1. Determine Background Color
    counts = {}
    for r in range(rows):
        for c in range(cols):
            val = input_grid[r][c]
            counts[val] = counts.get(val, 0) + 1
    bg_color = max(counts, key=counts.get)
    
    output_grid = [row[:] for row in input_grid]
    
    # 2. Identify Raw Components (Strict Connectivity)
    raw_components = []
    visited = set()
    
    for r in range(rows):
        for c in range(cols):
            if input_grid[r][c] == bg_color:
                continue
            if (r, c) in visited:
                continue
            
            comp_pixels = [(r, c)]
            visited.add((r, c))
            idx = 0
            while idx < len(comp_pixels):
                curr_r, curr_c = comp_pixels[idx]
                idx += 1
                for nr, nc in [(curr_r-1, curr_c), (curr_r+1, curr_c), (curr_r, curr_c-1), (curr_r, curr_c+1)]:
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if input_grid[nr][nc] != bg_color and (nr, nc) not in visited:
                            visited.add((nr, nc))
                            comp_pixels.append((nr, nc))
            
            comp_counts = {}
            for (cr, cc) in comp_pixels:
                val = input_grid[cr][cc]
                comp_counts[val] = comp_counts.get(val, 0) + 1
            shape_color = max(comp_counts, key=comp_counts.get)
            
            markers = []
            for (cr, cc) in comp_pixels:
                if input_grid[cr][cc] != shape_color:
                    markers.append((cr, cc, input_grid[cr][cc]))
            
            raw_components.append({
                'pixels': set(comp_pixels),
                'color': shape_color,
                'markers': markers,
                'raw_id': len(raw_components)
            })
            
    # 3. Merge into Super Components (Gap Jump)
    # Gap threshold 3 (dist 4)
    
    # Map raw_id to super_id
    raw_to_super = {i: i for i in range(len(raw_components))}
    
    # Union-Find or iterative merge
    # Since N is small, iterative merge sets
    
    super_components = []
    
    # Group raw by color
    by_color = {}
    for i, comp in enumerate(raw_components):
        c = comp['color']
        if c not in by_color: by_color[c] = []
        by_color[c].append(i)
        
    for color, indices in by_color.items():
        # Merge indices
        sets = [{i} for i in indices]
        
        while True:
            merged = False
            new_sets = []
            used_sets = [False] * len(sets)
            
            for i in range(len(sets)):
                if used_sets[i]: continue
                
                current_set = sets[i]
                current_pixels = set()
                for rid in current_set:
                    current_pixels.update(raw_components[rid]['pixels'])
                
                for j in range(i+1, len(sets)):
                    if used_sets[j]: continue
                    
                    # Check distance between set i and set j
                    # Optimize: Check distance between ANY pixel in set i and ANY pixel in set j
                    other_pixels = set()
                    for rid in sets[j]:
                        other_pixels.update(raw_components[rid]['pixels'])
                    
                    # Bounding box
                    min_r_i = min(r for r,c in current_pixels)
                    max_r_i = max(r for r,c in current_pixels)
                    min_c_i = min(c for r,c in current_pixels)
                    max_c_i = max(c for r,c in current_pixels)
                    
                    min_r_j = min(r for r,c in other_pixels)
                    max_r_j = max(r for r,c in other_pixels)
                    min_c_j = min(c for r,c in other_pixels)
                    max_c_j = max(c for r,c in other_pixels)
                    
                    if (min_r_j > max_r_i + 4) or (min_r_i > max_r_j + 4) or \
                       (min_c_j > max_c_i + 4) or (min_c_i > max_c_j + 4):
                        continue
                        
                    is_close = False
                    for (r1, c1) in current_pixels:
                        for (r2, c2) in other_pixels:
                            dist = abs(r1-r2) + abs(c1-c2)
                            if dist <= 4:
                                is_close = True
                                break
                        if is_close: break
                        
                    if is_close:
                        current_set.update(sets[j])
                        used_sets[j] = True
                        merged = True
                
                new_sets.append(current_set)
            
            sets = new_sets
            if not merged:
                break
        
        # Build Super Components from sets
        for s in sets:
            pixels = set()
            markers = []
            raw_ids = list(s)
            
            for rid in raw_ids:
                pixels.update(raw_components[rid]['pixels'])
                markers.extend(raw_components[rid]['markers'])
                
            # Bounds
            min_r, max_r = rows, -1
            for (cr, cc) in pixels:
                if cr < min_r: min_r = cr
                if cr > max_r: max_r = cr
                
            super_components.append({
                'pixels': pixels,
                'color': color,
                'markers': markers,
                'min_r': min_r,
                'max_r': max_r,
                'raw_ids': set(raw_ids) # Track which raw components belong here
            })

    # 4. Filter Noise (Size < 6)
    filtered_super_comps = []
    for sc in super_components:
        if len(sc['pixels']) < 6:
            for (cr, cc) in sc['pixels']:
                output_grid[cr][cc] = bg_color
        else:
            filtered_super_comps.append(sc)
    
    super_components = filtered_super_comps
    
    # 5. Group by Color (Global)
    color_groups = {}
    for sc in super_components:
        c = sc['color']
        if c not in color_groups: color_groups[c] = []
        color_groups[c].append(sc)
        
    # 6. Apply Transformations
    updates = {} # (r, c) -> (priority, color)
    
    for color, comps in color_groups.items():
        for sc in comps:
            for (mr, mc, m_color) in sc['markers']:
                k = m_color - color
                
                # Mode
                mode = "cross"
                if k % 3 == 0:
                    n = k // 3
                    if n % 2 != 0:
                        mode = "vert" # Local to Raw
                    else:
                        mode = "horiz" # Global Top/Bottom of Super
                
                target_box_color = bg_color
                
                # Find Raw Component containing this marker
                # (For Local operations)
                raw_owner_pixels = set()
                for rid in sc['raw_ids']:
                    # Check if (mr, mc) in raw_components[rid]
                    if (mr, mc) in raw_components[rid]['pixels']:
                        raw_owner_pixels = raw_components[rid]['pixels']
                        break
                
                # 1. Apply Local 3x3 Box (Priority 0)
                # Apply to Raw Component? Or Super?
                # Let's say Raw Component (safe)
                for br in range(mr-1, mr+2):
                    for bc in range(mc-1, mc+2):
                        if (br, bc) in raw_owner_pixels:
                            current_prio, _ = updates.get((br, bc), (-1, -1))
                            if 0 > current_prio:
                                updates[(br, bc)] = (0, target_box_color)
                
                # 2. Apply Scan Lines (Priority 1)
                
                if mode == "horiz":
                    # Global Top/Bottom Lines of SUPER Component
                    top_row = sc['min_r']
                    bottom_row = sc['max_r']
                    
                    all_targets = set()
                    for c in comps:
                        all_targets.update(c['pixels'])
                    
                    # Apply Top Line
                    for (tr, tc) in all_targets:
                        if tr == top_row:
                            current_prio, _ = updates.get((tr, tc), (-1, -1))
                            if 1 > current_prio:
                                updates[(tr, tc)] = (1, m_color)
                                
                    # Apply Bottom Line
                    for (tr, tc) in all_targets:
                        if tr == bottom_row:
                            current_prio, _ = updates.get((tr, tc), (-1, -1))
                            if 1 > current_prio:
                                updates[(tr, tc)] = (1, bg_color)
                                
                elif mode == "vert":
                    # Local Vertical Line at mc
                    # Apply to RAW Component
                    for (tr, tc) in raw_owner_pixels:
                        if tc == mc:
                            current_prio, _ = updates.get((tr, tc), (-1, -1))
                            if 1 > current_prio:
                                updates[(tr, tc)] = (1, m_color)
                                
                else: # cross
                    # Local Cross of Bg Color
                    # Apply to RAW Component? Or Super?
                    # Train 2 shapes are simple (Raw == Super).
                    # Let's use Raw Component.
                    for (tr, tc) in raw_owner_pixels:
                        if tr == mr or tc == mc:
                            current_prio, _ = updates.get((tr, tc), (-1, -1))
                            if 1 > current_prio:
                                updates[(tr, tc)] = (1, bg_color)
                                
                # 3. Apply Center (Priority 2)
                updates[(mr, mc)] = (2, m_color)
                
    # Apply updates
    for (ur, uc), (prio, color) in updates.items():
        output_grid[ur][uc] = color
            
    return output_grid
