"""
Solver for 1142083f - Fold/Merge transformation

Pattern: Two halves with different backgrounds, merge by matching marker patterns.
- Detect fold direction (vertical or horizontal)
- Identify sparse half (fewer non-bg cells) and filled half (more non-bg cells)
- Match marker groups between halves (multi-cell groups have unique shapes)
- Copy filled regions with offset determined by matching markers
- Output uses sparse half's background, preserves sparse markers
"""

from collections import Counter

def flood_nonbg(grid, start, bg):
    H, W = len(grid), len(grid[0])
    all_nonbg = set((r, c) for r in range(H) for c in range(W) if grid[r][c] != bg)
    region, visited = set(), set([start])
    queue = [start]
    while queue:
        r, c = queue.pop(0)
        region.add((r, c))
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if (nr, nc) in all_nonbg and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
    return [(r, c, grid[r][c]) for r, c in region]

def get_groups_4(grid, color):
    H, W = len(grid), len(grid[0])
    cells = [(r, c) for r in range(H) for c in range(W) if grid[r][c] == color]
    if not cells:
        return []
    groups, remaining = [], set(cells)
    while remaining:
        start = remaining.pop()
        group, stack = [start], [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (nr, nc) in remaining:
                    remaining.remove((nr, nc))
                    group.append((nr, nc))
                    stack.append((nr, nc))
        groups.append(group)
    return groups

def group_sig(cells):
    if not cells:
        return frozenset(), None, None
    minr, minc = min(r for r, c in cells), min(c for r, c in cells)
    return frozenset((r - minr, c - minc) for r, c in cells), minr, minc

def transform(inp):
    H, W = len(inp), len(inp[0])
    
    # Detect fold direction by comparing backgrounds
    top_bg = Counter(inp[r][c] for r in range(H // 2) for c in range(W)).most_common(1)[0][0]
    bottom_bg = Counter(inp[r][c] for r in range(H // 2, H) for c in range(W)).most_common(1)[0][0]
    left_bg = Counter(inp[r][c] for r in range(H) for c in range(W // 2)).most_common(1)[0][0]
    right_bg = Counter(inp[r][c] for r in range(H) for c in range(W // 2, W)).most_common(1)[0][0]
    
    if top_bg != bottom_bg:
        half1, half2 = inp[:H // 2], inp[H // 2:]
        bg1, bg2 = top_bg, bottom_bg
    else:
        half1 = [row[:W // 2] for row in inp]
        half2 = [row[W // 2:] for row in inp]
        bg1, bg2 = left_bg, right_bg
    
    # Determine sparse (fewer non-bg) vs filled (more non-bg)
    h1_cnt = sum(1 for row in half1 for v in row if v != bg1)
    h2_cnt = sum(1 for row in half2 for v in row if v != bg2)
    
    if h1_cnt <= h2_cnt:
        sparse, sparse_bg = half1, bg1
        filled, filled_bg = half2, bg2
    else:
        sparse, sparse_bg = half2, bg2
        filled, filled_bg = half1, bg1
    
    oH, oW = len(sparse), len(sparse[0])
    result = [row[:] for row in sparse]
    
    # Find common colors (present in both halves, excluding their backgrounds)
    sparse_colors = set(v for row in sparse for v in row if v != sparse_bg)
    filled_colors = set(v for row in filled for v in row if v != filled_bg)
    common = sparse_colors & filled_colors
    
    used_filled_regions = set()
    
    # Track which sparse marker positions have been covered by placed regions
    covered_marker_positions = set()
    
    # Find all filled non-bg regions for shape matching
    def get_all_filled_regions():
        fH, fW = len(filled), len(filled[0])
        visited = set()
        regions = []
        for r in range(fH):
            for c in range(fW):
                if filled[r][c] != filled_bg and (r,c) not in visited:
                    region = flood_nonbg(filled, (r,c), filled_bg)
                    # region is list of (r, c, v) tuples
                    visited.update((rr, cc) for rr, cc, _ in region)
                    regions.append(region)
        return regions
    
    all_filled_regions = get_all_filled_regions()
    
    # Match by marker color groups
    for marker in common:
        sparse_groups = get_groups_4(sparse, marker)
        filled_groups = get_groups_4(filled, marker)
        
        # Multi-cell group matching (unique shapes)
        sparse_multi = [g for g in sparse_groups if len(g) >= 2]
        filled_multi = [g for g in filled_groups if len(g) >= 2]
        
        for sg in sparse_multi:
            s_sig, s_mr, s_mc = group_sig(sg)
            for fg in filled_multi:
                f_sig, f_mr, f_mc = group_sig(fg)
                if s_sig == f_sig:
                    dr, dc = s_mr - f_mr, s_mc - f_mc
                    region = flood_nonbg(filled, fg[0], filled_bg)
                    reg_id = frozenset((r, c) for r, c, _ in region)
                    if reg_id not in used_filled_regions:
                        used_filled_regions.add(reg_id)
                        for r, c, v in region:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < oH and 0 <= nc < oW:
                                if result[nr][nc] == sparse_bg:
                                    result[nr][nc] = v
                                # Mark covered marker positions
                                if v == marker:
                                    covered_marker_positions.add((nr, nc))
                    break
        
        # Single-cell group matching (only if NO multi-cell groups in sparse)
        # If there are multi-cell groups, position-based matching with specificity handles all groups better
        if not sparse_multi:
            sparse_single = [g for g in sparse_groups if len(g) == 1]
            filled_single = [g for g in filled_groups if len(g) == 1]
            
            filled_single_info = []
            for fg in filled_single:
                r, c = fg[0]
                region = flood_nonbg(filled, (r, c), filled_bg)
                reg_id = frozenset((rr, cc) for rr, cc, _ in region)
                if reg_id not in used_filled_regions:
                    filled_single_info.append((fg[0], region, reg_id))
            
            for sg in sparse_single:
                s_r, s_c = sg[0]
                for f_pos, region, reg_id in filled_single_info:
                    if reg_id in used_filled_regions:
                        continue
                    f_r, f_c = f_pos
                    dr, dc = s_r - f_r, s_c - f_c
                    used_filled_regions.add(reg_id)
                    for r, c, v in region:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < oH and 0 <= nc < oW:
                            if result[nr][nc] == sparse_bg:
                                result[nr][nc] = v
                    break
    
    # Position-based matching: for sparse marker groups that didn't match by signature,
    # find filled regions that have the same color at positions that align with sparse markers
    
    # First, collect all sparse groups across all marker colors with their specificity
    # (how many filled regions can match them)
    all_sparse_groups_with_specificity = []
    
    for marker in common:
        sparse_groups = get_groups_4(sparse, marker)
        
        for sg in sparse_groups:
            # Skip if very large group (probably already handled)
            if len(sg) > 10:
                continue
            
            sparse_cells = set(sg)
            
            # Count how many regions can match this sparse group
            matching_regions = []
            for region in all_filled_regions:
                reg_cells = [(r, c) for r, c, _ in region]
                reg_id = frozenset(reg_cells)
                
                region_marker_cells = [(r, c) for r, c, v in region if v == marker]
                if len(region_marker_cells) < len(sg):
                    continue
                
                region_marker_set = set(region_marker_cells)
                
                # Check if ANY offset works
                has_match = False
                for fr, fc in region_marker_cells:
                    for sr, sc in sg:
                        dr, dc = sr - fr, sc - fc
                        shifted_sparse = {(r - dr, c - dc) for r, c in sparse_cells}
                        if shifted_sparse <= region_marker_set:
                            has_match = True
                            break
                    if has_match:
                        break
                
                if has_match:
                    matching_regions.append(reg_id)
            
            # Specificity = number of matching regions (lower = more specific = process first)
            specificity = len(matching_regions)
            all_sparse_groups_with_specificity.append((specificity, marker, sg))
    
    # Sort by specificity (process more specific groups first)
    all_sparse_groups_with_specificity.sort(key=lambda x: x[0])
    
    # Now process sparse groups in specificity order
    for _, marker, sg in all_sparse_groups_with_specificity:
        # Skip if all markers in this group are already covered
        if all((r, c) in covered_marker_positions for r, c in sg):
            continue
        
        s_sig, _, _ = group_sig(sg)
        sparse_cells = set(sg)
        sparse_min_r = min(r for r, c in sparse_cells)
        sparse_max_r = max(r for r, c in sparse_cells)
        sparse_center_r = (sparse_min_r + sparse_max_r) / 2
        sparse_height = sparse_max_r - sparse_min_r + 1
        
        # Collect ALL (region, offset, score) candidates from all regions
        all_candidates = []
        
        for region in all_filled_regions:
            reg_cells = [(r, c) for r, c, _ in region]
            reg_id = frozenset(reg_cells)
            
            if reg_id in used_filled_regions:
                continue
            
            # Find cells in this region with the marker color
            region_marker_cells = [(r, c) for r, c, v in region if v == marker]
            if len(region_marker_cells) < len(sg):
                continue
            
            # Collect all valid offsets for this region
            region_marker_set = set(region_marker_cells)
            
            # Calculate region "density" (how rectangular/filled it is)
            rmc_rows = [r for r, c in region_marker_cells]
            rmc_cols = [c for r, c in region_marker_cells]
            rmc_height = max(rmc_rows) - min(rmc_rows) + 1
            rmc_width = max(rmc_cols) - min(rmc_cols) + 1
            rmc_density = len(region_marker_cells) / (rmc_height * rmc_width) if rmc_height * rmc_width > 0 else 0
            
            for fr, fc in region_marker_cells:
                for sr, sc in sg:
                    dr, dc = sr - fr, sc - fc
                    shifted_sparse = {(r - dr, c - dc) for r, c in sparse_cells}
                    if shifted_sparse <= region_marker_set:
                        # Calculate properties of this offset
                        out_rows = [r + dr for r, c in region_marker_cells]
                        out_min_r, out_max_r = min(out_rows), max(out_rows)
                        out_center = (out_min_r + out_max_r) / 2
                        out_height = out_max_r - out_min_r + 1
                        
                        # Score: prefer offsets where sparse markers are centered
                        # Also prefer denser (more rectangular) regions
                        center_diff = abs(sparse_center_r - out_center)
                        
                        # Primary: prefer denser regions (negative for lower = better)
                        # Secondary: center alignment
                        # Tertiary: smaller regions (avoid huge patterns)
                        score = (-rmc_density, center_diff, len(region_marker_cells))
                        all_candidates.append((score, region, dr, dc, reg_id))
        
        if all_candidates:
            # Pick best candidate
            all_candidates.sort()
            _, best_region, best_dr, best_dc, best_reg_id = all_candidates[0]
            
            # Apply the best match
            used_filled_regions.add(best_reg_id)
            for r, c, v in best_region:
                nr, nc = r + best_dr, c + best_dc
                if 0 <= nr < oH and 0 <= nc < oW:
                    if result[nr][nc] == sparse_bg:
                        result[nr][nc] = v
                    # Mark marker positions as covered (regardless of whether we wrote)
                    if v == marker:
                        covered_marker_positions.add((nr, nc))
    
    # Shape-based matching for regions based on their "hole" patterns
    # The hole cells (filled_bg in bounding box) should match sparse filled_bg cells
    # Note: A region can be placed MULTIPLE times if its hole pattern matches multiple sparse positions
    sparse_filled_bg_cells = set((r, c) for r in range(oH) for c in range(oW) 
                                  if sparse[r][c] == filled_bg)
    
    for region in all_filled_regions:
        reg_cells = [(r, c) for r, c, _ in region]
        reg_id = frozenset(reg_cells)
        
        if reg_id in used_filled_regions:
            continue
        
        # Get hole cells in bounding box
        minr = min(r for r, c in reg_cells)
        maxr = max(r for r, c in reg_cells)
        minc = min(c for r, c in reg_cells)
        maxc = max(c for r, c in reg_cells)
        
        region_set = set(reg_cells)
        hole_cells = [(r, c) for r in range(minr, maxr+1) for c in range(minc, maxc+1)
                     if filled[r][c] == filled_bg and (r,c) not in region_set]
        
        if not hole_cells:
            continue
        
        # Find ALL valid offsets and place at each one
        placements = []
        for dr in range(-oH, oH):
            for dc in range(-oW, oW):
                shifted_holes = {(r + dr, c + dc) for r, c in hole_cells}
                
                # Check if all shifted holes are valid positions and match sparse filled_bg
                valid = all(0 <= r < oH and 0 <= c < oW for r, c in shifted_holes)
                if valid and shifted_holes <= sparse_filled_bg_cells:
                    # Additional check: non-hole cells must land on sparse_bg, not sparse_filled_bg
                    # (azure cells shouldn't overwrite maroon markers)
                    non_hole_cells = [(r, c) for r, c in reg_cells]
                    shifted_non_holes = [(r + dr, c + dc) for r, c in non_hole_cells]
                    conflict = False
                    for nr, nc in shifted_non_holes:
                        if 0 <= nr < oH and 0 <= nc < oW:
                            if (nr, nc) in sparse_filled_bg_cells:
                                conflict = True
                                break
                    if not conflict:
                        placements.append((dr, dc))
        
        # Place at each valid offset
        for dr, dc in placements:
            for r in range(minr, maxr+1):
                for c in range(minc, maxc+1):
                    v = filled[r][c]
                    if v != filled_bg:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < oH and 0 <= nc < oW:
                            if result[nr][nc] == sparse_bg:
                                result[nr][nc] = v
        
        # Mark region as used (even though we placed it multiple times)
        if placements:
            used_filled_regions.add(reg_id)
    
    return result
