import numpy as np
from collections import Counter, defaultdict

def get_connected_components(grid, bg=4):
    rows, cols = len(grid), len(grid[0])
    visited = set()
    components = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg and (r, c) not in visited:
                comp = []
                stack = [(r, c)]
                visited.add((r, c))
                while stack:
                    curr_r, curr_c = stack.pop()
                    comp.append((curr_r, curr_c, grid[curr_r][curr_c]))
                    for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                        nr, nc = curr_r + dr, curr_c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            if grid[nr][nc] != bg and (nr, nc) not in visited:
                                visited.add((nr, nc))
                                stack.append((nr, nc))
                components.append(comp)
    return components

def apply_transform(points, trans_params):
    rot_idx, s0, t_anchor = trans_params
    transforms = [
        lambda r,c: (r, c),
        lambda r,c: (c, -r),
        lambda r,c: (-r, -c),
        lambda r,c: (-c, r),
        lambda r,c: (r, -c),
        lambda r,c: (c, r),
        lambda r,c: (-r, c),
        lambda r,c: (-c, -r),
    ]
    trans_func = transforms[rot_idx]
    
    res = []
    for r, c, val in points:
        r0, c0 = r - s0[0], c - s0[1]
        r1, c1 = trans_func(r0, c0)
        r2, c2 = r1 + t_anchor[0], c1 + t_anchor[1]
        res.append((r2, c2, val))
    return res

def transform(grid):
    grid_np = np.array(grid)
    rows, cols = grid_np.shape
    bg = 4
    
    components = get_connected_components(grid, bg)
    templates = [c for c in components if len(c) > 3]
    mode = "TEMPLATE" if templates else "POINT_CLOUD"
    
    output_grid = np.full((rows, cols), bg, dtype=int)
    
    if mode == "TEMPLATE":
        template_pixels = set()
        for t in templates:
            for r, c, v in t:
                template_pixels.add((r, c))
        
        all_points = []
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] != bg and (r, c) not in template_pixels:
                    all_points.append((r, c, grid[r][c]))
        
        grid_map = {}
        for r, c, v in all_points:
            grid_map[(r, c)] = v
            
        transforms = [
            lambda r,c: (r, c),
            lambda r,c: (c, -r),
            lambda r,c: (-r, -c),
            lambda r,c: (-c, r),
            lambda r,c: (r, -c),
            lambda r,c: (c, r),
            lambda r,c: (-r, c),
            lambda r,c: (-c, -r),
        ]
        
        for template in templates:
            colors = [p[2] for p in template]
            body_color = Counter(colors).most_common(1)[0][0]
            skeleton = [p for p in template if p[2] != body_color]
            if not skeleton: skeleton = template
            
            pivot_color = Counter([p[2] for p in skeleton]).most_common()[-1][0]
            skeleton_anchors = [p for p in skeleton if p[2] == pivot_color]
            grid_candidates = [p for p in all_points if p[2] == pivot_color]
            
            # Use instance_predictions map
            # Key: tuple(sorted(instance_points))
            # Value: list of (spread, body)
            instance_predictions = defaultdict(list)
            
            for cand in grid_candidates:
                for anchor in skeleton_anchors:
                    for rot_idx, trans_func in enumerate(transforms):
                        valid = True
                        current_instance_points = []
                        for sk_p in skeleton:
                            dr, dc = sk_p[0] - anchor[0], sk_p[1] - anchor[1]
                            r_rot, c_rot = trans_func(dr, dc)
                            r_abs, c_abs = r_rot + cand[0], c_rot + cand[1]
                            if (r_abs, c_abs) not in grid_map or grid_map[(r_abs, c_abs)] != sk_p[2]:
                                valid = False
                                break
                            current_instance_points.append((r_abs, c_abs))
                        
                        if valid:
                            instance_key = tuple(sorted(current_instance_points))
                            trans_params = (rot_idx, anchor, cand)
                            new_body = apply_transform(template, trans_params)
                            
                            # Calculate Spread (Sum of Sq Dist to Centroid)
                            if len(current_instance_points) > 0:
                                cr = sum(p[0] for p in current_instance_points) / len(current_instance_points)
                                cc = sum(p[1] for p in current_instance_points) / len(current_instance_points)
                            else:
                                cr, cc = cand[0], cand[1]
                                
                            spread = 0
                            for br, bc, bv in new_body:
                                spread += (br - cr)**2 + (bc - cc)**2
                                
                            instance_predictions[instance_key].append((spread, new_body))
            
            for key, predictions in instance_predictions.items():
                if not predictions: continue
                # Filter by Min Spread first
                min_spread = min(p[0] for p in predictions)
                best_matches = [p[1] for p in predictions if abs(p[0] - min_spread) < 0.001]
                
                # Voting Logic
                pixel_counts = Counter()
                for body in best_matches:
                    for p in body:
                        pixel_counts[p] += 1
                
                # Keep pixels present in >= 40% of best matches
                threshold = max(1, len(best_matches) * 0.4)
                final_pixels = set()
                for p, count in pixel_counts.items():
                    if count >= threshold:
                        final_pixels.add(p)
                        
                for br, bc, bv in final_pixels:
                    if 0 <= br < rows and 0 <= bc < cols:
                        output_grid[br][bc] = bv

    else:
        # POINT_CLOUD mode (Train 2)
        all_points = []
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] != bg:
                    all_points.append((r, c, grid[r][c]))
        
        if len(all_points) < 2: return grid
            
        pair_counts = defaultdict(int)
        pairs = []
        for i in range(len(all_points)):
            for j in range(i+1, len(all_points)):
                p1, p2 = all_points[i], all_points[j]
                dr, dc = p2[0]-p1[0], p2[1]-p1[1]
                dist_sq = dr*dr + dc*dc
                c1, c2 = sorted((p1[2], p2[2]))
                key = (c1, c2, dist_sq)
                pair_counts[key] += 1
                pairs.append((key, p1, p2))
                
        if not pair_counts: return output_grid.tolist()
        best_key, _ = max(pair_counts.items(), key=lambda x: x[1])
        matching_pairs = [p for p in pairs if p[0] == best_key]
        
        c_min, c_max = best_key[0], best_key[1]
        bias_counts = {"min_above": 0, "max_above": 0}
        
        for _, p_a, p_b in matching_pairs:
            if c_min == c_max: continue
            if p_a[2] == c_min: pa, pb = p_a, p_b
            else: pa, pb = p_b, p_a
            if pa[0] < pb[0]: bias_counts["min_above"] += 1
            elif pa[0] > pb[0]: bias_counts["max_above"] += 1
                
        dominant_bias = None
        if bias_counts["min_above"] > bias_counts["max_above"]: dominant_bias = "min_above"
        elif bias_counts["max_above"] > bias_counts["min_above"]: dominant_bias = "max_above"
            
        keep_points = set()
        for _, p_a, p_b in matching_pairs:
            if c_min == c_max:
                keep_points.add((p_a[0], p_a[1], p_a[2]))
                keep_points.add((p_b[0], p_b[1], p_b[2]))
                continue
            if p_a[2] == c_min: pa, pb = p_a, p_b
            else: pa, pb = p_b, p_a
            bias = None
            if pa[0] < pb[0]: bias = "min_above"
            elif pa[0] > pb[0]: bias = "max_above"
            
            if dominant_bias is None or bias == dominant_bias or bias is None:
                keep_points.add((p_a[0], p_a[1], p_a[2]))
                keep_points.add((p_b[0], p_b[1], p_b[2]))
                
        for r, c, v in keep_points:
            output_grid[r][c] = v
            
    return output_grid.tolist()
