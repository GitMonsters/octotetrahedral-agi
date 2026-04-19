from collections import Counter, defaultdict

def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    bg = Counter(grid[r][c] for r in range(rows) for c in range(cols)).most_common(1)[0][0]
    
    visited = set()
    comps = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == bg or (r,c) in visited:
                continue
            color = grid[r][c]
            comp = set()
            stack = [(r,c)]
            comp.add((r,c))
            visited.add((r,c))
            while stack:
                cr,cc = stack.pop()
                for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr,nc = cr+dr,cc+dc
                    if 0<=nr<rows and 0<=nc<cols and (nr,nc) not in visited and grid[nr][nc]==color:
                        visited.add((nr,nc))
                        comp.add((nr,nc))
                        stack.append((nr,nc))
            min_r = min(pr for pr,pc in comp)
            max_r = max(pr for pr,pc in comp)
            min_c = min(pc for pr,pc in comp)
            max_c = max(pc for pr,pc in comp)
            comps.append({
                'color': color, 'pixels': comp, 'size': len(comp),
                'min_r': min_r, 'max_r': max_r, 'min_c': min_c, 'max_c': max_c,
                'h': max_r-min_r+1, 'w': max_c-min_c+1,
                'center_r': (min_r+max_r)/2, 'center_c': (min_c+max_c)/2
            })
    
    color_size_groups = defaultdict(list)
    for i, comp in enumerate(comps):
        key = (comp['color'], min(comp['h'], comp['w']), max(comp['h'], comp['w']))
        color_size_groups[key].append(i)
    
    raw_frames = []  # (d, color, bracket_size, is_full_ring)
    
    def calc_frame(indices):
        group_comps = [comps[i] for i in indices]
        all_pixels = set()
        for c in group_comps:
            all_pixels |= c['pixels']
        min_r = min(pr for pr,pc in all_pixels)
        max_r = max(pr for pr,pc in all_pixels)
        min_c = min(pc for pr,pc in all_pixels)
        max_c = max(pc for pr,pc in all_pixels)
        h = max_r - min_r + 1
        w = max_c - min_c + 1
        fs = max(h, w)
        d = (fs - 1) // 2
        color = group_comps[0]['color']
        largest = max(group_comps, key=lambda c: c['size'])
        bh, bw = largest['h'], largest['w']
        bracket_size = max(bh, bw)
        perim = max(0, 2 * (bh + bw) - 4)
        is_full_ring = (bh >= 3 and bw >= 3 and largest['size'] >= perim)
        return (d, color, bracket_size, is_full_ring)
    
    for key, indices in color_size_groups.items():
        if len(indices) in (1, 2, 4):
            raw_frames.append(calc_frame(indices))
        elif len(indices) == 3:
            best_pair = None
            best_dist = float('inf')
            for i in range(len(indices)):
                for j in range(i+1, len(indices)):
                    ci = comps[indices[i]]
                    cj = comps[indices[j]]
                    dist = abs(ci['center_r'] - cj['center_r']) + abs(ci['center_c'] - cj['center_c'])
                    if dist < best_dist:
                        best_dist = dist
                        best_pair = (i, j)
            pi, pj = best_pair
            raw_frames.append(calc_frame([indices[pi], indices[pj]]))
            remaining = [indices[k] for k in range(len(indices)) if k != pi and k != pj]
            for idx in remaining:
                raw_frames.append(calc_frame([idx]))
        else:
            raw_frames.append(calc_frame(indices))
    
    # Merge frames at same (d, color): use largest bracket_size, is_full_ring only if all are
    merged = {}
    for d, color, bs, ifr in raw_frames:
        key = (d, color)
        if key not in merged:
            merged[key] = (d, color, bs, ifr)
        else:
            old_d, old_c, old_bs, old_ifr = merged[key]
            # Use the larger bracket_size, and only is_full_ring if the larger bracket says so
            if bs > old_bs:
                merged[key] = (d, color, bs, ifr)
            # If both have same bs, keep the one that's NOT full ring (brackets are more specific)
            elif bs == old_bs and not ifr:
                merged[key] = (d, color, bs, ifr)
    
    frames = list(merged.values())
    
    if not frames:
        return [[bg]]
    
    max_d = max(f[0] for f in frames)
    out_size = 2 * max_d + 1
    output = [[bg] * out_size for _ in range(out_size)]
    center = max_d
    
    # Draw from outermost to innermost
    for d, color, bracket_size, is_full_ring in sorted(frames, key=lambda f: -f[0]):
        if d == 0:
            output[center][center] = color
            continue
        
        fs = 2 * d + 1
        start = center - d
        
        if is_full_ring or bracket_size >= fs:
            for r in range(out_size):
                for c in range(out_size):
                    if max(abs(r - center), abs(c - center)) == d:
                        output[r][c] = color
        else:
            k = bracket_size
            # Draw 4 L-brackets
            for br in range(k):
                for bc in range(k):
                    if br == 0 or bc == 0:
                        # Top-left
                        output[start + br][start + bc] = color
                        # Top-right
                        output[start + br][start + fs - 1 - bc] = color
                        # Bottom-left
                        output[start + fs - 1 - br][start + bc] = color
                        # Bottom-right
                        output[start + fs - 1 - br][start + fs - 1 - bc] = color
    
    return output

