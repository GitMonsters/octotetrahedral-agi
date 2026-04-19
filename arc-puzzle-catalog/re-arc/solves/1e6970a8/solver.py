from collections import Counter, deque

def transform(input_grid):
    R = len(input_grid)
    C = len(input_grid[0])
    grid = [row[:] for row in input_grid]

    # Count colors
    color_count = Counter()
    for r in range(R):
        for c in range(C):
            color_count[grid[r][c]] += 1

    c9 = color_count[9]
    bg = 2

    # Find all 9 positions
    nines = set()
    for r in range(R):
        for c in range(C):
            if grid[r][c] == 9:
                nines.add((r, c))

    # Find 8-connected components of 9s
    def get_nine_clusters():
        visited = set()
        clusters = []
        for pos in nines:
            if pos in visited:
                continue
            cluster = set()
            queue = deque([pos])
            while queue:
                p = queue.popleft()
                if p in visited:
                    continue
                visited.add(p)
                cluster.add(p)
                pr, pc = p
                for dr in range(-1, 2):
                    for dc in range(-1, 2):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = pr + dr, pc + dc
                        if (nr, nc) in nines and (nr, nc) not in visited:
                            queue.append((nr, nc))
            clusters.append(cluster)
        return clusters

    # Find connected all-bg region size for a cell
    def bg_region_size(sr, sc):
        visited = set()
        queue = deque([(sr, sc)])
        visited.add((sr, sc))
        while queue:
            r, c = queue.popleft()
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < R and 0 <= nc < C and (nr,nc) not in visited and grid[nr][nc] == bg:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        return len(visited)

    # Check if a rectangle (height 2, from (r, c_start) to (r+1, c_end)) is all-bg and not 8-adjacent to any 9
    def rect_is_valid(r, c_start, c_end):
        width = c_end - c_start + 1
        # Check all cells are bg
        for dr in range(2):
            for c in range(c_start, c_end + 1):
                if grid[r + dr][c] != bg:
                    return False
        # Check 8-border has no 9
        for dr in range(-1, 3):
            for c in range(c_start - 1, c_end + 2):
                if 0 <= dr <= 1 and c_start <= c <= c_end:
                    continue
                rr, cc = r + dr, c
                if 0 <= rr < R and 0 <= cc < C and grid[rr][cc] == 9:
                    return False
        return True

    # Min Manhattan distance from any cell in rectangle to any 9
    def rect_min_dist_to_9(r, c_start, c_end):
        min_d = float('inf')
        for dr in range(2):
            for c in range(c_start, c_end + 1):
                for nr, nc in nines:
                    d = abs((r + dr) - nr) + abs(c - nc)
                    min_d = min(min_d, d)
        return min_d

    # Try count-matching approach
    valid_targets = []
    for v, cnt in color_count.items():
        if v == bg or v == 9:
            continue
        diff = cnt - c9
        if diff > 0 and diff % 2 == 0:
            valid_targets.append((diff, cnt, diff // 2))

    if valid_targets:
        # Sort by smallest need first
        valid_targets.sort()
        
        for need, target_count, width in valid_targets:
            # Find all valid height-2 rectangles of this width, not 8-adjacent to any 9
            candidates = []
            for r in range(R - 1):
                for cs in range(C - width + 1):
                    ce = cs + width - 1
                    if rect_is_valid(r, cs, ce):
                        region = bg_region_size(r, cs)
                        min_d = rect_min_dist_to_9(r, cs, ce)
                        candidates.append((r, cs, ce, region, min_d))
            
            if not candidates:
                continue
            
            # Select: largest region, then closest to 9 (smallest min_d)
            candidates.sort(key=lambda x: (-x[3], x[4]))
            best = candidates[0]
            r, cs, ce = best[0], best[1], best[2]
            
            # Fill
            for dr in range(2):
                for c in range(cs, ce + 1):
                    grid[r + dr][c] = 9
            return grid
    
    # No valid count-matching target: use cluster extension approach
    clusters = get_nine_clusters()
    small_clusters = [cl for cl in clusters if len(cl) <= 2]
    
    # Find all 2x2 all-bg blocks
    all_blocks = []
    for r in range(R - 1):
        for c in range(C - 1):
            if all(grid[r+dr][c+dc] == bg for dr in range(2) for dc in range(2)):
                all_blocks.append((r, c))
    
    # For each small cluster, find which 2x2 blocks are 8-adjacent to it
    cluster_to_blocks = {}
    block_to_clusters = {b: [] for b in all_blocks}
    
    for ci, cl in enumerate(small_clusters):
        adj_blocks = []
        for br, bc in all_blocks:
            block_cells = [(br+dr, bc+dc) for dr in range(2) for dc in range(2)]
            is_adj = False
            for bcr, bcc in block_cells:
                for cr, cc in cl:
                    if max(abs(bcr - cr), abs(bcc - cc)) == 1:
                        is_adj = True
                        break
                if is_adj:
                    break
            if is_adj:
                adj_blocks.append((br, bc))
                block_to_clusters[(br, bc)].append(ci)
        cluster_to_blocks[ci] = adj_blocks
    
    # Greedy set cover
    covered = set()
    selected_blocks = []
    uncovered_clusters = set(range(len(small_clusters)))
    
    while uncovered_clusters:
        best_block = None
        best_score = (-1, -1, float('inf'), float('inf'))
        
        for br, bc in all_blocks:
            if any(br <= sr <= br + 1 and bc <= sc <= bc + 1 for sr, sc in 
                   [(sbr + dr, sbc + dc) for sbr, sbc in selected_blocks for dr in range(2) for dc in range(2)]):
                continue
            
            covers = set(block_to_clusters[(br, bc)]) & uncovered_clusters
            if not covers:
                continue
            
            n_covers = len(covers)
            n_size1 = sum(1 for ci in covers if len(small_clusters[ci]) == 1)
            
            score = (n_covers, n_size1, -br, -bc)  # more covers, more size-1, topmost, leftmost
            if score > best_score:
                best_score = score
                best_block = (br, bc)
        
        if best_block is None:
            break
        
        selected_blocks.append(best_block)
        br, bc = best_block
        for ci in block_to_clusters[(br, bc)]:
            uncovered_clusters.discard(ci)
    
    # Fill selected blocks
    for br, bc in selected_blocks:
        for dr in range(2):
            for dc in range(2):
                grid[br + dr][bc + dc] = 9
    
    return grid
