import numpy as np
from collections import deque


def transform(grid):
    """
    Solve ARC puzzle 66bf4d60 with cluster-based expansion.
    """
    
    grid = np.array(grid)
    output = grid.copy()
    
    # Identify separators
    sep_rows = []
    sep_cols = []
    sep_val = None
    
    for i in range(grid.shape[0]):
        if len(set(grid[i])) == 1:
            sep_rows.append(i)
            if sep_val is None:
                sep_val = grid[i, 0]
    
    for j in range(grid.shape[1]):
        if len(set(grid[:, j])) == 1:
            sep_cols.append(j)
            if sep_val is None:
                sep_val = grid[0, j]
    
    if not sep_rows or not sep_cols:
        return output.tolist()
    
    # Extract blocks
    row_ranges = []
    prev_r = 0
    for sr in sep_rows:
        if sr > prev_r:
            row_ranges.append((prev_r, sr))
        prev_r = sr + 1
    if prev_r < grid.shape[0]:
        row_ranges.append((prev_r, grid.shape[0]))
    
    col_ranges = []
    prev_c = 0
    for sc in sep_cols:
        if sc > prev_c:
            col_ranges.append((prev_c, sc))
        prev_c = sc + 1
    if prev_c < grid.shape[1]:
        col_ranges.append((prev_c, grid.shape[1]))
    
    max_r = len(row_ranges)
    max_c = len(col_ranges)
    
    inp_blocks = {}
    for bi, (r_start, r_end) in enumerate(row_ranges):
        for bj, (c_start, c_end) in enumerate(col_ranges):
            block = grid[r_start:r_end, c_start:c_end]
            flat = block.flatten()
            non_sep = flat[flat != sep_val]
            if len(non_sep) > 0:
                color = non_sep[0]
                inp_blocks[(bi, bj)] = color
    
    # Find background
    color_counts = {}
    for color in inp_blocks.values():
        color_counts[color] = color_counts.get(color, 0) + 1
    background = max(color_counts, key=color_counts.get)
    
    # Find clusters using DFS
    visited = set()
    clusters = []
    
    def dfs(start_pos):
        stack = [start_pos]
        cluster = []
        local_visited = set()
        while stack:
            pos = stack.pop()
            if pos in visited or pos in local_visited:
                continue
            if inp_blocks.get(pos) == background:
                continue
            
            visited.add(pos)
            local_visited.add(pos)
            cluster.append(pos)
            
            r, c = pos
            for nr, nc in [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]:
                if 0 <= nr < max_r and 0 <= nc < max_c and (nr, nc) not in visited:
                    stack.append((nr, nc))
        
        return cluster
    
    for pos in inp_blocks:
        if inp_blocks[pos] != background and pos not in visited:
            cluster = dfs(pos)
            if cluster:
                clusters.append(cluster)
    
    result_blocks = dict(inp_blocks)
    
    # Expand mono-colored clusters only
    for cluster in clusters:
        colors = {inp_blocks.get(pos) for pos in cluster}
        
        if len(colors) == 1:
            seed_color = colors.pop()
            alt_color = 1 if seed_color == 4 else (4 if seed_color == 1 else (0 if seed_color == 3 else 3))
            
            # BFS to distance 1 from cluster seeds
            queue = deque([(pos, 0) for pos in cluster])
            visited_bfs = set(cluster)
            
            while queue:
                pos, dist = queue.popleft()
                
                if dist >= 1:
                    continue
                
                r, c = pos
                for nr, nc in [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]:
                    if (nr, nc) not in visited_bfs and 0 <= nr < max_r and 0 <= nc < max_c:
                        visited_bfs.add((nr, nc))
                        current = inp_blocks.get((nr, nc), background)
                        if current == background:
                            result_blocks[(nr, nc)] = alt_color
                            queue.append(((nr, nc), dist + 1))
    
    # Write back to output
    for (bi, bj), color in result_blocks.items():
        if bi < len(row_ranges) and bj < len(col_ranges):
            r_start, r_end = row_ranges[bi]
            c_start, c_end = col_ranges[bj]
            output[r_start:r_end, c_start:c_end] = color
    
    return output.tolist()
