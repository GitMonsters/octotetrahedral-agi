from collections import Counter, deque

def transform(input_grid: list[list[int]]) -> list[list[int]]:
    rows = len(input_grid)
    cols = len(input_grid[0])
    
    # Find background color (most common)
    flat = [v for row in input_grid for v in row]
    bg = Counter(flat).most_common(1)[0][0]
    
    # Find connected components of non-bg cells
    visited = [[False] * cols for _ in range(rows)]
    components = []
    for r in range(rows):
        for c in range(cols):
            if input_grid[r][c] != bg and not visited[r][c]:
                comp = []
                q = deque([(r, c)])
                visited[r][c] = True
                while q:
                    cr, cc = q.popleft()
                    comp.append((cr, cc, input_grid[cr][cc]))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and input_grid[nr][nc] != bg:
                            visited[nr][nc] = True
                            q.append((nr, nc))
                components.append(comp)
    
    # Count cells per color in single-colored components
    single_color_totals = Counter()
    for comp in components:
        colors = set(v for _, _, v in comp)
        if len(colors) == 1:
            single_color_totals[list(colors)[0]] += len(comp)
    
    # Determine structure color and cells
    structure_color = None
    structure_cells = set()
    if single_color_totals:
        structure_color = single_color_totals.most_common(1)[0][0]
        for comp in components:
            colors = set(v for _, _, v in comp)
            if colors == {structure_color}:
                for r, c, v in comp:
                    structure_cells.add((r, c))
    
    # Sprite = all non-bg cells not in structure
    sprite_positions = []
    for r in range(rows):
        for c in range(cols):
            if input_grid[r][c] != bg and (r, c) not in structure_cells:
                sprite_positions.append((r, c))
    
    # Sprite bounding box
    min_r = min(r for r, c in sprite_positions)
    max_r = max(r for r, c in sprite_positions)
    min_c = min(c for r, c in sprite_positions)
    max_c = max(c for r, c in sprite_positions)
    sprite_h = max_r - min_r + 1
    sprite_w = max_c - min_c + 1
    
    # Extract sprite from input
    sprite = [[input_grid[min_r + i][min_c + j] for j in range(sprite_w)] for i in range(sprite_h)]
    
    if structure_cells:
        # Structure bounding box
        struct_min_r = min(r for r, c in structure_cells)
        struct_max_r = max(r for r, c in structure_cells)
        struct_min_c = min(c for r, c in structure_cells)
        struct_max_c = max(c for r, c in structure_cells)
        struct_h = struct_max_r - struct_min_r + 1
        struct_w = struct_max_c - struct_min_c + 1
        
        # Block size
        block_h = struct_h // sprite_h
        block_w = struct_w // sprite_w
        
        # Create block mask
        mask = [[0] * sprite_w for _ in range(sprite_h)]
        for bi in range(sprite_h):
            for bj in range(sprite_w):
                found = False
                for dr in range(block_h):
                    for dc in range(block_w):
                        r = struct_min_r + bi * block_h + dr
                        c = struct_min_c + bj * block_w + dc
                        if (r, c) in structure_cells:
                            found = True
                            break
                    if found:
                        break
                mask[bi][bj] = 1 if found else 0
        
        # Apply mask to sprite
        output = [[bg] * sprite_w for _ in range(sprite_h)]
        for i in range(sprite_h):
            for j in range(sprite_w):
                if mask[i][j]:
                    output[i][j] = sprite[i][j]
        return output
    else:
        # No separate structure: remove isolated non-dominant pixels with interior bg neighbors
        sprite_colors = [sprite[i][j] for i in range(sprite_h) for j in range(sprite_w) if sprite[i][j] != bg]
        dominant = Counter(sprite_colors).most_common(1)[0][0]
        
        # Find non-dominant, non-bg cells
        non_dom = set()
        for i in range(sprite_h):
            for j in range(sprite_w):
                if sprite[i][j] != bg and sprite[i][j] != dominant:
                    non_dom.add((i, j))
        
        # Connected components of non-dominant cells
        nd_visited = set()
        nd_components = []
        for cell in non_dom:
            if cell not in nd_visited:
                comp = []
                q = deque([cell])
                nd_visited.add(cell)
                while q:
                    cr, cc = q.popleft()
                    comp.append((cr, cc))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if (nr, nc) in non_dom and (nr, nc) not in nd_visited:
                            nd_visited.add((nr, nc))
                            q.append((nr, nc))
                nd_components.append(comp)
        
        # Remove isolated (size 1) cells that have a bg neighbor inside the sprite
        output = [row[:] for row in sprite]
        for comp in nd_components:
            if len(comp) == 1:
                r, c = comp[0]
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < sprite_h and 0 <= nc < sprite_w and sprite[nr][nc] == bg:
                        output[r][c] = bg
                        break
        return output
