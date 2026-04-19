def transform(input_grid):
    from collections import Counter

    rows = len(input_grid)
    cols = len(input_grid[0])

    # Find background color (most common)
    flat = [c for row in input_grid for c in row]
    bg = Counter(flat).most_common(1)[0][0]

    # Group non-background cells by color
    non_bg = {}
    for r in range(rows):
        for c in range(cols):
            if input_grid[r][c] != bg:
                non_bg.setdefault(input_grid[r][c], []).append((r, c))

    # Find connected components (4-connectivity)
    def find_components(cells):
        cell_set = set(cells)
        visited = set()
        components = []
        for cell in cells:
            if cell in visited:
                continue
            comp = []
            stack = [cell]
            visited.add(cell)
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
                    nr, nc = cr+dr, cc+dc
                    if (nr, nc) in cell_set and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        stack.append((nr, nc))
            components.append(comp)
        return components

    # Find largest connected component = template shape
    best_comp, best_color, best_size = None, None, 0
    for color, cells in non_bg.items():
        for comp in find_components(cells):
            if len(comp) > best_size:
                best_size = len(comp)
                best_comp = comp
                best_color = color

    template_cells = set(best_comp)

    # Template bounding box and pattern
    min_r = min(r for r, c in template_cells)
    max_r = max(r for r, c in template_cells)
    min_c = min(c for r, c in template_cells)
    max_c = max(c for r, c in template_cells)
    H = max_r - min_r + 1
    W = max_c - min_c + 1

    pattern = [[False]*W for _ in range(H)]
    for r, c in template_cells:
        pattern[r - min_r][c - min_c] = True

    # Pre-compute filled positions
    filled = [(lr, lc) for lr in range(H) for lc in range(W) if pattern[lr][lc]]

    # Lattice step size (shape size + 1 gap)
    step_r = H + 1
    step_c = W + 1

    # For each marker, determine its lattice direction
    direction_colors = {}
    for r in range(rows):
        for c in range(cols):
            if input_grid[r][c] != bg and (r, c) not in template_cells:
                mcolor = input_grid[r][c]
                for lr, lc in filled:
                    num_r = r - min_r - lr
                    num_c = c - min_c - lc
                    if num_r % step_r == 0 and num_c % step_c == 0:
                        dr = num_r // step_r
                        dc = num_c // step_c
                        if dr != 0 or dc != 0:
                            direction_colors[(dr, dc)] = mcolor
                            break

    # Build output grid
    output = [[bg]*cols for _ in range(rows)]

    # Place template
    for lr, lc in filled:
        output[min_r + lr][min_c + lc] = best_color

    # Propagate copies in each direction
    for (dr, dc), color in direction_colors.items():
        k = 1
        while True:
            base_r = min_r + k * dr * step_r
            base_c = min_c + k * dc * step_c
            any_placed = False
            for lr, lc in filled:
                pr, pc = base_r + lr, base_c + lc
                if 0 <= pr < rows and 0 <= pc < cols:
                    output[pr][pc] = color
                    any_placed = True
            if not any_placed:
                break
            k += 1

    return output
