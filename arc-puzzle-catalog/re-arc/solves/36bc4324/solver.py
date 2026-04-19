from collections import Counter

def get_components(grid, fg):
    rows, cols = len(grid), len(grid[0])
    visited = set()
    components = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == fg and (r, c) not in visited:
                comp = set()
                queue = [(r, c)]
                visited.add((r, c))
                while queue:
                    cr, cc = queue.pop(0)
                    comp.add((cr, cc))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited and grid[nr][nc] == fg:
                            visited.add((nr, nc))
                            queue.append((nr, nc))
                components.append(comp)
    return components

def can_tile_2x2(cells):
    if len(cells) % 4 != 0:
        return False
    if not cells:
        return True
    remaining = set(cells)
    while remaining:
        r, c = min(remaining)
        block = {(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)}
        if block.issubset(remaining):
            remaining -= block
        else:
            return False
    return True

def find_line_to_remove(comp):
    comp_list = sorted(comp)
    for r, c in comp_list:
        line_h = {(r, c), (r, c + 1), (r, c + 2)}
        if line_h.issubset(comp):
            remainder = comp - line_h
            if can_tile_2x2(remainder):
                return line_h
        line_v = {(r, c), (r + 1, c), (r + 2, c)}
        if line_v.issubset(comp):
            remainder = comp - line_v
            if can_tile_2x2(remainder):
                return line_v
    return None

def transform(grid):
    rows, cols = len(grid), len(grid[0])
    bg = Counter(v for row in grid for v in row).most_common(1)[0][0]
    fg_colors = set(v for row in grid for v in row) - {bg}
    fg = list(fg_colors)[0]

    result = [row[:] for row in grid]
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg:
                result[r][c] = bg

    components = get_components(grid, fg)
    for comp in components:
        if len(comp) % 4 == 0:
            for r, c in comp:
                result[r][c] = 9
        elif len(comp) == 3:
            for r, c in comp:
                result[r][c] = 3
        else:
            line = find_line_to_remove(comp)
            if line:
                for r, c in comp:
                    result[r][c] = 3 if (r, c) in line else 9
            else:
                for r, c in comp:
                    result[r][c] = 9
    return result
