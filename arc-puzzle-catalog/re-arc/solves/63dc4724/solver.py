from collections import Counter, deque

def transform(grid):
    rows, cols = len(grid), len(grid[0])
    bg = Counter(v for row in grid for v in row).most_common(1)[0][0]
    
    visited = set()
    components = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg and (r,c) not in visited:
                color = grid[r][c]
                comp = set()
                q = deque([(r,c)])
                visited.add((r,c))
                while q:
                    cr,cc = q.popleft()
                    comp.add((cr,cc))
                    for dr,dc in [(0,1),(0,-1),(1,0),(-1,0)]:
                        nr,nc = cr+dr,cc+dc
                        if 0<=nr<rows and 0<=nc<cols and (nr,nc) not in visited and grid[nr][nc]==color:
                            visited.add((nr,nc))
                            q.append((nr,nc))
                components.append((color, comp))
    
    valid_cells = set()
    for color, comp in components:
        row_count = Counter(r for r,c in comp)
        col_count = Counter(c for r,c in comp)
        for (r,c) in comp:
            cells = [(r,c),(r,c+1),(r+1,c),(r+1,c+1)]
            if all((cr,cc) in comp for cr,cc in cells):
                row_check = (row_count[r] == 2 and row_count[r+1] == 2)
                col_check = (col_count[c] == 2 and col_count[c+1] == 2)
                if row_check or col_check:
                    for cell in cells:
                        valid_cells.add(cell)
    
    result = [[bg]*cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg:
                result[r][c] = 0 if (r,c) in valid_cells else 8
    return result
