from collections import Counter
from itertools import product

def transform(grid):
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    
    # Find background color
    flat = [grid[r][c] for r in range(rows) for c in range(cols)]
    bg = Counter(flat).most_common(1)[0][0]
    
    # Collect non-bg cells grouped by color
    color_cells = {}
    for r in range(rows):
        for c in range(cols):
            v = grid[r][c]
            if v != bg:
                color_cells.setdefault(v, []).append((r, c))
    
    OUT_R, OUT_C = 7, 6
    
    def is_border(r, c):
        return r == 0 or r == OUT_R - 1 or c == 0 or c == OUT_C - 1
    
    def valid_displacements(cells):
        """Find all (dr, dc) that map all cells onto the border of 7x6."""
        if not cells:
            return []
        
        # For each cell, enumerate border positions it could map to
        # Then intersect constraints
        
        # Start with candidates from first cell
        r0, c0 = cells[0]
        candidates = set()
        for tr in range(OUT_R):
            for tc in range(OUT_C):
                if is_border(tr, tc):
                    dr = tr - r0
                    dc = tc - c0
                    candidates.add((dr, dc))
        
        # Filter by remaining cells
        for r, c in cells[1:]:
            valid = set()
            for dr, dc in candidates:
                nr, nc = r + dr, c + dc
                if 0 <= nr < OUT_R and 0 <= nc < OUT_C and is_border(nr, nc):
                    valid.add((dr, dc))
            candidates = valid
        
        return list(candidates)
    
    # Build candidate displacements per color
    color_list = sorted(color_cells.keys())
    candidates = {}
    for color in color_list:
        candidates[color] = valid_displacements(color_cells[color])
    
    # CSP: find assignment with no two colors occupying same output cell
    # Output grid starts as bg
    def backtrack(idx, assignment, used_cells):
        if idx == len(color_list):
            return assignment.copy()
        
        color = color_list[idx]
        for dr, dc in candidates[color]:
            # Compute output cells for this color
            out_cells = []
            conflict = False
            for r, c in color_cells[color]:
                nr, nc = r + dr, c + dc
                if (nr, nc) in used_cells and used_cells[(nr, nc)] != color:
                    conflict = True
                    break
                out_cells.append((nr, nc))
            
            if conflict:
                continue
            
            # Place
            for cell in out_cells:
                used_cells[cell] = color
            assignment[color] = (dr, dc)
            
            result = backtrack(idx + 1, assignment, used_cells)
            if result is not None:
                return result
            
            # Undo
            for cell in out_cells:
                del used_cells[cell]
            del assignment[color]
        
        return None
    
    assignment = backtrack(0, {}, {})
    
    if assignment is None:
        # Fallback: use first valid displacement per color, ignore conflicts
        assignment = {}
        for color in color_list:
            if candidates[color]:
                assignment[color] = candidates[color][0]
    
    # Build output grid
    out = [[bg] * OUT_C for _ in range(OUT_R)]
    for color, (dr, dc) in assignment.items():
        for r, c in color_cells[color]:
            nr, nc = r + dr, c + dc
            out[nr][nc] = color
    
    return out


# ---- Verification ----
train0_input = [
    [5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5],
    [5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5],
    [5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,3,5,5,5,5],
    [5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,1,5,3,5,5,5,5],
    [5,5,5,5,5,5,5,5,5,5,5,5,3,5,5,5,5,5,5,5,5,5],
    [5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5],
    [5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5],
    [5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,3,5,5,5,5,5,5],
    [5,5,5,5,5,5,5,5,5,5,5,5,5,1,5,5,5,5,1,5,5,5],
    [5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,2,5,5,5,5,2,5],
    [5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5],
    [5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5],
    [5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5],
    [5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5],
    [5,5,5,5,5,5,5,5,5,5,0,0,5,5,5,5,5,5,5,5,5,5],
    [5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,2,5,5,5,5,2,5],
    [5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5],
    [5,5,5,5,5,5,5,5,5,5,5,5,0,5,5,5,5,5,5,5,5,5],
    [5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5],
    [5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5],
    [5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5],
    [5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,4,5],
]
# Fix: row 17 col 21 = 4
train0_input[17][21] = 4
train0_input[18][20] = 4
train0_input[19][20] = 4
train0_input[21][20] = 4

train0_output = [
    [2,4,1,0,0,2],
    [4,5,5,5,5,3],
    [4,5,5,5,5,3],
    [3,5,5,5,5,0],
    [4,5,5,5,5,5],
    [1,5,5,5,5,1],
    [2,5,5,3,5,2]
]

train1_input = [
    [0]*20 for _ in range(21)
]
# Fill from problem data
cells1 = {(1,16):7,(2,12):7,(4,7):2,(5,12):7,(6,14):6,(7,12):2,(8,8):2,(8,16):6,(9,11):6,(17,8):1,(18,8):1,(19,3):1,(20,5):1}
for (r,c),v in cells1.items():
    train1_input[r][c] = v

train1_output = [
    [0,0,0,6,7,0],
    [7,0,0,0,0,0],
    [2,0,0,0,0,6],
    [6,0,0,0,0,1],
    [7,0,0,0,0,1],
    [1,0,0,0,0,2],
    [0,2,1,0,0,0]
]

train2_input = [
    [0]*21 for _ in range(21)
]
cells2 = {(2,3):3,(3,12):6,(3,17):6,(5,2):3,(8,6):3,(9,8):5,(9,12):6,(9,17):6,(10,5):5,(15,8):5}
for (r,c),v in cells2.items():
    train2_input[r][c] = v

train2_output = [
    [6,3,0,5,0,6],
    [5,0,0,0,0,0],
    [0,0,0,0,0,0],
    [3,0,0,0,0,0],
    [0,0,0,0,0,0],
    [0,0,0,0,0,0],
    [6,0,0,5,3,6]
]

def check(name, inp, expected):
    result = transform(inp)
    if result == expected:
        print(f"{name}: PASS")
    else:
        print(f"{name}: FAIL")
        print(f"  Expected: {expected}")
        print(f"  Got:      {result}")

check("Train 0", train0_input, train0_output)
check("Train 1", train1_input, train1_output)
check("Train 2", train2_input, train2_output)
