import json, sys
from collections import Counter, deque

def solve(grid):
    rows, cols = len(grid), len(grid[0])
    g = [row[:] for row in grid]
    
    # bg = most common color on the border
    border = [grid[0][c] for c in range(cols)] + [grid[rows-1][c] for c in range(cols)]
    border += [grid[r][0] for r in range(rows)] + [grid[r][cols-1] for r in range(rows)]
    bg = Counter(border).most_common(1)[0][0]
    
    black_cells = [(r, c) for r in range(rows) for c in range(cols) if g[r][c] == 0]
    red_cells = [(r, c) for r in range(rows) for c in range(cols) if g[r][c] == 2]
    
    min_r = min(r for r, c in black_cells)
    max_r = max(r for r, c in black_cells)
    N = max_r - min_r + 1
    min_c = min(c for r, c in black_cells)
    start = (min_r, min_c)
    
    path_counts = Counter()
    for r in range(rows):
        for c in range(cols):
            v = g[r][c]
            if v != bg and v != 0 and v != 2:
                path_counts[v] += 1
    path_color = path_counts.most_common(1)[0][0]
    
    def is_valid(r, c):
        if r < 0 or r + N > rows or c < 0 or c + N > cols:
            return False
        for dr in range(N):
            for dc in range(N):
                if g[r + dr][c + dc] == bg:
                    return False
        return True
    
    visited = {start: 0}
    queue = deque([(start, 0)])
    max_dist = 0
    
    while queue:
        (r, c), d = queue.popleft()
        if d > max_dist:
            max_dist = d
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r + dr, c + dc
            if (nr, nc) not in visited and is_valid(nr, nc):
                visited[(nr, nc)] = d + 1
                queue.append(((nr, nc), d + 1))
    
    candidates = [pos for pos, d in visited.items() if d == max_dist]
    if red_cells:
        red_tl = (min(r for r, c in red_cells), min(c for r, c in red_cells))
        dest = min(candidates, key=lambda p: abs(p[0]-red_tl[0]) + abs(p[1]-red_tl[1]))
    else:
        dest = candidates[0]
    
    for r, c in black_cells:
        g[r][c] = path_color
    for dr in range(N):
        for dc in range(N):
            g[dest[0]+dr][dest[1]+dc] = 0
    
    return g

if __name__ == '__main__':
    with open(sys.argv[1]) as f:
        task = json.load(f)
    for i, ex in enumerate(task['train']):
        result = solve(ex['input'])
        ok = result == ex['output']
        print(f"Train {i}: {'PASS' if ok else 'FAIL'}")
        if not ok:
            for r in range(len(result)):
                for c in range(len(result[0])):
                    if result[r][c] != ex['output'][r][c]:
                        print(f"  ({r},{c}): got {result[r][c]}, expected {ex['output'][r][c]}")
