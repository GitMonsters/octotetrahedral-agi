from collections import deque

def solve(grid):
    H, W = len(grid), len(grid[0])
    bg = 8  # blue diamond background

    # Find purple marker
    purple_pos = None
    for r in range(H):
        for c in range(W):
            if grid[r][c] == 6:
                purple_pos = (r, c)
                break

    # BFS from purple through non-blue cells (4-connectivity) -> reachable region
    reachable = set()
    queue = deque([purple_pos])
    reachable.add(purple_pos)
    while queue:
        r, c = queue.popleft()
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in reachable and grid[nr][nc] != 1:
                reachable.add((nr, nc))
                queue.append((nr, nc))

    # Find blue connected components
    blue_visited = set()
    blue_components = []
    for r in range(H):
        for c in range(W):
            if grid[r][c] == 1 and (r, c) not in blue_visited:
                comp = set()
                q = deque([(r, c)])
                blue_visited.add((r, c))
                while q:
                    cr, cc = q.popleft()
                    comp.add((cr, cc))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] == 1 and (nr, nc) not in blue_visited:
                            blue_visited.add((nr, nc))
                            q.append((nr, nc))
                blue_components.append(comp)

    # Preserved blue: entire component if ANY cell is 4-adjacent to reachable
    preserved_blue = set()
    for comp in blue_components:
        touches = False
        for r, c in comp:
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                if (r+dr, c+dc) in reachable:
                    touches = True
                    break
            if touches:
                break
        if touches:
            preserved_blue |= comp

    # Find exterior: cells not in reachable, connected to outside the grid
    exterior = set()
    ext_queue = deque()
    # Add border cells that are NOT in reachable
    for r in range(H):
        for c in range(W):
            if (r == 0 or r == H-1 or c == 0 or c == W-1):
                if (r, c) not in reachable:
                    if (r, c) not in exterior:
                        exterior.add((r, c))
                        ext_queue.append((r, c))
    while ext_queue:
        r, c = ext_queue.popleft()
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in reachable and (nr, nc) not in exterior:
                exterior.add((nr, nc))
                ext_queue.append((nr, nc))

    # Build output
    output = [[bg]*W for _ in range(H)]

    # Place purple
    output[purple_pos[0]][purple_pos[1]] = 6

    # Place preserved blue (entire component if any cell touches reachable)
    for r, c in preserved_blue:
        output[r][c] = 1

    # Orange: reachable cells on grid border OR 8-adjacent to exterior cells
    for r, c in reachable:
        if (r, c) == purple_pos:
            continue
        is_orange = False
        # Check all 8 neighbors + grid boundary
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r+dr, c+dc
                if nr < 0 or nr >= H or nc < 0 or nc >= W:
                    is_orange = True  # adjacent to grid boundary
                    break
                if (nr, nc) in exterior:
                    is_orange = True
                    break
            if is_orange:
                break
        if is_orange:
            output[r][c] = 7

    return output


if __name__ == '__main__':
    import json, sys
    with open(sys.argv[1]) as f:
        task = json.load(f)
    for i, ex in enumerate(task['train']):
        result = solve(ex['input'])
        print(f"Train {i}: {'PASS' if result == ex['output'] else 'FAIL'}")
        if result != ex['output']:
            diffs = [(r,c,result[r][c],ex['output'][r][c])
                     for r in range(len(ex['output'])) for c in range(len(ex['output'][0]))
                     if result[r][c] != ex['output'][r][c]]
            print(f"  {len(diffs)} diffs: {diffs[:10]}")
