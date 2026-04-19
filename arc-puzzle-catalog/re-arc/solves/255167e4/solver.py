from collections import Counter

def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    output = [row[:] for row in grid]
    
    bg = Counter(grid[r][c] for r in range(rows) for c in range(cols)).most_common(1)[0][0]
    
    visited = set()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == bg or (r,c) in visited:
                continue
            color = grid[r][c]
            comp = []
            stack = [(r,c)]
            visited.add((r,c))
            while stack:
                cr,cc = stack.pop()
                comp.append((cr,cc))
                for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr,nc = cr+dr,cc+dc
                    if 0<=nr<rows and 0<=nc<cols and (nr,nc) not in visited and grid[nr][nc]==color:
                        visited.add((nr,nc))
                        stack.append((nr,nc))
            
            comp_set = set(comp)
            
            # Find endpoints and junctions
            endpoints = []
            for pr, pc in comp:
                nbrs = sum(1 for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)] if (pr+dr,pc+dc) in comp_set)
                if nbrs == 1:
                    endpoints.append((pr,pc))
            
            n_endpoints = len(endpoints)
            
            if n_endpoints == 3:
                # T-junction → color 5
                out_color = 5
            elif n_endpoints == 2:
                # Trace path and count turns
                turns = 0
                prev = None
                curr = endpoints[0]
                direction = None
                for _ in range(len(comp) + 1):
                    nbrs = [(curr[0]+dr,curr[1]+dc) for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]
                           if (curr[0]+dr,curr[1]+dc) in comp_set and (curr[0]+dr,curr[1]+dc) != prev]
                    if not nbrs:
                        break
                    nxt = nbrs[0]
                    new_dir = (nxt[0]-curr[0], nxt[1]-curr[1])
                    if direction is not None and new_dir != direction:
                        turns += 1
                    direction = new_dir
                    prev = curr
                    curr = nxt
                    if curr == endpoints[1]:
                        break
                
                if turns >= 2:
                    out_color = 4  # U-shape
                else:
                    out_color = 1  # L-shape
            else:
                out_color = bg  # fallback
            
            for pr, pc in comp:
                output[pr][pc] = out_color
    
    return output

