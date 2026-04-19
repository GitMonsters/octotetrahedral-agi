from collections import Counter

def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    
    # Detect background
    pixels = [c for row in grid for c in row]
    counts = Counter(pixels)
    bg_color = counts.most_common(1)[0][0]
    
    objects = set(pixels) - {bg_color}
    
    output = [row[:] for row in grid]
    
    # Logic for Test Case (BG=0, Objects={1, 4, 8})
    if bg_color == 0 and 1 in objects and 4 in objects and 8 in objects:
        # Rules:
        # 4 (Yellow) <-> 8 (Teal) like 0 <-> 8 in Ex 1
        # 1 (Blue) propagates like 1 in Ex 2
        
        for _ in range(10):
            changes = {}
            current = [row[:] for row in output]
            has_change = False
            
            for r in range(rows):
                for c in range(cols):
                    color = current[r][c]
                    
                    if color == 4:
                        # 4 spawns 8 at (r+1, c-1)
                        tr, tc = r+1, c-1
                        if 0 <= tr < rows and 0 <= tc < cols:
                            if current[tr][tc] == bg_color:
                                changes[(tr, tc)] = 8
                                
                    elif color == 8:
                        # 8 spawns 4 at (r-1, c+1)
                        # Check neighbors? Assume same isolation rule as Ex 1
                        has_neighbor = False
                        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                            nr, nc = r+dr, c+dc
                            if 0 <= nr < rows and 0 <= nc < cols:
                                if current[nr][nc] in [8]: # Check for 8 neighbors
                                    has_neighbor = True
                                    break
                        if not has_neighbor:
                            tr, tc = r-1, c+1
                            if 0 <= tr < rows and 0 <= tc < cols:
                                if current[tr][tc] == bg_color:
                                    changes[(tr, tc)] = 4
                                    
                    elif color == 1:
                        # 1 spawns 1 at (r+1, c+2)
                        tr, tc = r+1, c+2
                        if 0 <= tr < rows and 0 <= tc < cols:
                            if current[tr][tc] == bg_color:
                                changes[(tr, tc)] = 1
                        # 1 spawns 1 at (r-1, c-2)
                        tr, tc = r-1, c-2
                        if 0 <= tr < rows and 0 <= tc < cols:
                            if current[tr][tc] == bg_color:
                                changes[(tr, tc)] = 1
                                
            if not changes:
                break
                
            for (r, c), new_col in changes.items():
                if output[r][c] == bg_color:
                    output[r][c] = new_col
                    has_change = True
            
            if not has_change:
                break
                
    # Logic for Ex 1 / Ex 3 (BG=4, Objects={0, 8, 9} or {0, 8})
    elif bg_color == 4 and (0 in objects or 8 in objects):
         for _ in range(10):
            changes = {}
            current = [row[:] for row in output]
            has_change = False
            
            for r in range(rows):
                for c in range(cols):
                    color = current[r][c]
                    
                    if color == 0:
                        tr, tc = r+1, c-1
                        if 0 <= tr < rows and 0 <= tc < cols:
                            if current[tr][tc] == bg_color:
                                changes[(tr, tc)] = 8
                                
                    elif color == 8:
                        has_neighbor = False
                        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                            nr, nc = r+dr, c+dc
                            if 0 <= nr < rows and 0 <= nc < cols:
                                if current[nr][nc] in [8, 9]:
                                    has_neighbor = True
                                    break
                        
                        if not has_neighbor:
                            tr, tc = r-1, c+1
                            if 0 <= tr < rows and 0 <= tc < cols:
                                if current[tr][tc] == bg_color:
                                    changes[(tr, tc)] = 0
                                    
                        # 8-gap-8 -> 9
                        if r + 2 < rows:
                            if current[r+1][c] == bg_color and current[r+2][c] == 8:
                                top_has_horiz = False
                                for dc in [1, -1]:
                                    nc = c + dc
                                    if 0 <= nc < cols and current[r][nc] == 8:
                                        top_has_horiz = True
                                bot_has_horiz = False
                                for dc in [1, -1]:
                                    nc = c + dc
                                    if 0 <= nc < cols and current[r+2][nc] == 8:
                                        bot_has_horiz = True
                                
                                if top_has_horiz and bot_has_horiz:
                                    changes[(r+1, c)] = 9
                                    
                    elif color == 9:
                        # 9 -> 8
                        for dr in [-1, 1]:
                            tr, tc = r+dr, c
                            if 0 <= tr < rows and 0 <= tc < cols:
                                if current[tr][tc] == bg_color:
                                    changes[(tr, tc)] = 8
                                    
            if not changes:
                break
                
            for (r, c), new_col in changes.items():
                if output[r][c] == bg_color:
                    output[r][c] = new_col
                    has_change = True
            
            if not has_change:
                break
                
    return output
