def transform(grid):
    """
    Transform for RE-ARC task 743d2f1c.
    
    Pattern:
    1. Scale input 3x
    2. Find isolated non-5 cells (single-cell colored regions)
       - Draw diagonals through adjacent 5-cells toward opposite corners
       - Only if endpoint is a different color
    3. Find non-5 cells at corners where their region meets a different non-5 region
       diagonally, with a 5-region in the opposite diagonal direction
       - Draw diagonal through that cell if the 5-neighbor is at corner of 5-region
    """
    R, C = len(grid), len(grid[0])
    
    # Scale 3x
    out = [[0] * (C * 3) for _ in range(R * 3)]
    for r in range(R):
        for c in range(C):
            for dr in range(3):
                for dc in range(3):
                    out[r*3+dr][c*3+dc] = grid[r][c]
    
    def count_5_neighbors(r, c):
        """Count orthogonal neighbors with color 5"""
        count = 0
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < R and 0 <= nc < C and grid[nr][nc] == 5:
                count += 1
        return count
    
    def draw_diagonal(r, c, is_main):
        """Draw a diagonal line of 1s through cell (r,c)"""
        if is_main:  # main diagonal (top-left to bottom-right)
            for i in range(3):
                out[r*3+i][c*3+i] = 1
        else:  # anti diagonal (top-right to bottom-left)
            for i in range(3):
                out[r*3+i][c*3+(2-i)] = 1
    
    # Part 1: Handle isolated non-5 cells
    for r in range(R):
        for c in range(C):
            if grid[r][c] == 5:
                continue
            color = grid[r][c]
            
            # Check if isolated (no same-color orthogonal neighbors)
            same_neighbors = 0
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < R and 0 <= nc < C and grid[nr][nc] == color:
                    same_neighbors += 1
            
            if same_neighbors > 0:
                continue  # Not isolated
            
            # Draw diagonals through adjacent 5-cells
            for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < R and 0 <= nc < C):
                    continue
                if grid[nr][nc] != 5:
                    continue
                
                # Trace through 5-cells in this diagonal direction
                path = []
                while 0 <= nr < R and 0 <= nc < C and grid[nr][nc] == 5:
                    path.append((nr, nc))
                    nr += dr
                    nc += dc
                
                # Check what's at the endpoint
                if 0 <= nr < R and 0 <= nc < C:
                    end_color = grid[nr][nc]
                else:
                    end_color = None  # boundary
                
                # Draw only if endpoint is different from isolated cell's color
                if end_color != color:
                    is_main = (dr == dc)
                    for pr, pc in path:
                        draw_diagonal(pr, pc, is_main)
    
    # Part 2: Handle corner cells of non-5 regions
    for r in range(R):
        for c in range(C):
            color = grid[r][c]
            if color == 5:
                continue
            
            # Check each diagonal direction
            for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                diag_r, diag_c = r + dr, c + dc
                opp_r, opp_c = r - dr, c - dc
                
                if not (0 <= diag_r < R and 0 <= diag_c < C):
                    continue
                if not (0 <= opp_r < R and 0 <= opp_c < C):
                    continue
                
                diag_color = grid[diag_r][diag_c]
                opp_color = grid[opp_r][opp_c]
                
                # Condition: diagonal neighbor is different non-5 color
                # and opposite diagonal neighbor is 5
                if diag_color == 5 or diag_color == color:
                    continue
                if opp_color != 5:
                    continue
                
                # Additional check: the 5-cell must be at a corner of the 5-region
                # (i.e., it has at most 1 orthogonal 5-neighbor)
                if count_5_neighbors(opp_r, opp_c) > 1:
                    continue
                
                # Draw diagonal through this cell
                is_main = (dr == dc)
                draw_diagonal(r, c, is_main)
    
    return out
