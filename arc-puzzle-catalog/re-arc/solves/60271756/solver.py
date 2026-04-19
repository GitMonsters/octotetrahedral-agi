def transform(grid: list[list[int]]) -> list[list[int]]:
    R = len(grid)
    C = len(grid[0])
    out = [[0]*C for _ in range(R)]
    
    for r in range(R):
        for c in range(C):
            dist = min(r, c, R-1-r, C-1-c)
            ring = dist
            # Spiral seam correction: bottom-left diagonal
            if r + dist >= R - 2 and R-1-r > dist and C-1-c > dist:
                ring += 1
            
            if ring % 2 == 0:
                out[r][c] = 5
            else:
                out[r][c] = grid[r][c]
    
    return out


if __name__ == "__main__":
    import json
