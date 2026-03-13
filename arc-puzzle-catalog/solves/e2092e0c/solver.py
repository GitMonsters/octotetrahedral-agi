def solve(grid: list[list[int]]) -> list[list[int]]:
    """Find the 3x3 pattern inside the L-shaped 5-border at top-left,
    locate its duplicate elsewhere in the grid, and draw a 5x5
    rectangular border of 5s around it."""
    import copy
    result = copy.deepcopy(grid)
    rows, cols = len(grid), len(grid[0])

    # Extract the 3x3 key from the L interior (rows 0-2, cols 0-2)
    key = [grid[r][c] for r in range(3) for c in range(3)]

    # Search for the matching 3x3 block elsewhere
    for r in range(rows - 2):
        for c in range(cols - 2):
            if r < 3 and c < 3:
                continue  # skip the L region itself
            block = [grid[r + dr][c + dc] for dr in range(3) for dc in range(3)]
            if block == key:
                # Draw 5x5 rectangle border around (r-1, c-1) to (r+3, c+3)
                tr, lc = r - 1, c - 1
                br, rc = r + 3, c + 3
                for cc in range(lc, rc + 1):
                    result[tr][cc] = 5
                    result[br][cc] = 5
                for rr in range(tr, br + 1):
                    result[rr][lc] = 5
                    result[rr][rc] = 5
                return result

    return result
