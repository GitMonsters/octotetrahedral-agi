def solve(grid):
    """4s drop 1 row downward. If landing at floor_row - 1, bounce diagonally upward."""
    rows = len(grid)
    cols = len(grid[0])
    out = [[0] * cols for _ in range(rows)]

    # Find floor row (row of all 2s)
    floor_row = None
    for r in range(rows):
        if all(grid[r][c] == 2 for c in range(cols)):
            floor_row = r
            break

    # Copy floor
    for c in range(cols):
        out[floor_row][c] = 2

    # Find all 4 positions
    fours = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 4]

    for r, c in fours:
        new_r = r + 1  # drop 1 row down

        if new_r >= rows or new_r == floor_row:
            continue

        out[new_r][c] = 4

        # Check if at floor_row - 1 (above floor, bounce)
        if new_r == floor_row - 1:
            # Bounce diagonally up-left
            for i in range(1, rows):
                nr, nc = new_r - i, c - i
                if 0 <= nr < rows and 0 <= nc < cols:
                    out[nr][nc] = 4
                else:
                    break
            # Bounce diagonally up-right
            for i in range(1, rows):
                nr, nc = new_r - i, c + i
                if 0 <= nr < rows and 0 <= nc < cols:
                    out[nr][nc] = 4
                else:
                    break

    return out


if __name__ == "__main__":
    import sys, json
    with open(sys.argv[1]) as f:
        task = json.load(f)
    for i, ex in enumerate(task['train'] + task['test']):
        result = solve(ex['input'])
        status = "PASS" if result == ex['output'] else "FAIL"
        print(f"{'Train' if i < len(task['train']) else 'Test'} {i if i < len(task['train']) else i - len(task['train'])}: {status}")
