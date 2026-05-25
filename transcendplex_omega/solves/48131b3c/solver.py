def solve(grid):
    """Invert (swap 0 and non-zero color) then tile 2x2."""
    R, C = len(grid), len(grid[0])
    # Find the non-zero color
    color = 0
    for row in grid:
        for v in row:
            if v != 0:
                color = v
                break
        if color:
            break
    # Create inverted grid
    inv = [[color if v == 0 else 0 for v in row] for row in grid]
    # Tile 2x2
    out = []
    for br in range(2):
        for r in range(R):
            row = []
            for bc in range(2):
                row.extend(inv[r])
            out.append(row)
    return out


if __name__ == "__main__":
    import sys, json
    with open(sys.argv[1]) as f:
        task = json.load(f)
    for i, ex in enumerate(task['train'] + task['test']):
        result = solve(ex['input'])
        status = "PASS" if result == ex['output'] else "FAIL"
        print(f"{'Train' if i < len(task['train']) else 'Test'} {i if i < len(task['train']) else i - len(task['train'])}: {status}")
