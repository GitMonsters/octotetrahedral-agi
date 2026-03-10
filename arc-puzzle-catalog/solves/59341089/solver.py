def solve(grid):
    """Reverse each row, concat with original, tile 2x horizontally."""
    R = len(grid)
    half = []
    for r in range(R):
        half.append(grid[r][::-1] + grid[r])
    # Tile the 3x6 half twice to get 3x12
    return [row * 2 for row in half]


if __name__ == "__main__":
    import sys, json
    with open(sys.argv[1]) as f:
        task = json.load(f)
    for i, ex in enumerate(task['train'] + task['test']):
        result = solve(ex['input'])
        status = "PASS" if result == ex['output'] else "FAIL"
        print(f"{'Train' if i < len(task['train']) else 'Test'} {i if i < len(task['train']) else i - len(task['train'])}: {status}")
