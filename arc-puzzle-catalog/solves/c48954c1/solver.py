def solve(grid):
    """3x3 input → 9x9 output. Tile 3x3 with reflections creating seamless wallpaper."""
    n = len(grid)
    # Precompute transformations
    O = [row[:] for row in grid]
    H = [row[::-1] for row in grid]               # horizontal flip
    V = [row[:] for row in reversed(grid)]         # vertical flip
    R180 = [row[::-1] for row in reversed(grid)]   # 180° rotation

    # Block layout:
    # R180  V    R180
    # H     O    H
    # R180  V    R180
    blocks = [
        [R180, V, R180],
        [H, O, H],
        [R180, V, R180],
    ]

    result = []
    for br in range(3):
        for r in range(n):
            row = []
            for bc in range(3):
                row.extend(blocks[br][bc][r])
            result.append(row)
    return result


if __name__ == "__main__":
    import sys, json
    with open(sys.argv[1]) as f:
        task = json.load(f)
    for i, ex in enumerate(task['train'] + task['test']):
        result = solve(ex['input'])
        status = "PASS" if result == ex['output'] else "FAIL"
        print(f"{'Train' if i < len(task['train']) else 'Test'} {i if i < len(task['train']) else i - len(task['train'])}: {status}")
