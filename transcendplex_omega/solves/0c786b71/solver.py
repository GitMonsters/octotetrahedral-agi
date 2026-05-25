def solve(grid: list[list[int]]) -> list[list[int]]:
    """2x2 arrangement: top-left=rot180, top-right=flip_ud, bottom-left=flip_lr, bottom-right=identity."""
    r = len(grid)
    c = len(grid[0])

    rot180 = [row[::-1] for row in grid[::-1]]
    flip_ud = grid[::-1]
    flip_lr = [row[::-1] for row in grid]
    identity = grid

    out = []
    for i in range(r):
        out.append(rot180[i] + flip_ud[i])
    for i in range(r):
        out.append(flip_lr[i] + identity[i])
    return out


if __name__ == "__main__":
    import sys, json
    with open(sys.argv[1]) as f:
        task = json.load(f)
    for i, ex in enumerate(task['train'] + task['test']):
        result = solve(ex['input'])
        status = "PASS" if result == ex['output'] else "FAIL"
        print(f"{'Train' if i < len(task['train']) else 'Test'} {i if i < len(task['train']) else i - len(task['train'])}: {status}")
