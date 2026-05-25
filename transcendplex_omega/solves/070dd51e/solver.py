def solve(grid: list[list[int]]) -> list[list[int]]:
    """Find pairs of same-colored dots. Same-row pairs draw horizontal bars,
    same-column pairs draw vertical bars. Vertical bars take precedence at intersections."""
    rows = len(grid)
    cols = len(grid[0])
    output = [[0]*cols for _ in range(rows)]

    from collections import defaultdict
    dots: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                dots[grid[r][c]].append((r, c))

    h_lines = []
    v_lines = []

    for color, positions in dots.items():
        (r1, c1), (r2, c2) = positions
        if r1 == r2:
            h_lines.append((r1, min(c1, c2), max(c1, c2), color))
        else:
            v_lines.append((c1, min(r1, r2), max(r1, r2), color))

    for row, c1, c2, color in h_lines:
        for c in range(c1, c2 + 1):
            output[row][c] = color

    for col, r1, r2, color in v_lines:
        for r in range(r1, r2 + 1):
            output[r][col] = color

    return output


if __name__ == "__main__":
    import sys, json
    with open(sys.argv[1]) as f:
        task = json.load(f)
    for i, ex in enumerate(task['train'] + task['test']):
        result = solve(ex['input'])
        status = "PASS" if result == ex['output'] else "FAIL"
        print(f"{'Train' if i < len(task['train']) else 'Test'} {i if i < len(task['train']) else i - len(task['train'])}: {status}")
