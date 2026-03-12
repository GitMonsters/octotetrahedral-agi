def solve(grid: list[list[int]]) -> list[list[int]]:
    """Each 4-shape projects a beam through the 8-line to the grid edge.
    Original 4s become 3, gap to line fills with 4, past line fills with 8,
    and existing 2-cells in the beam path get pushed to the far edge."""
    H = len(grid)
    W = len(grid[0])
    result = [row[:] for row in grid]

    # Find the 8-line (full row or full column of 8s)
    line_type = None
    line_pos = None
    for r in range(H):
        if all(grid[r][c] == 8 for c in range(W)):
            line_type = 'h'
            line_pos = r
            break
    if line_type is None:
        for c in range(W):
            if all(grid[r][c] == 8 for r in range(H)):
                line_type = 'v'
                line_pos = c
                break

    if line_type == 'h':
        # Determine which side has 4s
        fours = [(r, c) for r in range(H) for c in range(W) if grid[r][c] == 4]
        four_above = any(r < line_pos for r, _ in fours)

        for c in range(W):
            four_rows = sorted(r for r in range(H) if grid[r][c] == 4)
            if not four_rows:
                continue

            # Change original 4s to 3
            for r in four_rows:
                result[r][c] = 3

            if four_above:
                closest = max(four_rows)
                # Fill gap between shape and line with 4
                for r in range(closest + 1, line_pos):
                    result[r][c] = 4
                # Count 2s on the far side
                two_count = sum(1 for r in range(line_pos + 1, H) if grid[r][c] == 2)
                # Fill far side with 8
                for r in range(line_pos + 1, H):
                    result[r][c] = 8
                # Push 2s to far edge
                for r in range(H - two_count, H):
                    result[r][c] = 2
            else:
                closest = min(four_rows)
                for r in range(line_pos + 1, closest):
                    result[r][c] = 4
                two_count = sum(1 for r in range(0, line_pos) if grid[r][c] == 2)
                for r in range(0, line_pos):
                    result[r][c] = 8
                for r in range(0, two_count):
                    result[r][c] = 2

    elif line_type == 'v':
        fours = [(r, c) for r in range(H) for c in range(W) if grid[r][c] == 4]
        four_left = any(c < line_pos for _, c in fours)

        for r in range(H):
            four_cols = sorted(c for c in range(W) if grid[r][c] == 4)
            if not four_cols:
                continue

            for c in four_cols:
                result[r][c] = 3

            if four_left:
                closest = max(four_cols)
                for c in range(closest + 1, line_pos):
                    result[r][c] = 4
                two_count = sum(1 for c in range(line_pos + 1, W) if grid[r][c] == 2)
                for c in range(line_pos + 1, W):
                    result[r][c] = 8
                for c in range(W - two_count, W):
                    result[r][c] = 2
            else:
                closest = min(four_cols)
                for c in range(line_pos + 1, closest):
                    result[r][c] = 4
                two_count = sum(1 for c in range(0, line_pos) if grid[r][c] == 2)
                for c in range(0, line_pos):
                    result[r][c] = 8
                for c in range(0, two_count):
                    result[r][c] = 2

    return result


if __name__ == "__main__":
    import sys, json
    with open(sys.argv[1]) as f:
        task = json.load(f)
    for i, ex in enumerate(task['train'] + task['test']):
        result = solve(ex['input'])
        expected = ex.get('output')
        if expected:
            status = "PASS" if result == expected else "FAIL"
            print(f"Example {i}: {status}")
        else:
            print(f"Example {i}: no expected output")
