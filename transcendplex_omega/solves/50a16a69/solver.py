def solve(grid):
    """Repeating tiled pattern with border color. Extend pattern with +1 cycle shift."""
    R, C = len(grid), len(grid[0])
    border_color = grid[-1][-1]

    # Find body dimensions (top-left rectangle of non-border values)
    body_h = 0
    for r in range(R):
        if grid[r][0] == border_color:
            break
        body_h += 1

    body_w = 0
    for c in range(C):
        if grid[0][c] == border_color:
            break
        body_w += 1

    # Extract column cycle from first body row
    first_row = [grid[0][c] for c in range(body_w)]
    period = 1
    for p in range(1, body_w + 1):
        if all(first_row[i] == first_row[i % p] for i in range(body_w)):
            period = p
            break
    cycle = first_row[:period]

    # Find row shifts: body[r][c] = cycle[(c + shift) % period]
    row_shifts = []
    for r in range(body_h):
        for s in range(period):
            if all(grid[r][c] == cycle[(c + s) % period] for c in range(body_w)):
                row_shifts.append(s)
                break

    # Find row period
    row_period = 1
    for p in range(1, body_h + 1):
        if all(row_shifts[i] == row_shifts[i % p] for i in range(body_h)):
            row_period = p
            break

    # Generate output with +1 shift
    out = []
    for r in range(R):
        s = row_shifts[r % row_period] + 1
        out.append([cycle[(c + s) % period] for c in range(C)])
    return out


if __name__ == "__main__":
    import sys, json
    with open(sys.argv[1]) as f:
        task = json.load(f)
    for i, ex in enumerate(task['train'] + task['test']):
        result = solve(ex['input'])
        status = "PASS" if result == ex['output'] else "FAIL"
        print(f"{'Train' if i < len(task['train']) else 'Test'} {i if i < len(task['train']) else i - len(task['train'])}: {status}")
