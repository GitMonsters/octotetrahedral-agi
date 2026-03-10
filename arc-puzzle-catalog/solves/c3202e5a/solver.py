def solve(grid):
    """Grid divided by separator lines. Return section with fewest distinct non-zero colors."""
    rows = len(grid)
    cols = len(grid[0])

    # Find separator value: the value that forms complete rows
    from collections import Counter
    sep_val = None
    for r in range(rows):
        if len(set(grid[r])) == 1 and grid[r][0] != 0:
            sep_val = grid[r][0]
            break

    # Find separator rows and cols
    sep_rows = [r for r in range(rows) if all(grid[r][c] == sep_val for c in range(cols))]
    sep_cols = [c for c in range(cols) if all(grid[r][c] == sep_val for r in range(rows))]

    def ranges_from_seps(seps, total):
        boundaries = [-1] + seps + [total]
        result = []
        for i in range(len(boundaries) - 1):
            start = boundaries[i] + 1
            end = boundaries[i + 1]
            if start < end:
                result.append((start, end))
        return result

    row_ranges = ranges_from_seps(sep_rows, rows)
    col_ranges = ranges_from_seps(sep_cols, cols)

    best_section = None
    best_count = float('inf')

    for rs, re in row_ranges:
        for cs, ce in col_ranges:
            colors = set()
            for r in range(rs, re):
                for c in range(cs, ce):
                    if grid[r][c] != 0:
                        colors.add(grid[r][c])
            count = len(colors)
            if 0 < count < best_count:
                best_count = count
                best_section = [[grid[r][c] for c in range(cs, ce)] for r in range(rs, re)]

    return best_section


if __name__ == "__main__":
    import sys, json
    with open(sys.argv[1]) as f:
        task = json.load(f)
    for i, ex in enumerate(task['train'] + task['test']):
        result = solve(ex['input'])
        status = "PASS" if result == ex['output'] else "FAIL"
        print(f"{'Train' if i < len(task['train']) else 'Test'} {i if i < len(task['train']) else i - len(task['train'])}: {status}")
