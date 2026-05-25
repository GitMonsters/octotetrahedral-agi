def solve(grid: list[list[int]]) -> list[list[int]]:
    """Grid divided into sub-blocks by zero borders. Invert each sub-block: 8->0, 0->2."""
    rows = len(grid)
    cols = len(grid[0])
    output = [[0]*cols for _ in range(rows)]

    # Find separator rows and cols (all zeros)
    sep_rows = [r for r in range(rows) if all(grid[r][c] == 0 for c in range(cols))]
    sep_cols = [c for c in range(cols) if all(grid[r][c] == 0 for r in range(rows))]

    for i in range(len(sep_rows) - 1):
        for j in range(len(sep_cols) - 1):
            r_start = sep_rows[i] + 1
            r_end = sep_rows[i + 1]
            c_start = sep_cols[j] + 1
            c_end = sep_cols[j + 1]
            for r in range(r_start, r_end):
                for c in range(c_start, c_end):
                    output[r][c] = 0 if grid[r][c] == 8 else 2

    return output


if __name__ == "__main__":
    import sys, json
    with open(sys.argv[1]) as f:
        task = json.load(f)
    for i, ex in enumerate(task['train'] + task['test']):
        result = solve(ex['input'])
        status = "PASS" if result == ex['output'] else "FAIL"
        print(f"{'Train' if i < len(task['train']) else 'Test'} {i if i < len(task['train']) else i - len(task['train'])}: {status}")
