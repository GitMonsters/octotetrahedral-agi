def solve(grid):
    """Grid divided by 0-separators into sub-blocks. Output majority color per block with 0 border."""
    rows = len(grid)
    cols = len(grid[0])

    # Find separator rows and cols (all zeros)
    sep_rows = [r for r in range(rows) if all(grid[r][c] == 0 for c in range(cols))]
    sep_cols = [c for c in range(cols) if all(grid[r][c] == 0 for r in range(rows))]

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

    from collections import Counter
    block_grid = []
    for rs, re in row_ranges:
        block_row = []
        for cs, ce in col_ranges:
            counts = Counter()
            for r in range(rs, re):
                for c in range(cs, ce):
                    counts[grid[r][c]] += 1
            # Majority color (most common)
            majority = counts.most_common(1)[0][0]
            block_row.append(majority)
        block_grid.append(block_row)

    # Build output with 0 border
    brows = len(block_grid)
    bcols = len(block_grid[0])
    out_rows = brows + 2
    out_cols = bcols + 2
    result = [[0] * out_cols for _ in range(out_rows)]
    for r in range(brows):
        for c in range(bcols):
            result[r + 1][c + 1] = block_grid[r][c]
    return result


if __name__ == "__main__":
    import sys, json
    with open(sys.argv[1]) as f:
        task = json.load(f)
    for i, ex in enumerate(task['train'] + task['test']):
        result = solve(ex['input'])
        status = "PASS" if result == ex['output'] else "FAIL"
        print(f"{'Train' if i < len(task['train']) else 'Test'} {i if i < len(task['train']) else i - len(task['train'])}: {status}")
