def solve(grid):
    """Sort column-pairs by descending notch count in a repeating tile pattern."""
    rows = len(grid)
    cols = len(grid[0])

    # Find the non-zero color
    color = None
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                color = grid[r][c]
                break
        if color is not None:
            break

    # Identify column pairs: columns NOT always 0 (separator cols are always 0)
    sep_cols = set()
    for c in range(cols):
        if all(grid[r][c] == 0 for r in range(rows)):
            sep_cols.add(c)

    # Group non-separator columns into pairs
    col_pairs = []
    pair = []
    for c in range(cols):
        if c in sep_cols:
            if pair:
                col_pairs.append(tuple(pair))
                pair = []
        else:
            pair.append(c)
    if pair:
        col_pairs.append(tuple(pair))

    # Identify odd rows (rows with notch variation) vs even rows (all full)
    odd_rows = []
    even_rows = []
    for r in range(rows):
        is_full = all(grid[r][c] == color for c in range(cols) if c not in sep_cols)
        if is_full:
            even_rows.append(r)
        else:
            odd_rows.append(r)

    # Build notch grid: notch[row_idx][col_idx] = 1 if notched, 0 if full
    notch = []
    for r in odd_rows:
        row_notch = []
        for pair in col_pairs:
            # Check if last col in pair is 0 (notch) or color (full)
            is_notch = grid[r][pair[-1]] == 0
            row_notch.append(1 if is_notch else 0)
        notch.append(row_notch)

    num_cols = len(col_pairs)
    # Column notch counts
    col_counts = []
    for j in range(num_cols):
        count = sum(notch[i][j] for i in range(len(odd_rows)))
        col_counts.append(count)

    # Sort columns by descending notch count (stable sort preserves original order for ties)
    sorted_indices = sorted(range(num_cols), key=lambda j: col_counts[j], reverse=True)

    # Build new notch grid with sorted columns
    new_notch = []
    for i in range(len(odd_rows)):
        new_notch.append([notch[i][sorted_indices[j]] for j in range(num_cols)])

    # Reconstruct the output grid
    out = [row[:] for row in grid]
    for idx, r in enumerate(odd_rows):
        for j, pair in enumerate(col_pairs):
            if new_notch[idx][j] == 1:
                # Notch: first col = color, last col = 0
                out[r][pair[0]] = color
                if len(pair) > 1:
                    out[r][pair[-1]] = 0
            else:
                # Full: all cols = color
                for c in pair:
                    out[r][c] = color

    return out


if __name__ == "__main__":
    import sys, json
    with open(sys.argv[1]) as f:
        task = json.load(f)
    for i, ex in enumerate(task['train'] + task['test']):
        result = solve(ex['input'])
        status = "PASS" if result == ex['output'] else "FAIL"
        print(f"{'Train' if i < len(task['train']) else 'Test'} {i if i < len(task['train']) else i - len(task['train'])}: {status}")
