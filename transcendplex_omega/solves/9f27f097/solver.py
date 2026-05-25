def solve(grid):
    """Find colored pattern and 0-filled rectangle. Copy pattern horizontally flipped into 0-rect."""
    rows = len(grid)
    cols = len(grid[0])
    result = [row[:] for row in grid]

    # Find background color (most common)
    from collections import Counter
    counts = Counter()
    for row in grid:
        for cell in row:
            counts[cell] += 1
    bg = counts.most_common(1)[0][0]

    # Find the 0-filled rectangle
    zero_cells = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 0]
    if not zero_cells:
        return result
    zr_min = min(r for r, c in zero_cells)
    zr_max = max(r for r, c in zero_cells)
    zc_min = min(c for r, c in zero_cells)
    zc_max = max(c for r, c in zero_cells)

    # Find the pattern (non-bg, non-0 region)
    pattern_cells = [(r, c) for r in range(rows) for c in range(cols)
                     if grid[r][c] != bg and grid[r][c] != 0]
    # Include bg cells that are within the pattern's bounding box
    pr_min = min(r for r, c in pattern_cells)
    pr_max = max(r for r, c in pattern_cells)
    pc_min = min(c for r, c in pattern_cells)
    pc_max = max(c for r, c in pattern_cells)

    # Extract pattern
    pat_h = pr_max - pr_min + 1
    pat_w = pc_max - pc_min + 1
    pattern = []
    for r in range(pr_min, pr_max + 1):
        row = []
        for c in range(pc_min, pc_max + 1):
            row.append(grid[r][c])
        pattern.append(row)

    # Horizontally flip the pattern
    flipped = [row[::-1] for row in pattern]

    # Place into 0-rectangle
    for r in range(zr_min, zr_max + 1):
        for c in range(zc_min, zc_max + 1):
            pr = r - zr_min
            pc = c - zc_min
            if pr < len(flipped) and pc < len(flipped[0]):
                result[r][c] = flipped[pr][pc]

    return result


if __name__ == "__main__":
    import sys, json
    with open(sys.argv[1]) as f:
        task = json.load(f)
    for i, ex in enumerate(task['train'] + task['test']):
        result = solve(ex['input'])
        status = "PASS" if result == ex['output'] else "FAIL"
        print(f"{'Train' if i < len(task['train']) else 'Test'} {i if i < len(task['train']) else i - len(task['train'])}: {status}")
