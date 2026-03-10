def solve(grid):
    # Rows marked with 5 in leftmost col define a "template".
    # The template (with 5s replaced by 0) repeats cyclically to fill empty rows below content.
    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]

    # Find template rows (rows containing 5)
    template_rows = []
    for r in range(rows):
        if any(grid[r][c] == 5 for c in range(cols)):
            template_rows.append(r)

    # Build template: copy those rows but replace 5 with 0
    template = []
    for r in template_rows:
        t_row = [0 if grid[r][c] == 5 else grid[r][c] for c in range(cols)]
        template.append(t_row)

    # Find first empty row (all zeros)
    first_empty = None
    for r in range(rows):
        if all(grid[r][c] == 0 for c in range(cols)):
            if first_empty is None:
                first_empty = r

    if first_empty is not None and len(template) > 0:
        # Fill empty rows with template cyclically
        t_idx = 0
        for r in range(first_empty, rows):
            if all(grid[r][c] == 0 for c in range(cols)):
                result[r] = template[t_idx % len(template)][:]
                t_idx += 1

    return result

if __name__ == "__main__":
    import sys, json
    with open(sys.argv[1]) as f:
        task = json.load(f)
    for i, ex in enumerate(task['train'] + task['test']):
        result = solve(ex['input'])
        status = "PASS" if result == ex['output'] else "FAIL"
        print(f"{'Train' if i < len(task['train']) else 'Test'} {i if i < len(task['train']) else i - len(task['train'])}: {status}")
