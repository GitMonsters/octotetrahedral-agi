def solve(grid):
    rows = len(grid)
    cols = len(grid[0])
    block = 5
    out_rows = rows // block
    out_cols = cols // block

    # Determine background color (most common)
    counts = {}
    for r in range(rows):
        for c in range(cols):
            v = grid[r][c]
            counts[v] = counts.get(v, 0) + 1
    bg = max(counts, key=counts.get)

    out = []
    for br in range(out_rows):
        row = []
        for bc in range(out_cols):
            has_non_bg = False
            for r in range(br * block, (br + 1) * block):
                for c in range(bc * block, (bc + 1) * block):
                    if grid[r][c] != bg:
                        has_non_bg = True
                        break
                if has_non_bg:
                    break
            row.append(0 if has_non_bg else bg)
        out.append(row)
    return out


if __name__ == "__main__":
    import sys, json
    with open(sys.argv[1]) as f:
        task = json.load(f)
    for i, ex in enumerate(task['train'] + task['test']):
        result = solve(ex['input'])
        status = "PASS" if result == ex['output'] else "FAIL"
        print(f"{'Train' if i < len(task['train']) else 'Test'} {i if i < len(task['train']) else i - len(task['train'])}: {status}")
