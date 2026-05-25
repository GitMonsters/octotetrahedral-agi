def solve(grid):
    # Grid divided by lines of 4s into rectangular sections.
    # Each section pair has a "key" block (single colored cell in 3x3 of 0s) and a "pattern" block (1s).
    # Replace 1s in pattern blocks with the corresponding key color.
    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]

    # Find divider rows (all 4s) and divider cols (all 4s)
    div_rows = [r for r in range(rows) if all(grid[r][c] == 4 for c in range(cols))]
    div_cols = [c for c in range(cols) if all(grid[r][c] == 4 for r in range(rows))]

    # Determine section boundaries
    row_bounds = []
    prev = 0
    for dr in div_rows:
        if dr > prev:
            row_bounds.append((prev, dr))
        prev = dr + 1
    if prev < rows:
        row_bounds.append((prev, rows))

    col_bounds = []
    prev = 0
    for dc in div_cols:
        if dc > prev:
            col_bounds.append((prev, dc))
        prev = dc + 1
    if prev < cols:
        col_bounds.append((prev, cols))

    # For each row-band, find key colors and pattern blocks
    for rb_start, rb_end in row_bounds:
        sections = []
        for cb_start, cb_end in col_bounds:
            # Analyze this section
            key_color = None
            has_ones = False
            for r in range(rb_start, rb_end):
                for c in range(cb_start, cb_end):
                    v = grid[r][c]
                    if v == 1:
                        has_ones = True
                    elif v != 0 and v != 4:
                        key_color = v
            sections.append((cb_start, cb_end, key_color, has_ones))

        # Match key sections with pattern sections
        # Find the key color for this row-band
        key = None
        for _, _, kc, _ in sections:
            if kc is not None:
                key = kc
                break

        if key is not None:
            # Replace 1s in pattern sections with key color
            for cb_start, cb_end, kc, has_ones in sections:
                if has_ones:
                    for r in range(rb_start, rb_end):
                        for c in range(cb_start, cb_end):
                            if grid[r][c] == 1:
                                result[r][c] = key

    # Also handle column-band organization
    for cb_start, cb_end in col_bounds:
        sections = []
        for rb_start, rb_end in row_bounds:
            key_color = None
            has_ones = False
            for r in range(rb_start, rb_end):
                for c in range(cb_start, cb_end):
                    v = grid[r][c]
                    if v == 1:
                        has_ones = True
                    elif v != 0 and v != 4:
                        key_color = v
            sections.append((rb_start, rb_end, key_color, has_ones))

        key = None
        for _, _, kc, _ in sections:
            if kc is not None:
                key = kc
                break

        # This is already handled if each row-band has its own key
        # But let me handle column-wise: find key per column band
        # Actually, let me re-approach: pair each pattern section with its key

    # Better approach: each row-band × col-band cell is a section.
    # Sections with a single non-zero non-4 non-1 value are "key" sections.
    # Sections with 1s are "pattern" sections.
    # Match keys to patterns within the same row-band OR same col-band.

    result = [row[:] for row in grid]

    section_info = {}
    for ri, (rb_start, rb_end) in enumerate(row_bounds):
        for ci, (cb_start, cb_end) in enumerate(col_bounds):
            key_color = None
            has_ones = False
            for r in range(rb_start, rb_end):
                for c in range(cb_start, cb_end):
                    v = grid[r][c]
                    if v == 1:
                        has_ones = True
                    elif v != 0 and v != 4:
                        key_color = v
            section_info[(ri, ci)] = (rb_start, rb_end, cb_start, cb_end, key_color, has_ones)

    # For each row-band, find key-pattern pairs
    for ri, (rb_start, rb_end) in enumerate(row_bounds):
        # Get all sections in this row-band
        row_sections = [(ci, section_info[(ri, ci)]) for ci in range(len(col_bounds))]
        # Find key section
        key_section = None
        for ci, info in row_sections:
            if info[4] is not None:  # has key color
                key_section = info
                break
        if key_section is not None:
            key_color = key_section[4]
            # Apply to all pattern sections in same row-band
            for ci, info in row_sections:
                if info[5]:  # has ones
                    rb_s, rb_e, cb_s, cb_e = info[:4]
                    for r in range(rb_s, rb_e):
                        for c in range(cb_s, cb_e):
                            if grid[r][c] == 1:
                                result[r][c] = key_color

    # For each col-band, find key-pattern pairs
    for ci, (cb_start, cb_end) in enumerate(col_bounds):
        col_sections = [(ri, section_info[(ri, ci)]) for ri in range(len(row_bounds))]
        key_section = None
        for ri, info in col_sections:
            if info[4] is not None:
                key_section = info
                break
        if key_section is not None:
            key_color = key_section[4]
            for ri, info in col_sections:
                if info[5]:
                    rb_s, rb_e, cb_s, cb_e = info[:4]
                    for r in range(rb_s, rb_e):
                        for c in range(cb_s, cb_e):
                            if grid[r][c] == 1:
                                result[r][c] = key_color

    return result

if __name__ == "__main__":
    import sys, json
    with open(sys.argv[1]) as f:
        task = json.load(f)
    for i, ex in enumerate(task['train'] + task['test']):
        result = solve(ex['input'])
        status = "PASS" if result == ex['output'] else "FAIL"
        print(f"{'Train' if i < len(task['train']) else 'Test'} {i if i < len(task['train']) else i - len(task['train'])}: {status}")
