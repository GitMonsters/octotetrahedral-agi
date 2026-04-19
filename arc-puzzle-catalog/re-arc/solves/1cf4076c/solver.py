def transform(input_grid):
    H = len(input_grid)
    W = len(input_grid[0])

    from collections import defaultdict

    # Find uniform rows per color and colors in non-uniform rows
    uniform_count = defaultdict(int)
    uniform_rows_by_color = defaultdict(list)
    non_uniform_colors = set()
    for r in range(H):
        row = input_grid[r]
        if len(set(row)) == 1:
            uniform_count[row[0]] += 1
            uniform_rows_by_color[row[0]].append(r)
        else:
            for c in row:
                non_uniform_colors.add(c)

    # Separator color: fewest uniform rows among colors that also appear in non-uniform rows
    candidates = {c: uniform_count[c] for c in uniform_count if c in non_uniform_colors}
    if not candidates:
        candidates = dict(uniform_count)
    sep_color = min(candidates, key=candidates.get)

    # Separator rows are uniform rows of the separator color
    sep_rows = sorted(uniform_rows_by_color[sep_color])

    # Extract blocks between separator rows
    boundaries = [-1] + sep_rows + [H]
    blocks = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i] + 1
        end = boundaries[i + 1]
        if start < end:
            blocks.append([input_grid[r] for r in range(start, end)])

    N = len(blocks)

    # A block "has pattern" if it contains any cell of the separator color
    has_pattern = [
        any(cell == sep_color for row in block for cell in row)
        for block in blocks
    ]

    N_p = sum(has_pattern)
    N_e = N - N_p

    P = (N_p + 1) * (N_e + 1) - 1  # value for pattern blocks
    E = N_e + 1                      # value for empty blocks

    row = [P if has_pattern[i] else E for i in range(N)]
    return [list(row) for _ in range(N)]
