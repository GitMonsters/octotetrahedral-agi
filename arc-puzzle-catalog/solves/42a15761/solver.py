def solve(grid: list[list[int]]) -> list[list[int]]:
    """Sort column blocks by number of 0s (holes), ascending left-to-right."""
    rows = len(grid)
    cols = len(grid[0])

    # Find separator columns (all zeros)
    sep_cols = [c for c in range(cols) if all(grid[r][c] == 0 for r in range(rows))]

    # Extract blocks between separators
    blocks = []
    # Find block column ranges
    block_ranges = []
    # Blocks are between separator columns, plus before first and after last if needed
    all_boundaries = [-1] + sep_cols + [cols]
    for i in range(len(all_boundaries) - 1):
        c_start = all_boundaries[i] + 1
        c_end = all_boundaries[i + 1]
        if c_start < c_end:
            block = [row[c_start:c_end] for row in grid]
            block_ranges.append((c_start, c_end))
            blocks.append(block)

    # Count zeros in each block
    zero_counts = []
    for block in blocks:
        count = sum(cell == 0 for row in block for cell in row)
        zero_counts.append(count)

    # Sort blocks by zero count ascending
    sorted_indices = sorted(range(len(blocks)), key=lambda i: zero_counts[i])
    sorted_blocks = [blocks[sorted_indices[i]] for i in range(len(blocks))]

    # Reconstruct output
    output = [row[:] for row in grid]
    for idx, (c_start, c_end) in enumerate(block_ranges):
        block = sorted_blocks[idx]
        for r in range(rows):
            for c in range(c_end - c_start):
                output[r][c_start + c] = block[r][c]

    return output


if __name__ == "__main__":
    import sys, json
    with open(sys.argv[1]) as f:
        task = json.load(f)
    for i, ex in enumerate(task['train'] + task['test']):
        result = solve(ex['input'])
        status = "PASS" if result == ex['output'] else "FAIL"
        print(f"{'Train' if i < len(task['train']) else 'Test'} {i if i < len(task['train']) else i - len(task['train'])}: {status}")
