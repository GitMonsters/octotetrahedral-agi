def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    ARC puzzle 5af49b42.

    Rule: "Key" rows (multiple non-zero cells) hold one or more color sequences.
    "Seed" rows (exactly one non-zero cell) hold a single seed pixel whose
    color belongs to one of the sequences.  Place that sequence in the seed row
    so the matching color sits at the seed's column, clipped to grid bounds.
    Key rows are copied through unchanged.
    """
    result = [row[:] for row in grid]
    if not grid:
        return result

    rows = len(grid)
    cols = len(grid[0])

    # Classify every row
    key_row_indices = set()
    seeds = []  # (row, col, value)

    for r, row in enumerate(grid):
        nz = [(c, v) for c, v in enumerate(row) if v != 0]
        if len(nz) >= 2:
            key_row_indices.add(r)
        elif len(nz) == 1:
            seeds.append((r, nz[0][0], nz[0][1]))

    # Extract contiguous non-zero runs from key rows as sequences
    sequences: list[list[int]] = []
    for r in key_row_indices:
        row = grid[r]
        c = 0
        while c < cols:
            if row[c] != 0:
                start = c
                while c < cols and row[c] != 0:
                    c += 1
                sequences.append(list(row[start:c]))
            else:
                c += 1

    # Map each value to every sequence that contains it
    value_to_seqs: dict[int, list[list[int]]] = {}
    for seq in sequences:
        for v in seq:
            value_to_seqs.setdefault(v, [])
            if seq not in value_to_seqs[v]:
                value_to_seqs[v].append(seq)

    # Place sequences for each seed
    for r, c, v in seeds:
        if v not in value_to_seqs:
            continue
        # In well-formed puzzles each seed value is unambiguous
        seq = value_to_seqs[v][0]
        pos = seq.index(v)
        start_col = c - pos
        for i, sv in enumerate(seq):
            col = start_col + i
            if 0 <= col < cols:
                result[r][col] = sv

    return result


if __name__ == "__main__":
    # Load task
    task_path = "~/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/5af49b42.json".replace("~", "/Users/evanpieser")
    with open(task_path, 'r') as f:
        task = json.load(f)
    
    # Test training examples
    print("Testing training examples:")
    all_pass = True
    for idx, example in enumerate(task["train"]):
        result = solve(example["input"])
        expected = example["output"]
        
        if result == expected:
            print(f"  Training {idx}: PASS")
        else:
            print(f"  Training {idx}: FAIL")
            all_pass = False
            # Find first difference
            for r_idx, (res_row, exp_row) in enumerate(zip(result, expected)):
                if res_row != exp_row:
                    print(f"    First diff at row {r_idx}")
                    print(f"      Expected: {exp_row}")
                    print(f"      Got:      {res_row}")
                    break
    
    if all_pass:
        print("\nAll training examples passed!")
    else:
        print("\nSome training examples failed!")
