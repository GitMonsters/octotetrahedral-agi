import json

def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    ARC puzzle 5af49b42 solver.
    
    Rule: Find all complete sequences (consecutive non-zero values in the grid).
    For each row with isolated non-zero values, fill them with the complete 
    sequence containing those values, aligned properly.
    """
    grid = [row[:] for row in grid]  # Deep copy
    
    if not grid:
        return grid
    
    # Extract all sequences from the grid (consecutive non-zero values, length >= 2)
    sequences = []
    
    for row in grid:
        current_seq = []
        current_start_col = None
        
        for col_idx, val in enumerate(row):
            if val != 0:
                if current_start_col is None:
                    current_start_col = col_idx
                current_seq.append(val)
            elif current_seq:
                if len(current_seq) > 1:  # Only keep sequences of length > 1
                    sequences.append({
                        'values': current_seq[:],
                        'start_col': current_start_col
                    })
                current_seq = []
                current_start_col = None
        
        if current_seq and len(current_seq) > 1:
            sequences.append({
                'values': current_seq[:],
                'start_col': current_start_col
            })
    
    if not sequences:
        return grid
    
    # Build a map from value to sequence
    value_to_seq = {}
    for seq in sequences:
        for val in seq['values']:
            value_to_seq[val] = seq
    
    # Process each row
    for row_idx in range(len(grid)):
        row = grid[row_idx]
        
        # Find all non-zero values in this row
        non_zero_positions = [(col_idx, val) for col_idx, val in enumerate(row) if val != 0]
        
        # Process each non-zero value that is part of a known sequence
        for col_idx, val in non_zero_positions:
            if val in value_to_seq:
                seq = value_to_seq[val]
                sequence = seq['values']
                
                # Find position of val in the sequence
                seq_pos = sequence.index(val)
                
                # Place the sequence such that val aligns
                start_col = col_idx - seq_pos
                
                # Write the sequence
                for i, seq_val in enumerate(sequence):
                    pos = start_col + i
                    if 0 <= pos < len(row):
                        grid[row_idx][pos] = seq_val
    
    return grid


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
