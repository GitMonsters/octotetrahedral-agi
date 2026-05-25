import json
with open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/33b52de3.json") as f:
    task = json.load(f)

def analyze_example(ex_num):
    example = task["train"][ex_num]
    input_grid = example["input"]
    output_grid = example["output"]
    
    print(f"\n=== EXAMPLE {ex_num} ===")
    
    # Find key pattern
    key_positions = []
    for i in range(len(input_grid)):
        for j in range(len(input_grid[0])):
            if input_grid[i][j] != 0 and input_grid[i][j] != 5:
                key_positions.append((i, j))
    
    key_min_row = min(pos[0] for pos in key_positions)
    key_max_row = max(pos[0] for pos in key_positions)
    key_min_col = min(pos[1] for pos in key_positions)
    key_max_col = max(pos[1] for pos in key_positions)
    
    print(f"Key bounds: rows {key_min_row}-{key_max_row}, cols {key_min_col}-{key_max_col}")
    
    # Print key matrix
    print("Key matrix:")
    for i in range(key_min_row, key_max_row + 1):
        row_vals = []
        for j in range(key_min_col, key_max_col + 1):
            row_vals.append(input_grid[i][j])
        print(f"  Row {i-key_min_row}: {row_vals}")
    
    # Analyze blocks and their values
    block_starts = []
    # Find where 5s start (blocks)
    for j in range(len(input_grid[0])):
        for i in range(len(input_grid)):
            if input_grid[i][j] == 5:
                # Check if this is a new block start
                is_new_block = True
                for start_r, start_c in block_starts:
                    if abs(i - start_r) <= 3 and abs(j - start_c) <= 3:
                        is_new_block = False
                        break
                if is_new_block:
                    block_starts.append((i, j))
                    if len(block_starts) >= 5:  # Limit to 5 blocks
                        break
        if len(block_starts) >= 5:
            break
    
    print(f"Block starts found: {block_starts}")
    
    # Analyze each block
    for block_idx, (start_r, start_c) in enumerate(block_starts[:5]):
        values_in_block = set()
        for dr in range(3):
            for dc in range(3):
                r, c = start_r + dr, start_c + dc
                if r < len(output_grid) and c < len(output_grid[0]):
                    if input_grid[r][c] == 5:  # Only look at positions that had 5s
                        values_in_block.add(output_grid[r][c])
        
        print(f"Block {block_idx} (at {start_r},{start_c}): values {sorted(values_in_block)}")

# Analyze both examples
for i in range(2):
    analyze_example(i)