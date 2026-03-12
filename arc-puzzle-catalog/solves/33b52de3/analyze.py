import json

# Load the task
with open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/33b52de3.json") as f:
    task = json.load(f)

example = task["train"][0]
input_grid = example["input"]
output_grid = example["output"]

print("=== MAPPING ANALYSIS ===")

# Key pattern - 3x5 matrix
key_matrix = {}
for i in range(19, 22):
    for j in range(1, 6):
        key_matrix[(i-19, j-1)] = input_grid[i][j]

print("Key matrix:")
for i in range(3):
    row = []
    for j in range(5):
        row.append(key_matrix[(i,j)])
    print(f"Row {i}: {row}")

# Analyze all block mappings
block_positions = [
    (1, 2),   # First block position (top-left)
    (1, 6),   # Second block 
    (1, 10),  # Third block
    (1, 14),  # Fourth block
    (1, 18),  # Fifth block
]

print("\nBlock analysis:")
for block_idx, (start_row, start_col) in enumerate(block_positions):
    print(f"\nBlock {block_idx} (starts at {start_row},{start_col}):")
    print(f"Uses key column {block_idx}")
    
    # Show what values this block gets
    block_values = set()
    for i in range(start_row, start_row + 3):
        for j in range(start_col, start_col + 3):
            if i < len(output_grid) and j < len(output_grid[0]):
                if input_grid[i][j] == 5:
                    block_values.add(output_grid[i][j])
    print(f"Block values: {sorted(block_values)}")
    
    # Check if it matches key column
    key_col_values = set()
    for row in range(3):
        key_col_values.add(key_matrix[(row, block_idx)])
    print(f"Key col {block_idx} values: {sorted(key_col_values)}")
    
# Let me check the exact mapping
print("\nDetailed mapping for block 0:")
for i in range(1, 4):
    for j in range(2, 5):
        if input_grid[i][j] == 5:
            block_row = i - 1
            block_col = j - 2
            output_val = output_grid[i][j]
            key_val = key_matrix[(block_row, 0)]  # Use column 0 for first block
            print(f"Block[{block_row}][{block_col}] -> output={output_val}, key={key_val}")