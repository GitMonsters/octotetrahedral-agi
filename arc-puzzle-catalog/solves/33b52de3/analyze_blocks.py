import json
with open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/33b52de3.json") as f:
    task = json.load(f)

# Let me look at this differently - maybe each block gets uniform color
example = task["train"][0]
output = example["output"]

blocks = [
    (1, 2, 3),   # Block 0: rows 1-3, cols 2-4
    (1, 6, 3),   # Block 1: rows 1-3, cols 6-8
    (1, 10, 3),  # Block 2: rows 1-3, cols 10-12
    (1, 14, 3),  # Block 3: rows 1-3, cols 14-16  
    (1, 18, 3),  # Block 4: rows 1-3, cols 18-20
]

print("Values in each block:")
for i, (start_row, start_col, size) in enumerate(blocks):
    values = set()
    for r in range(start_row, start_row + size):
        for c in range(start_col, start_col + size):
            if r < len(output) and c < len(output[0]) and output[r][c] != 0:
                values.add(output[r][c])
    print(f"Block {i}: {sorted(values)}")

# Check key values by column
print("\nKey values by column:")
for col in range(5):
    values = set()
    for row in range(3):
        key_val = example["input"][19 + row][1 + col]
        if key_val != 0:
            values.add(key_val)
    print(f"Key col {col}: {sorted(values)}")

# Hmm, block 0 only has value 2, but key col 0 has [1, 2]
# Let me check if each block uses a specific position from the key column...
print("\nTesting if each block uses key[0, col] (top row):")
for col in range(5):
    key_val = example["input"][19][1 + col]  # Top row of key
    print(f"Key[0][{col}] = {key_val}")

print("\nComparing to block values (first position in each block):")
for i, (start_row, start_col, size) in enumerate(blocks):
    block_val = output[start_row][start_col]
    print(f"Block {i} first value: {block_val}")