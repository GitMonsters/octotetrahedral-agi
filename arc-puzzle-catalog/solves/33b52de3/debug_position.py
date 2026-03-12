import json
with open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/33b52de3.json") as f:
    task = json.load(f)

example = task["train"][0]

# Check position (2,2)
i, j = 2, 2
block_col_idx = (j - 2) // 4  # = 0
in_block_col = (j - 2) % 4    # = 0
block_row_idx = (i - 1) // 4  # = 0  
in_block_row = (i - 1) % 4    # = 1

print(f"Position (2,2): block_col_idx={block_col_idx}, in_block_row={in_block_row}")
print(f"Key lookup: ({in_block_row}, {block_col_idx})")

# Build key matrix to check
key_min_row, key_min_col = 19, 1
key_matrix = {}
for i_key in range(19, 22):
    for j_key in range(1, 6):
        key_row = i_key - key_min_row
        key_col = j_key - key_min_col
        key_matrix[(key_row, key_col)] = example["input"][i_key][j_key]

print(f"Key matrix [(1, 0)] = {key_matrix[(1, 0)]}")
print(f"Expected output: {example['output'][2][2]}")

# The issue might be in my understanding of the pattern. Let me check the cross pattern more carefully
print("\nInput 5s pattern in first block:")
for row in range(1, 4):
    line = ""
    for col in range(2, 5):
        if example["input"][row][col] == 5:
            line += "5 "
        else:
            line += ". "
    print(f"Row {row}: {line}")

print("\nOutput pattern in first block:")
for row in range(1, 4):
    line = ""
    for col in range(2, 5):
        line += f"{example['output'][row][col]} "
    print(f"Row {row}: {line}")

# Notice that the center of row 2 (position 2,3) is 0 in input and 0 in output!