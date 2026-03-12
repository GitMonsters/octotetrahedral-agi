import json
with open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/33b52de3.json") as f:
    task = json.load(f)

# Test on first training example
example = task["train"][0]
input_grid = example["input"]
output_grid = example["output"]

# Check if block_col_idx bounds are correct
for j in [2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20]:
    block_col_idx = (j - 2) // 4
    in_block_col = (j - 2) % 4
    print(f"Col {j}: block_col_idx={block_col_idx}, in_block_col={in_block_col}")

print("\nExpected blocks: 0, 1, 2, 3, 4")

# Check a few specific mappings
print(f"\nExample checks:")
print(f"Col 2 -> block 0: {(2-2)//4} = 0 ✓")
print(f"Col 6 -> block 1: {(6-2)//4} = 1 ✓") 
print(f"Col 10 -> block 2: {(10-2)//4} = 2 ✓")
print(f"Col 14 -> block 3: {(14-2)//4} = 3 ✓")
print(f"Col 18 -> block 4: {(18-2)//4} = 4 ✓")

# But wait, the key is only 5 columns (0-4), what about block_col_idx=4?
print(f"\nKey matrix has columns 0-4")
print(f"Block 4 (col 18-20) should map to key column 4")