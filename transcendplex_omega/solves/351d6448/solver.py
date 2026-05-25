def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Extract blocks separated by rows of all 5s.
    Find the content row (non-zero) in each block.
    Determine the progression pattern and extend it by one more row.
    
    Train 0: Constant sequence (2,2,3) shifts right by 1 each block
    Train 1: Sequence of 1s grows by 2 each block
    
    Output the next iteration of this pattern as a 3-row grid.
    """
    
    # Find rows that are all 5s (separators)
    blocks = []
    current_block = []
    for i, row in enumerate(grid):
        if all(val == 5 for val in row):
            if current_block:
                blocks.append(current_block)
            current_block = []
        else:
            current_block.append(row)
    if current_block:
        blocks.append(current_block)
    
    if not blocks:
        return [[0] * len(grid[0]) for _ in range(3)]
    
    # Extract the content row (first non-zero row) from each block
    content_rows = []
    for block in blocks:
        for row in block:
            if any(val != 0 for val in row):
                content_rows.append(row[:])
                break
    
    if not content_rows:
        return [[0] * len(grid[0]) for _ in range(3)]
    
    # Find leftmost and rightmost non-zero positions in each content row
    patterns = []
    for row in content_rows:
        left = -1
        right = -1
        for j in range(len(row)):
            if row[j] != 0:
                if left == -1:
                    left = j
                right = j
        if left != -1:
            patterns.append((left, right, right - left + 1))  # (leftmost, rightmost, sequence_length)
    
    if len(patterns) < 2:
        # Not enough patterns to determine progression
        # Return the last content row
        return [
            [0] * len(grid[0]),
            content_rows[-1],
            [0] * len(grid[0])
        ]
    
    # Determine the progression
    # Check if it's a leftward shift (Train 0) or a growth (Train 1)
    
    # Calculate diffs in left, right, and length
    left_diffs = [patterns[i+1][0] - patterns[i][0] for i in range(len(patterns)-1)]
    length_diffs = [patterns[i+1][2] - patterns[i][2] for i in range(len(patterns)-1)]
    
    # Determine the pattern
    if all(diff == 0 for diff in length_diffs) and all(diff != 0 for diff in left_diffs):
        # Case 1: Constant sequence length, shifting position (Train 0)
        left_shift = left_diffs[0]
        seq_length = patterns[0][2]
        
        next_left = patterns[-1][0] + left_shift
        next_right = next_left + seq_length - 1
        
        # Build the output row
        output_row = [0] * len(grid[0])
        last_row = content_rows[-1]
        
        # Copy the actual values from the last row
        for j in range(len(last_row)):
            if last_row[j] != 0:
                # Get the relative position within the sequence
                rel_pos = j - patterns[-1][0]
                new_pos = next_left + rel_pos
                if 0 <= new_pos < len(output_row):
                    output_row[new_pos] = last_row[j]
    
    elif all(diff != 0 for diff in length_diffs):
        # Case 2: Growing sequence (Train 1)
        length_growth = length_diffs[0]
        next_length = patterns[-1][2] + length_growth
        next_left = patterns[-1][0]
        
        # Build the output row
        output_row = [0] * len(grid[0])
        last_row = content_rows[-1]
        
        # Get the value to extend (usually the value at the rightmost position)
        extend_val = last_row[patterns[-1][1]]
        
        # Fill from leftmost to next_length
        for j in range(next_left, next_left + next_length):
            if j < len(output_row):
                output_row[j] = extend_val
    
    else:
        # Fallback: return last content row shifted
        output_row = [0] + content_rows[-1][:-1]
    
    # Return 3 rows: all-zeros, pattern, all-zeros
    return [
        [0] * len(grid[0]),
        output_row,
        [0] * len(grid[0])
    ]


if __name__ == "__main__":
    import json
    with open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/351d6448.json") as f:
        task = json.load(f)
    for i, ex in enumerate(task["train"]):
        result = solve(ex["input"])
        status = "PASS" if result == ex["output"] else "FAIL"
        print(f"Train {i}: {status}")
        if status == "FAIL":
            print(f"  Expected: {ex['output'][1]}")
            print(f"  Got:      {result[1]}")

