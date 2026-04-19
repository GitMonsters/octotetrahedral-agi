"""ARC puzzle 27a77e38 solver.
Rule: Find the most frequent color above the gray(5) separator line,
place it at the center of the last row in the black area below.
"""
from collections import Counter

def transform(input_grid: list[list[int]]) -> list[list[int]]:
    grid = [row[:] for row in input_grid]
    rows = len(grid)
    cols = len(grid[0])
    
    # Find the gray separator row (all 5s)
    gray_row = None
    for r in range(rows):
        if all(c == 5 for c in grid[r]):
            gray_row = r
            break
    
    # Count colors above gray (exclude 0 and 5)
    color_counts: Counter = Counter()
    for r in range(gray_row):
        for c in range(cols):
            v = grid[r][c]
            if v != 0 and v != 5:
                color_counts[v] += 1
    
    most_common_color = color_counts.most_common(1)[0][0]
    
    # Place at center of last row
    last_row = rows - 1
    center_col = cols // 2
    grid[last_row][center_col] = most_common_color
    
    return grid


# === Test against all training examples ===
examples = [
    ([[3,6,4,2,4],[8,4,3,3,4],[5,5,5,5,5],[0,0,0,0,0],[0,0,0,0,0]],
     [[3,6,4,2,4],[8,4,3,3,4],[5,5,5,5,5],[0,0,0,0,0],[0,0,4,0,0]]),
    ([[2,2,3],[5,5,5],[0,0,0]],
     [[2,2,3],[5,5,5],[0,2,0]]),
    ([[1,9,9,6,1,8,4],[4,6,7,8,9,7,1],[9,3,1,4,1,3,6],[5,5,5,5,5,5,5],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0]],
     [[1,9,9,6,1,8,4],[4,6,7,8,9,7,1],[9,3,1,4,1,3,6],[5,5,5,5,5,5,5],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,1,0,0,0]]),
]

all_pass = True
for i, (inp, expected) in enumerate(examples):
    result = transform(inp)
    ok = result == expected
    print(f"Example {i}: {'PASS' if ok else 'FAIL'}")
    if not ok:
        all_pass = False
        print(f"  Expected: {expected}")
        print(f"  Got:      {result}")

# Test input
test_input = [[9,1,2,8,4,9,8,2,1],[4,4,3,1,2,7,6,7,9],[2,1,6,9,7,8,4,3,6],[9,8,6,3,4,2,9,1,7],[5,5,5,5,5,5,5,5,5],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0]]
test_expected = [[9,1,2,8,4,9,8,2,1],[4,4,3,1,2,7,6,7,9],[2,1,6,9,7,8,4,3,6],[9,8,6,3,4,2,9,1,7],[5,5,5,5,5,5,5,5,5],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,9,0,0,0,0]]
test_result = transform(test_input)
test_ok = test_result == test_expected
print(f"Test:      {'PASS' if test_ok else 'FAIL'}")
if not test_ok:
    all_pass = False
    print(f"  Expected: {test_expected[-1]}")
    print(f"  Got:      {test_result[-1]}")

print("\nSOLVED" if all_pass else "\nFAILED")
