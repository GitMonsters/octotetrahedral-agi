def transform(input_grid: list[list[int]]) -> list[list[int]]:
    rows = len(input_grid)
    cols = len(input_grid[0])
    bg = 7
    
    # Find vertical bars: each is on a column, extends from bottom upward
    bars = []  # list of (col, color, size)
    for c in range(cols):
        # Check bottom row for non-bg
        color = None
        size = 0
        for r in range(rows - 1, -1, -1):
            if input_grid[r][c] != bg:
                if color is None:
                    color = input_grid[r][c]
                if input_grid[r][c] == color:
                    size += 1
                else:
                    break
            else:
                break
        if color is not None and size > 0:
            bars.append((c, color, size))
    
    # Sort by column (should already be)
    bars.sort(key=lambda x: x[0])
    
    n = len(bars)
    bar_cols = [b[0] for b in bars]
    colors = [b[1] for b in bars]
    sizes = [b[2] for b in bars]
    
    # Rotate colors right by 1, sizes left by 1
    new_colors = [colors[-1]] + colors[:-1]
    new_sizes = sizes[1:] + [sizes[0]]
    
    # Build output
    output = [[bg] * cols for _ in range(rows)]
    for i in range(n):
        c = bar_cols[i]
        color = new_colors[i]
        sz = new_sizes[i]
        for r in range(rows - sz, rows):
            output[r][c] = color
    
    return output


# Test against all training examples
examples = [
    ([[7,7,7,7,7,7,7,7,7],[7,7,7,7,7,6,7,7,7],[7,7,7,7,7,6,7,7,7],[7,7,7,7,7,6,7,7,7],[7,7,7,7,7,6,7,7,7],[7,7,7,7,7,6,7,7,7],[7,7,7,8,7,6,7,1,7],[7,9,7,8,7,6,7,1,7],[7,9,7,8,7,6,7,1,7]],
     [[7,7,7,7,7,7,7,7,7],[7,7,7,9,7,7,7,7,7],[7,7,7,9,7,7,7,7,7],[7,7,7,9,7,7,7,7,7],[7,7,7,9,7,7,7,7,7],[7,7,7,9,7,7,7,7,7],[7,1,7,9,7,8,7,7,7],[7,1,7,9,7,8,7,6,7],[7,1,7,9,7,8,7,6,7]]),
    ([[7,7,7,7,7,7,7],[7,2,7,7,7,7,7],[7,2,7,7,7,7,7],[7,2,7,7,7,5,7],[7,2,7,8,7,5,7],[7,2,7,8,7,5,7],[7,2,7,8,7,5,7]],
     [[7,7,7,7,7,7,7],[7,7,7,7,7,8,7],[7,7,7,7,7,8,7],[7,7,7,2,7,8,7],[7,5,7,2,7,8,7],[7,5,7,2,7,8,7],[7,5,7,2,7,8,7]]),
    ([[7,7,7,7,7,7,7,7,7,7,7],[7,7,7,7,7,7,7,7,7,7,7],[7,7,7,7,7,7,7,7,7,7,7],[7,7,7,7,7,7,7,4,7,7,7],[7,7,7,7,7,7,7,4,7,7,7],[7,7,7,7,7,7,7,4,7,8,7],[7,7,7,2,7,7,7,4,7,8,7],[7,1,7,2,7,7,7,4,7,8,7],[7,1,7,2,7,5,7,4,7,8,7],[7,1,7,2,7,5,7,4,7,8,7],[7,1,7,2,7,5,7,4,7,8,7]],
     [[7,7,7,7,7,7,7,7,7,7,7],[7,7,7,7,7,7,7,7,7,7,7],[7,7,7,7,7,7,7,7,7,7,7],[7,7,7,7,7,2,7,7,7,7,7],[7,7,7,7,7,2,7,7,7,7,7],[7,7,7,7,7,2,7,5,7,7,7],[7,8,7,7,7,2,7,5,7,7,7],[7,8,7,7,7,2,7,5,7,4,7],[7,8,7,1,7,2,7,5,7,4,7],[7,8,7,1,7,2,7,5,7,4,7],[7,8,7,1,7,2,7,5,7,4,7]])
]

test_input = [[7,7,7,7,7,7,7,7,7,7,7],[7,0,7,7,7,7,7,7,7,7,7],[7,0,7,7,7,7,7,7,7,7,7],[7,0,7,7,7,7,7,7,7,7,7],[7,0,7,7,7,7,7,7,7,7,7],[7,0,7,7,7,7,7,7,7,7,7],[7,0,7,7,7,7,7,7,7,6,7],[7,0,7,7,7,8,7,9,7,6,7],[7,0,7,7,7,8,7,9,7,6,7],[7,0,7,7,7,8,7,9,7,6,7],[7,0,7,2,7,8,7,9,7,6,7]]
test_expected = [[7,7,7,7,7,7,7,7,7,7,7],[7,7,7,7,7,7,7,7,7,9,7],[7,7,7,7,7,7,7,7,7,9,7],[7,7,7,7,7,7,7,7,7,9,7],[7,7,7,7,7,7,7,7,7,9,7],[7,7,7,7,7,7,7,7,7,9,7],[7,7,7,7,7,7,7,8,7,9,7],[7,7,7,0,7,2,7,8,7,9,7],[7,7,7,0,7,2,7,8,7,9,7],[7,7,7,0,7,2,7,8,7,9,7],[7,6,7,0,7,2,7,8,7,9,7]]

all_pass = True
for i, (inp, exp) in enumerate(examples):
    out = transform(inp)
    if out == exp:
        print(f"Example {i}: PASS")
    else:
        print(f"Example {i}: FAIL")
        all_pass = False

test_out = transform(test_input)
if test_out == test_expected:
    print("Test: PASS")
else:
    print("Test: FAIL")
    all_pass = False

print("SOLVED" if all_pass else "FAILED")
