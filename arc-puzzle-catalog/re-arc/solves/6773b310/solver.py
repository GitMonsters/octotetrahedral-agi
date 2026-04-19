"""ARC puzzle 6773b310 solver.
Rule: Cyan(8) grid lines divide 11x11 into 3x3 cells. Count magenta(6) dots per cell.
If count == 2 → blue(1), else → black(0).
"""

def transform(input_grid: list[list[int]]) -> list[list[int]]:
    # Find row and column dividers (all-8 rows/cols)
    rows = len(input_grid)
    cols = len(input_grid[0])
    
    row_divs = [r for r in range(rows) if all(input_grid[r][c] == 8 for c in range(cols))]
    col_divs = [c for c in range(cols) if all(input_grid[r][c] == 8 for r in range(rows))]
    
    # Build row/col bands
    row_bands = []
    prev = 0
    for rd in row_divs:
        row_bands.append((prev, rd))
        prev = rd + 1
    row_bands.append((prev, rows))
    
    col_bands = []
    prev = 0
    for cd in col_divs:
        col_bands.append((prev, cd))
        prev = cd + 1
    col_bands.append((prev, cols))
    
    output = []
    for r_start, r_end in row_bands:
        row = []
        for c_start, c_end in col_bands:
            count = sum(
                1 for r in range(r_start, r_end) for c in range(c_start, c_end)
                if input_grid[r][c] == 6
            )
            row.append(1 if count >= 2 else 0)
        output.append(row)
    return output


# === Test against all training examples ===
examples = [
    ([[0,0,0,8,0,6,0,8,0,0,6],[0,0,0,8,0,0,0,8,0,6,0],[0,6,0,8,0,6,0,8,0,0,0],[8,8,8,8,8,8,8,8,8,8,8],[0,0,0,8,0,0,0,8,0,0,0],[0,0,0,8,0,6,0,8,0,0,0],[6,0,0,8,0,0,0,8,0,6,0],[8,8,8,8,8,8,8,8,8,8,8],[0,0,0,8,0,0,0,8,6,0,0],[0,6,0,8,0,0,0,8,0,0,6],[0,0,0,8,6,0,0,8,0,0,0]],
     [[0,1,1],[0,0,0],[0,0,1]]),
    ([[6,0,0,8,0,0,0,8,0,0,0],[0,0,0,8,0,0,6,8,0,0,6],[0,0,0,8,0,0,0,8,0,0,0],[8,8,8,8,8,8,8,8,8,8,8],[6,0,0,8,0,0,0,8,0,0,0],[0,0,0,8,0,0,0,8,0,6,0],[0,0,0,8,0,0,6,8,6,0,0],[8,8,8,8,8,8,8,8,8,8,8],[0,0,0,8,0,0,0,8,0,0,0],[6,0,0,8,0,0,0,8,0,0,0],[0,6,0,8,0,6,0,8,0,0,6]],
     [[0,0,0],[0,0,1],[1,0,0]]),
    ([[0,0,0,8,0,0,0,8,0,0,6],[0,0,6,8,0,0,0,8,6,0,0],[0,0,0,8,0,6,0,8,0,0,0],[8,8,8,8,8,8,8,8,8,8,8],[0,0,0,8,0,6,0,8,0,0,0],[6,0,0,8,0,0,6,8,0,0,0],[0,0,0,8,0,0,0,8,0,6,0],[8,8,8,8,8,8,8,8,8,8,8],[0,0,0,8,0,0,0,8,0,0,0],[0,0,6,8,0,0,0,8,6,0,0],[0,0,0,8,0,6,0,8,0,0,0]],
     [[0,0,1],[0,1,0],[0,0,0]]),
    ([[0,0,0,8,0,0,0,8,0,0,0],[6,0,0,8,0,6,0,8,0,0,6],[0,0,6,8,0,0,0,8,0,6,0],[8,8,8,8,8,8,8,8,8,8,8],[0,6,0,8,0,0,6,8,0,0,0],[0,0,0,8,0,0,0,8,0,0,0],[0,6,0,8,0,0,0,8,6,0,0],[8,8,8,8,8,8,8,8,8,8,8],[0,0,6,8,0,0,0,8,0,0,0],[0,0,0,8,0,0,0,8,0,6,0],[0,0,0,8,6,0,0,8,0,0,0]],
     [[1,0,1],[1,0,0],[0,0,0]]),
]

all_pass = True
for i, (inp, expected) in enumerate(examples):
    result = transform(inp)
    ok = result == expected
    print(f"Example {i}: {'PASS' if ok else 'FAIL'}  got={result}  expected={expected}")
    if not ok:
        all_pass = False

# Test input
test_input = [[0,0,0,8,0,0,0,8,6,0,6],[0,6,0,8,0,0,6,8,0,0,0],[0,0,0,8,0,0,0,8,0,0,0],[8,8,8,8,8,8,8,8,8,8,8],[0,0,0,8,0,0,0,8,0,6,0],[0,0,6,8,0,6,0,8,0,0,0],[0,0,0,8,6,0,0,8,0,0,6],[8,8,8,8,8,8,8,8,8,8,8],[0,0,6,8,0,0,0,8,0,0,0],[6,0,0,8,0,0,0,8,0,6,0],[0,0,0,8,0,6,0,8,0,0,0]]
test_expected = [[0,0,1],[0,1,1],[1,0,0]]
test_result = transform(test_input)
test_ok = test_result == test_expected
print(f"Test:      {'PASS' if test_ok else 'FAIL'}  got={test_result}  expected={test_expected}")

if all_pass and test_ok:
    print("SOLVED")
else:
    print("FAILED")
