import copy

def transform(input_grid: list[list[int]]) -> list[list[int]]:
    """Draw a diagonal line (cyan=8) through the black blob.
    
    Rule: Find topmost row with black(0) cells. Try two diagonals:
      \ from (top_row, leftmost_col) going down-right
      / from (top_row, rightmost_col) going down-left
    Pick whichever crosses more black cells. Mark those cells cyan(8).
    """
    grid = copy.deepcopy(input_grid)
    rows, cols = len(grid), len(grid[0])

    # Find topmost row with black cells, and its left/right extent
    top_row = left_col = right_col = None
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 0:
                if top_row is None:
                    top_row = r
                    left_col = right_col = c
                elif r == top_row:
                    left_col = min(left_col, c)
                    right_col = max(right_col, c)
        if top_row is not None and r > top_row:
            break

    if top_row is None:
        return grid

    def count_black(sr, sc, dr, dc):
        n, r, c = 0, sr, sc
        while 0 <= r < rows and 0 <= c < cols:
            if grid[r][c] == 0:
                n += 1
            r += dr; c += dc
        return n

    cnt_bs = count_black(top_row, left_col, 1, 1)   # \ diagonal
    cnt_sl = count_black(top_row, right_col, 1, -1)  # / diagonal

    if cnt_bs >= cnt_sl:
        sr, sc, dr, dc = top_row, left_col, 1, 1
    else:
        sr, sc, dr, dc = top_row, right_col, 1, -1

    r, c = sr, sc
    while 0 <= r < rows and 0 <= c < cols:
        if grid[r][c] == 0:
            grid[r][c] = 8
        r += dr; c += dc

    return grid


# === Verification ===
examples = [
    ([[5,5,5,5,5,5,5,5,5,5],[5,0,0,0,0,0,5,5,5,5],[5,5,0,0,0,0,5,0,0,5],[5,0,0,0,0,0,0,0,0,5],[5,5,0,0,0,0,0,0,0,5],[5,5,0,0,0,0,0,0,5,5],[5,0,0,0,0,0,0,0,0,5],[5,0,0,5,5,0,0,0,0,5],[5,5,5,5,5,0,5,5,0,5],[5,5,5,5,5,5,5,5,5,5]],
     [[5,5,5,5,5,5,5,5,5,5],[5,8,0,0,0,0,5,5,5,5],[5,5,8,0,0,0,5,0,0,5],[5,0,0,8,0,0,0,0,0,5],[5,5,0,0,8,0,0,0,0,5],[5,5,0,0,0,8,0,0,5,5],[5,0,0,0,0,0,8,0,0,5],[5,0,0,5,5,0,0,8,0,5],[5,5,5,5,5,0,5,5,8,5],[5,5,5,5,5,5,5,5,5,5]]),
    ([[5,5,5,5,5,5,5,5,5,5],[5,5,5,0,5,0,5,0,0,5],[5,5,0,0,5,0,0,0,0,5],[5,0,0,0,0,0,0,0,0,5],[5,5,0,0,0,0,0,0,0,5],[5,5,5,0,0,0,0,0,5,5],[5,0,0,0,0,0,0,0,0,5],[5,0,0,0,0,0,0,5,5,5],[5,5,0,5,0,0,5,5,5,5],[5,5,5,5,5,5,5,5,5,5]],
     [[5,5,5,5,5,5,5,5,5,5],[5,5,5,0,5,0,5,0,8,5],[5,5,0,0,5,0,0,8,0,5],[5,0,0,0,0,0,8,0,0,5],[5,5,0,0,0,8,0,0,0,5],[5,5,5,0,8,0,0,0,5,5],[5,0,0,8,0,0,0,0,0,5],[5,0,8,0,0,0,0,5,5,5],[5,5,0,5,0,0,5,5,5,5],[5,5,5,5,5,5,5,5,5,5]]),
    ([[5,5,5,5,5,5,5,5,5,5],[5,5,5,0,0,0,5,5,5,5],[5,5,5,0,0,0,0,5,5,5],[5,5,0,0,0,0,0,0,0,5],[5,0,0,0,0,0,0,0,5,5],[5,0,0,0,0,0,0,0,0,5],[5,5,5,0,5,5,0,0,0,5],[5,5,0,0,5,5,0,0,5,5],[5,5,5,0,5,5,5,0,5,5],[5,5,5,5,5,5,5,5,5,5]],
     [[5,5,5,5,5,5,5,5,5,5],[5,5,5,8,0,0,5,5,5,5],[5,5,5,0,8,0,0,5,5,5],[5,5,0,0,0,8,0,0,0,5],[5,0,0,0,0,0,8,0,5,5],[5,0,0,0,0,0,0,8,0,5],[5,5,5,0,5,5,0,0,8,5],[5,5,0,0,5,5,0,0,5,5],[5,5,5,0,5,5,5,0,5,5],[5,5,5,5,5,5,5,5,5,5]]),
    ([[5,5,5,5,5,5,5,5,5,5],[5,0,0,0,5,5,0,0,5,5],[5,5,0,0,0,5,0,0,0,5],[5,5,5,0,0,0,0,0,5,5],[5,5,0,0,0,0,0,0,0,5],[5,0,0,0,0,0,0,0,5,5],[5,5,0,0,0,5,0,0,0,5],[5,0,0,5,0,5,0,0,0,5],[5,5,0,5,5,5,0,5,0,5],[5,5,5,5,5,5,5,5,5,5]],
     [[5,5,5,5,5,5,5,5,5,5],[5,8,0,0,5,5,0,0,5,5],[5,5,8,0,0,5,0,0,0,5],[5,5,5,8,0,0,0,0,5,5],[5,5,0,0,8,0,0,0,0,5],[5,0,0,0,0,8,0,0,5,5],[5,5,0,0,0,5,8,0,0,5],[5,0,0,5,0,5,0,8,0,5],[5,5,0,5,5,5,0,5,8,5],[5,5,5,5,5,5,5,5,5,5]]),
]

test_input = [[5,5,5,5,5,5,5,5,5,5],[5,5,5,0,0,0,0,0,5,5],[5,5,0,0,0,0,0,0,5,5],[5,5,5,0,0,0,0,0,0,5],[5,0,0,0,0,0,0,5,5,5],[5,0,0,0,0,0,5,5,5,5],[5,0,0,0,0,0,0,0,0,5],[5,0,0,5,5,0,0,5,0,5],[5,5,5,5,5,5,5,5,0,5],[5,5,5,5,5,5,5,5,5,5]]
test_expected = [[5,5,5,5,5,5,5,5,5,5],[5,5,5,0,0,0,0,8,5,5],[5,5,0,0,0,0,8,0,5,5],[5,5,5,0,0,8,0,0,0,5],[5,0,0,0,8,0,0,5,5,5],[5,0,0,8,0,0,5,5,5,5],[5,0,8,0,0,0,0,0,0,5],[5,8,0,5,5,0,0,5,0,5],[5,5,5,5,5,5,5,5,0,5],[5,5,5,5,5,5,5,5,5,5]]

all_pass = True
for i, (inp, exp) in enumerate(examples):
    out = transform(inp)
    ok = out == exp
    print(f"Example {i}: {'PASS' if ok else 'FAIL'}")
    if not ok:
        all_pass = False
        for r in range(len(exp)):
            for c in range(len(exp[0])):
                if out[r][c] != exp[r][c]:
                    print(f"  Diff at ({r},{c}): got {out[r][c]} expected {exp[r][c]}")

test_out = transform(test_input)
test_ok = test_out == test_expected
print(f"Test:      {'PASS' if test_ok else 'FAIL'}")
if not test_ok:
    all_pass = False

print("\nSOLVED" if all_pass else "\nFAILED")
