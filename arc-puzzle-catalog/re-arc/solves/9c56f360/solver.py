"""ARC-AGI puzzle 9c56f360

Rule: Each row's contiguous block of 3s slides LEFT until it hits
an 8-wall (or the grid edge). The block lands immediately to the
right of the nearest 8 to its left.
"""


def transform(input_grid: list[list[int]]) -> list[list[int]]:
    grid = [row[:] for row in input_grid]
    cols = len(grid[0])

    for r, row in enumerate(grid):
        three_cols = [c for c in range(cols) if row[c] == 3]
        if not three_cols:
            continue

        width = len(three_cols)
        left_edge = min(three_cols)

        # Clear original 3s
        for c in three_cols:
            grid[r][c] = 0

        # Scan left from the original position to find the 8-wall
        new_start = 0
        for c in range(left_edge - 1, -1, -1):
            if grid[r][c] == 8:
                new_start = c + 1
                break

        # Place the 3-block at its new position
        for i in range(width):
            grid[r][new_start + i] = 3

    return grid


# ── Testing ──────────────────────────────────────────────────────
examples = [
    {
        "input": [
            [0, 0, 0, 8, 0, 0, 8, 3],
            [0, 8, 0, 0, 8, 0, 0, 3],
            [8, 8, 0, 8, 0, 0, 8, 3],
            [8, 8, 0, 0, 0, 0, 0, 3],
            [0, 0, 0, 8, 8, 0, 0, 8],
            [8, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 8, 8, 8, 0, 0],
        ],
        "output": [
            [0, 0, 0, 8, 0, 0, 8, 3],
            [0, 8, 0, 0, 8, 3, 0, 0],
            [8, 8, 0, 8, 0, 0, 8, 3],
            [8, 8, 3, 0, 0, 0, 0, 0],
            [0, 0, 0, 8, 8, 0, 0, 8],
            [8, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 8, 8, 8, 0, 0],
        ],
    },
    {
        "input": [
            [0, 0, 0, 8, 0, 0],
            [0, 0, 8, 0, 0, 8],
            [8, 0, 0, 0, 0, 8],
            [0, 0, 8, 0, 8, 0],
            [0, 0, 0, 0, 3, 3],
            [8, 0, 8, 0, 3, 3],
            [0, 8, 0, 8, 8, 0],
        ],
        "output": [
            [0, 0, 0, 8, 0, 0],
            [0, 0, 8, 0, 0, 8],
            [8, 0, 0, 0, 0, 8],
            [0, 0, 8, 0, 8, 0],
            [3, 3, 0, 0, 0, 0],
            [8, 0, 8, 3, 3, 0],
            [0, 8, 0, 8, 8, 0],
        ],
    },
    {
        "input": [
            [0, 0, 0, 0, 8, 8, 8, 8],
            [0, 0, 0, 8, 0, 8, 3, 3],
            [8, 0, 0, 8, 0, 0, 3, 3],
            [8, 8, 0, 0, 0, 0, 3, 3],
            [8, 8, 0, 0, 8, 8, 0, 8],
            [0, 0, 0, 8, 0, 8, 0, 3],
            [0, 8, 0, 0, 0, 0, 0, 3],
            [0, 0, 0, 8, 8, 0, 8, 3],
            [8, 0, 0, 8, 8, 8, 0, 8],
        ],
        "output": [
            [0, 0, 0, 0, 8, 8, 8, 8],
            [0, 0, 0, 8, 0, 8, 3, 3],
            [8, 0, 0, 8, 3, 3, 0, 0],
            [8, 8, 3, 3, 0, 0, 0, 0],
            [8, 8, 0, 0, 8, 8, 0, 8],
            [0, 0, 0, 8, 0, 8, 3, 0],
            [0, 8, 3, 0, 0, 0, 0, 0],
            [0, 0, 0, 8, 8, 0, 8, 3],
            [8, 0, 0, 8, 8, 8, 0, 8],
        ],
    },
]

test = {
    "input": [
        [0, 8, 8, 8, 8, 8, 8, 0, 8],
        [8, 8, 8, 0, 0, 8, 8, 0, 8],
        [0, 8, 8, 0, 8, 8, 0, 0, 8],
        [0, 8, 0, 0, 0, 0, 0, 3, 3],
        [0, 8, 0, 8, 0, 0, 0, 3, 3],
        [8, 0, 0, 0, 0, 0, 0, 3, 3],
        [0, 0, 8, 0, 8, 8, 0, 3, 3],
        [0, 8, 8, 8, 0, 0, 0, 0, 0],
        [0, 8, 0, 8, 0, 8, 8, 8, 0],
    ],
    "output": [
        [0, 8, 8, 8, 8, 8, 8, 0, 8],
        [8, 8, 8, 0, 0, 8, 8, 0, 8],
        [0, 8, 8, 0, 8, 8, 0, 0, 8],
        [0, 8, 3, 3, 0, 0, 0, 0, 0],
        [0, 8, 0, 8, 3, 3, 0, 0, 0],
        [8, 3, 3, 0, 0, 0, 0, 0, 0],
        [0, 0, 8, 0, 8, 8, 3, 3, 0],
        [0, 8, 8, 8, 0, 0, 0, 0, 0],
        [0, 8, 0, 8, 0, 8, 8, 8, 0],
    ],
}

all_pass = True
for i, ex in enumerate(examples):
    result = transform(ex["input"])
    ok = result == ex["output"]
    print(f"Example {i}: {'PASS' if ok else 'FAIL'}")
    if not ok:
        all_pass = False
        for r, (got, exp) in enumerate(zip(result, ex["output"])):
            if got != exp:
                print(f"  Row {r}: got {got} expected {exp}")

result = transform(test["input"])
ok = result == test["output"]
print(f"Test:      {'PASS' if ok else 'FAIL'}")
if not ok:
    all_pass = False
    for r, (got, exp) in enumerate(zip(result, test["output"])):
        if got != exp:
            print(f"  Row {r}: got {got} expected {exp}")

print("SOLVED" if all_pass else "FAILED")
