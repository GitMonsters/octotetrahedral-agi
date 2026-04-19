"""ARC-AGI puzzle 7e4d4f7c solver.

Rule: The grid has a pattern row (row 0), a marker row (row 1), and repeated
filler rows. Output is 3 rows: row 0, row 1, and a new row where every
non-background value in row 0 is replaced with 6.
"""


def transform(input_grid: list[list[int]]) -> list[list[int]]:
    row0 = list(input_grid[0])
    row1 = list(input_grid[1])
    background = row1[1]  # dominant fill color
    row2 = [6 if v != background else background for v in row0]
    return [row0, row1, row2]


# --- Verification ---
if __name__ == "__main__":
    examples = [
        {
            "input": [
                [0,7,0,7,0,7,0,7],
                [7,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0],
                [7,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0],
                [7,0,0,0,0,0,0,0],
            ],
            "output": [
                [0,7,0,7,0,7,0,7],
                [7,0,0,0,0,0,0,0],
                [0,6,0,6,0,6,0,6],
            ],
        },
        {
            "input": [
                [2,1,2,1,2,2,1,2,1,1],
                [7,2,2,2,2,2,2,2,2,2],
                [2,2,2,2,2,2,2,2,2,2],
                [7,2,2,2,2,2,2,2,2,2],
                [2,2,2,2,2,2,2,2,2,2],
                [7,2,2,2,2,2,2,2,2,2],
                [2,2,2,2,2,2,2,2,2,2],
                [7,2,2,2,2,2,2,2,2,2],
                [2,2,2,2,2,2,2,2,2,2],
                [7,2,2,2,2,2,2,2,2,2],
            ],
            "output": [
                [2,1,2,1,2,2,1,2,1,1],
                [7,2,2,2,2,2,2,2,2,2],
                [2,6,2,6,2,2,6,2,6,6],
            ],
        },
        {
            "input": [
                [1,1,1,4,1,1,1,4,4,1,4,4,1],
                [6,4,4,4,4,4,4,4,4,4,4,4,4],
                [4,4,4,4,4,4,4,4,4,4,4,4,4],
                [6,4,4,4,4,4,4,4,4,4,4,4,4],
                [4,4,4,4,4,4,4,4,4,4,4,4,4],
                [6,4,4,4,4,4,4,4,4,4,4,4,4],
                [4,4,4,4,4,4,4,4,4,4,4,4,4],
                [6,4,4,4,4,4,4,4,4,4,4,4,4],
                [4,4,4,4,4,4,4,4,4,4,4,4,4],
                [6,4,4,4,4,4,4,4,4,4,4,4,4],
                [4,4,4,4,4,4,4,4,4,4,4,4,4],
                [6,4,4,4,4,4,4,4,4,4,4,4,4],
            ],
            "output": [
                [1,1,1,4,1,1,1,4,4,1,4,4,1],
                [6,4,4,4,4,4,4,4,4,4,4,4,4],
                [6,6,6,4,6,6,6,4,4,6,4,4,6],
            ],
        },
        {
            "input": [
                [4,9,4,9,9,4,4,9,9,9,4],
                [7,9,9,9,9,9,9,9,9,9,9],
                [9,9,9,9,9,9,9,9,9,9,9],
                [7,9,9,9,9,9,9,9,9,9,9],
                [9,9,9,9,9,9,9,9,9,9,9],
                [7,9,9,9,9,9,9,9,9,9,9],
                [9,9,9,9,9,9,9,9,9,9,9],
                [7,9,9,9,9,9,9,9,9,9,9],
            ],
            "output": [
                [4,9,4,9,9,4,4,9,9,9,4],
                [7,9,9,9,9,9,9,9,9,9,9],
                [6,9,6,9,9,6,6,9,9,9,6],
            ],
        },
    ]

    test_input = [
        [8,1,8,1,8],
        [4,1,1,1,1],
        [1,1,1,1,1],
    ]
    test_expected = [
        [8,1,8,1,8],
        [4,1,1,1,1],
        [6,1,6,1,6],
    ]

    all_pass = True
    for i, ex in enumerate(examples):
        result = transform(ex["input"])
        ok = result == ex["output"]
        print(f"Example {i}: {'PASS' if ok else 'FAIL'}")
        if not ok:
            print(f"  Expected: {ex['output']}")
            print(f"  Got:      {result}")
            all_pass = False

    test_result = transform(test_input)
    test_ok = test_result == test_expected
    print(f"Test: {'PASS' if test_ok else 'FAIL'}")
    if not test_ok:
        print(f"  Expected: {test_expected}")
        print(f"  Got:      {test_result}")
        all_pass = False

    print("SOLVED" if all_pass else "FAILED")
