"""ARC-AGI puzzle a6953f00 solver.

Rule: Extract a 2×2 subgrid from the top two rows.
- If grid width is even: extract from the top-RIGHT corner (last 2 columns).
- If grid width is odd: extract from the top-LEFT corner (first 2 columns).
"""


def transform(input_grid: list[list[int]]) -> list[list[int]]:
    cols = len(input_grid[0])
    c = cols - 2 if cols % 2 == 0 else 0
    return [
        [input_grid[0][c], input_grid[0][c + 1]],
        [input_grid[1][c], input_grid[1][c + 1]],
    ]


if __name__ == "__main__":
    examples = [
        {"input": [[7,5,8,2],[8,0,4,7],[1,6,4,7],[8,9,6,9]], "output": [[8,2],[4,7]]},
        {"input": [[0,7,9],[5,6,5],[3,7,9]], "output": [[0,7],[5,6]]},
        {"input": [[5,8,8,9,2],[8,0,5,6,5],[7,7,2,2,9],[5,5,1,7,4],[3,3,8,7,7]], "output": [[5,8],[8,0]]},
    ]
    all_pass = True
    for i, ex in enumerate(examples):
        result = transform(ex["input"])
        ok = result == ex["output"]
        print(f"Example {i}: {'PASS' if ok else 'FAIL'} → {result}")
        if not ok:
            all_pass = False

    test_input = [[3,6,0,2,8,7,9,2],[6,7,5,6,3,7,3,4],[8,0,8,6,3,0,8,3],
                   [8,8,5,9,0,1,6,7],[7,6,9,7,8,7,4,3],[7,3,8,8,3,7,6,1],
                   [3,7,0,7,7,0,5,1],[8,7,5,2,7,7,6,6]]
    print(f"Test output: {transform(test_input)}")
    print("SOLVED" if all_pass else "FAILED")
