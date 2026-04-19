"""ARC puzzle 3cd86f4f — diagonal shear transformation."""


def transform(input_grid: list[list[int]]) -> list[list[int]]:
    nrows = len(input_grid)
    output = []
    for i, row in enumerate(input_grid):
        left_pad = nrows - 1 - i
        right_pad = i
        output.append([0] * left_pad + row + [0] * right_pad)
    return output


# --- Verification ---
training = [
    ([[9,7,9,7,7,7,5,5,5],[4,7,9,7,9,7,7,5,5],[4,4,7,7,9,7,9,7,5],[4,4,4,7,7,7,9,7,9]],
     [[0,0,0,9,7,9,7,7,7,5,5,5],[0,0,4,7,9,7,9,7,7,5,5,0],[0,4,4,7,7,9,7,9,7,5,0,0],[4,4,4,7,7,7,9,7,9,0,0,0]]),
    ([[4,8,8,8,8,7],[1,4,8,8,7,8],[4,1,4,7,8,8],[6,4,1,4,8,8],[6,6,4,1,4,8],[6,6,6,4,1,4]],
     [[0,0,0,0,0,4,8,8,8,8,7],[0,0,0,0,1,4,8,8,7,8,0],[0,0,0,4,1,4,7,8,8,0,0],[0,0,6,4,1,4,8,8,0,0,0],[0,6,6,4,1,4,8,0,0,0,0],[6,6,6,4,1,4,0,0,0,0,0]]),
    ([[1,6,6,6],[1,6,6,6],[1,6,6,6],[1,8,8,8],[1,5,5,5],[1,5,5,5],[1,5,5,5]],
     [[0,0,0,0,0,0,1,6,6,6],[0,0,0,0,0,1,6,6,6,0],[0,0,0,0,1,6,6,6,0,0],[0,0,0,1,8,8,8,0,0,0],[0,0,1,5,5,5,0,0,0,0],[0,1,5,5,5,0,0,0,0,0],[1,5,5,5,0,0,0,0,0,0]]),
]

all_pass = True
for idx, (inp, expected) in enumerate(training):
    result = transform(inp)
    ok = result == expected
    print(f"Example {idx}: {'PASS' if ok else 'FAIL'}")
    if not ok:
        all_pass = False
        for r, (got, exp) in enumerate(zip(result, expected)):
            if got != exp:
                print(f"  row {r}: got {got}\n         exp {exp}")

test_input = [[1],[9],[5],[4]]
test_output = transform(test_input)
print(f"\nTest output: {test_output}")
print(f"Expected:    [[0, 0, 0, 1], [0, 0, 9, 0], [0, 5, 0, 0], [4, 0, 0, 0]]")
test_ok = test_output == [[0,0,0,1],[0,0,9,0],[0,5,0,0],[4,0,0,0]]

if all_pass and test_ok:
    print("\nSOLVED")
else:
    print("\nFAILED")
