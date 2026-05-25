import json


def solve(grid):
    rows = len(grid)
    cols = len(grid[0])

    r2, c2 = None, None
    ones = set()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2:
                r2, c2 = r, c
            elif grid[r][c] == 1:
                ones.add((r, c))

    out = [[0] * cols for _ in range(rows)]
    for r, c in ones:
        out[r][c] = 1
    out[r2][c2] = 2

    cumulative_shift = 0
    prev_right = c2

    for step in range(1, rows):
        row = r2 - step
        if row < 0:
            break

        left = c2 + step - 1 + cumulative_shift
        right = c2 + step + cumulative_shift

        if left >= cols:
            break

        # Shift pair right past any 1-obstacles
        shift = 0
        while True:
            blocked = False
            for c in range(left + shift, min(right + shift + 1, cols)):
                if (row, c) in ones:
                    blocked = True
                    break
            if not blocked:
                break
            shift += 1
            if left + shift >= cols:
                break

        if left + shift >= cols:
            break

        if shift > 0:
            new_left = left + shift
            conn_row = row + 1
            # Bridge the connection row from previous right to new left
            for c in range(prev_right + 1, new_left + 1):
                if 0 <= c < cols and out[conn_row][c] != 1:
                    out[conn_row][c] = 2
            cumulative_shift += shift
            left += shift
            right += shift

        if left < cols:
            out[row][left] = 2
        if right < cols:
            out[row][right] = 2

        prev_right = right if right < cols else left

        if left >= cols:
            break

    return out


if __name__ == "__main__":
    with open("/Users/evanpieser/arc-puzzle-catalog/dataset/tasks/69889d6e.json") as f:
        task = json.load(f)

    all_pass = True
    for i, example in enumerate(task["train"]):
        result = solve(example["input"])
        if result == example["output"]:
            print(f"Train {i}: PASS")
        else:
            print(f"Train {i}: FAIL")
            for r, (got, exp) in enumerate(zip(result, example["output"])):
                if got != exp:
                    print(f"  Row {r}: got {got}")
                    print(f"       exp {exp}")
            all_pass = False

    for i, example in enumerate(task["test"]):
        result = solve(example["input"])
        if result == example["output"]:
            print(f"Test  {i}: PASS")
        else:
            print(f"Test  {i}: FAIL")
            for r, (got, exp) in enumerate(zip(result, example["output"])):
                if got != exp:
                    print(f"  Row {r}: got {got}")
                    print(f"       exp {exp}")
            all_pass = False

    print(f"\n{'ALL PASS' if all_pass else 'SOME FAILED'}")
