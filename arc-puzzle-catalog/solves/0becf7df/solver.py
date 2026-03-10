import json, sys

def solve(grid):
    rows = len(grid)
    cols = len(grid[0])
    result = [row[:] for row in grid]

    # 2x2 key: same-row elements swap
    mapping = {
        grid[0][0]: grid[0][1],
        grid[0][1]: grid[0][0],
        grid[1][0]: grid[1][1],
        grid[1][1]: grid[1][0],
    }

    for r in range(rows):
        for c in range(cols):
            if r < 2 and c < 2:
                continue
            if grid[r][c] != 0 and grid[r][c] in mapping:
                result[r][c] = mapping[grid[r][c]]

    return result

if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        task = json.load(f)
    pairs = task.get("train", []) + task.get("test", [])
    ok = 0
    for i, pair in enumerate(pairs):
        pred = solve(pair["input"])
        if pred == pair["output"]:
            ok += 1
            print(f"Pair {i}: PASS")
        else:
            print(f"Pair {i}: FAIL")
    print(f"{ok}/{len(pairs)} correct")
