"""Solver for ARC-AGI task 8ee62060.

Pattern: A small sprite repeats along a diagonal staircase.
The transformation reverses the vertical order of the staircase blocks.
"""

def solve(grid: list[list[int]]) -> list[list[int]]:
    H = len(grid)
    W = len(grid[0])

    # Find the minimum non-zero column per row to detect block boundaries
    def min_nz_col(row):
        for c, v in enumerate(row):
            if v != 0:
                return c
        return -1

    # Detect block height: rows sharing the same min non-zero column region
    first_col = min_nz_col(grid[0])
    block_height = H  # fallback
    for r in range(1, H):
        if min_nz_col(grid[r]) != first_col:
            block_height = r
            break

    # Reverse the order of blocks
    num_blocks = H // block_height
    result = [[0] * W for _ in range(H)]
    for i in range(num_blocks):
        src_start = i * block_height
        dst_start = (num_blocks - 1 - i) * block_height
        for j in range(block_height):
            result[dst_start + j] = list(grid[src_start + j])

    return result


if __name__ == "__main__":
    import json

    path = "/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/8ee62060.json"
    task = json.load(open(path))

    all_pass = True
    for i, pair in enumerate(task["train"]):
        out = solve(pair["input"])
        ok = out == pair["output"]
        print(f"Train {i}: {'PASS' if ok else 'FAIL'}")
        if not ok:
            all_pass = False
            for r, (got, exp) in enumerate(zip(out, pair["output"])):
                if got != exp:
                    print(f"  Row {r}: got {got} expected {exp}")

    for i, pair in enumerate(task["test"]):
        out = solve(pair["input"])
        if "output" in pair:
            ok = out == pair["output"]
            print(f"Test {i}: {'PASS' if ok else 'FAIL'}")
            if not ok:
                all_pass = False
        else:
            print(f"Test {i}: (no answer key) Output:")
            for row in out:
                print(f"  {row}")

    print(f"\nAll passed: {all_pass}")
