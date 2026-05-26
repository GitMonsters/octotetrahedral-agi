"""
ARC Task 5d9d1ff1 Solver

Rule (verified on both train pairs):
  1. Background = most common color in the grid.
  2. Each non-background row has a right-aligned solid-color segment.
     Detect segment length = number of non-bg cells in that row.
  3. Group rows by segment length, sort lengths ascending → rank 1, 2, 3+.
  4. Assign output colors by rank:
       rank 1 (shortest) → 4
       rank 2             → 3
       rank 3+            → 7
  5. Rebuild: bg cells stay bg; every non-bg cell in a row takes its group's
     assigned output color.

NOTE: Output colors {3, 4, 7} are an empirically observed fixed property of
this puzzle type, consistent across both training examples.
"""

from collections import Counter


def solve(grid: list[list[int]]) -> list[list[int]]:
    R = len(grid)
    C = len(grid[0])

    # Step 1: detect background
    flat = [cell for row in grid for cell in row]
    bg = Counter(flat).most_common(1)[0][0]

    # Step 2: for each non-bg row, record the segment length
    row_seg_len: dict[int, int] = {}
    for r in range(R):
        non_bg = [c for c in range(C) if grid[r][c] != bg]
        if non_bg:
            row_seg_len[r] = len(non_bg)

    # Step 3: sort distinct lengths ascending → determine rank
    distinct_lens = sorted(set(row_seg_len.values()))

    # Step 4: assign output colors by rank
    # rank 1 (shortest) → 4, rank 2 → 3, rank 3+ → 7
    rank_color: dict[int, int] = {}
    for rank, length in enumerate(distinct_lens):
        if rank == 0:
            rank_color[length] = 4
        elif rank == 1:
            rank_color[length] = 3
        else:
            rank_color[length] = 7

    # Step 5: build output grid
    out = [row[:] for row in grid]
    for r, seg_len in row_seg_len.items():
        assigned = rank_color[seg_len]
        for c in range(C):
            if grid[r][c] != bg:
                out[r][c] = assigned

    return out


if __name__ == "__main__":
    import json

    with open("/tmp/v17_failed/5d9d1ff1.json") as f:
        data = json.load(f)["5d9d1ff1"]

    all_pass = True
    for i, pair in enumerate(data["train"]):
        inp = pair["input"]
        expected = pair["output"]
        got = solve(inp)
        if got == expected:
            print(f"Train {i}: PASS")
        else:
            all_pass = False
            print(f"Train {i}: FAIL")
            for r in range(len(expected)):
                if expected[r] != got[r]:
                    print(f"  row {r} expected: {expected[r]}")
                    print(f"  row {r} got:      {got[r]}")

    if all_pass:
        print("\nVERIFIED")
    else:
        print("\nUNSOLVED")

    print("\n=== TEST PREDICTIONS ===")
    for i, pair in enumerate(data["test"]):
        inp = pair["input"]
        flat = [cell for row in inp for cell in row]
        bg = Counter(flat).most_common(1)[0][0]
        result = solve(inp)
        row_lens = {}
        for r in range(len(inp)):
            non_bg = [c for c in range(len(inp[0])) if inp[r][c] != bg]
            if non_bg:
                row_lens[r] = len(non_bg)
        distinct = sorted(set(row_lens.values()))
        print(f"Test {i}: bg={bg}, segment lengths (sorted)={distinct}")
        print(f"  → rank1(len={distinct[0]})→4, rank2(len={distinct[1]})→3, rank3+→7")
