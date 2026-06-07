#!/usr/bin/env python3
"""ARC puzzle ac3e2b04 solver.

Cross pattern rule:
1. Find all 3x3 blocks (bordered by 3s with 2 in center)
2. Detect ALL horizontal 2-lines and vertical 2-lines
3. For each block on a horizontal 2-line: draw vertical 1-cross at center col (0→1)
4. For each block on a vertical 2-line: draw horizontal 1-cross at center row (0→1)
5. For each horizontal 2-line R × each block center col CC:
   draw 3×3 ghost at (R±1, CC±1): all cells → 1, center stays 2; skip real block cells
6. For each vertical 2-line C × each block center row CR:
   draw 3×3 ghost at (CR±1, C±1): all cells → 1, center stays 2; skip real block cells
"""

def solve(grid: list[list[int]]) -> list[list[int]]:
    """Apply the cross/ghost pattern rule."""
    import copy
    result = copy.deepcopy(grid)
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0

    # Find all 3x3 blocks (3-bordered with 2-center)
    blocks = []
    for r in range(rows - 2):
        for c in range(cols - 2):
            if (grid[r][c] == 3 and grid[r][c+1] == 3 and grid[r][c+2] == 3 and
                grid[r+1][c] == 3 and grid[r+1][c+1] == 2 and grid[r+1][c+2] == 3 and
                grid[r+2][c] == 3 and grid[r+2][c+1] == 3 and grid[r+2][c+2] == 3):
                blocks.append((r, c))

    block_cells = set()
    for br, bc in blocks:
        for dr in range(3):
            for dc in range(3):
                block_cells.add((br + dr, bc + dc))

    # block_info: (top_r, top_c, center_r, center_c)
    block_info = [(br, bc, br + 1, bc + 1) for br, bc in blocks]

    # Detect horizontal 2-lines (rows with majority 2s, all cells in {0,2,3})
    def is_h2line(r: int) -> bool:
        return (all(grid[r][c] in (0, 2, 3) for c in range(cols)) and
                sum(1 for c in range(cols) if grid[r][c] == 2) > cols // 2)

    # Detect vertical 2-lines (cols with majority 2s, all cells in {0,2,3})
    def is_v2line(c: int) -> bool:
        return (all(grid[r][c] in (0, 2, 3) for r in range(rows)) and
                sum(1 for r in range(rows) if grid[r][c] == 2) > rows // 2)

    h2lines = [r for r in range(rows) if is_h2line(r)]
    v2lines = [c for c in range(cols) if is_v2line(c)]

    # Draw perpendicular 1-crosses through blocks that sit on 2-lines
    for _br, _bc, cr, cc in block_info:
        if cr in h2lines:
            # Block is on a horizontal 2-line → vertical 1-cross at center col
            for r in range(rows):
                if (r, cc) not in block_cells and grid[r][cc] == 0:
                    result[r][cc] = 1
        if cc in v2lines:
            # Block is on a vertical 2-line → horizontal 1-cross at center row
            for c in range(cols):
                if (cr, c) not in block_cells and grid[cr][c] == 0:
                    result[cr][c] = 1

    # Ghost 3×3 for each horizontal 2-line × each block
    for R in h2lines:
        for _br, _bc, _cr, cc in block_info:
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    gr, gc = R + dr, cc + dc
                    if not (0 <= gr < rows and 0 <= gc < cols):
                        continue
                    if (gr, gc) in block_cells:
                        continue
                    if dr == 0 and dc == 0:
                        pass  # center of ghost stays 2
                    else:
                        result[gr][gc] = 1

    # Ghost 3×3 for each vertical 2-line × each block
    for C in v2lines:
        for _br, _bc, cr, _cc in block_info:
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    gr, gc = cr + dr, C + dc
                    if not (0 <= gr < rows and 0 <= gc < cols):
                        continue
                    if (gr, gc) in block_cells:
                        continue
                    if dr == 0 and dc == 0:
                        pass  # center of ghost stays 2
                    else:
                        result[gr][gc] = 1

    return result


if __name__ == "__main__":
    import json

    # Load the puzzle
    with open("/Users/evanpieser/arc-puzzle-catalog/dataset/tasks/ac3e2b04.json") as f:
        puzzle = json.load(f)

    all_pass = True
    for idx, example in enumerate(puzzle["train"]):
        input_grid = example["input"]
        expected = example["output"]
        result = solve(input_grid)

        passed = result == expected
        all_pass = all_pass and passed

        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"Training example {idx + 1}: {status}")

        if not passed:
            for r in range(len(expected)):
                for c in range(len(expected[0])):
                    if result[r][c] != expected[r][c]:
                        print(f"  Diff at ({r}, {c}): expected {expected[r][c]}, got {result[r][c]}")

    print()
    if all_pass:
        print("All training examples PASSED! ✓")
    else:
        print("Some examples FAILED!")
