"""
67c52801 solver

Rule:
1. The last row is the "floor" (all same non-zero color).
2. The second-to-last row is the "gap row" (same color with 0-gaps).
3. Colored blocks (non-zero, non-floor-color rectangles) float above.
4. Each block drops into the gap whose width = block_area / 2.
5. The block becomes a 2-row-tall solid rectangle filling the gap,
   sitting at (gap_row - 1) and (gap_row).
6. Everything above is cleared to 0.
"""
from typing import List


def solve(grid: List[List[int]]) -> List[List[int]]:
    R = len(grid)
    C = len(grid[0])

    # Floor = last row (all same non-zero value)
    floor_row = R - 1
    floor_color = grid[floor_row][0]

    # Gap row = second-to-last row
    gap_row = floor_row - 1

    # Find gaps (contiguous 0s in gap row)
    gaps = []
    c = 0
    while c < C:
        if grid[gap_row][c] == 0:
            start = c
            while c < C and grid[gap_row][c] == 0:
                c += 1
            gaps.append((start, c - start))  # (start_col, width)
        else:
            c += 1

    # Find blocks: non-zero, non-floor-color connected rectangular regions
    visited = [[False] * C for _ in range(R)]
    blocks = []  # list of (color, height, width)

    for r in range(R):
        for c in range(C):
            val = grid[r][c]
            if val != 0 and val != floor_color and not visited[r][c]:
                # BFS to find bounding box of this color block
                color = val
                min_r, max_r, min_c, max_c = r, r, c, c
                stack = [(r, c)]
                visited[r][c] = True
                while stack:
                    cr, cc = stack.pop()
                    min_r = min(min_r, cr)
                    max_r = max(max_r, cr)
                    min_c = min(min_c, cc)
                    max_c = max(max_c, cc)
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < R and 0 <= nc < C and not visited[nr][nc] and grid[nr][nc] == color:
                            visited[nr][nc] = True
                            stack.append((nr, nc))
                h = max_r - min_r + 1
                w = max_c - min_c + 1
                blocks.append((color, h, w))

    # Match blocks to gaps: block_area / 2 = gap_width
    gap_by_width = {width: start for start, width in gaps}

    # Build output: start with all zeros except floor rows
    out = [[0] * C for _ in range(R)]
    # Keep floor row
    out[floor_row] = grid[floor_row][:]
    # Keep gap row structure (floor color cells)
    out[gap_row] = grid[gap_row][:]

    # Place blocks
    for color, h, w in blocks:
        area = h * w
        target_width = area // 2
        gap_start = gap_by_width[target_width]
        # Fill 2 rows: gap_row and gap_row-1
        for c in range(gap_start, gap_start + target_width):
            out[gap_row][c] = color
            out[gap_row - 1][c] = color

    return out


if __name__ == "__main__":
    import json
    with open("/Users/evanpieser/arc-puzzle-catalog/dataset/tasks/67c52801.json") as f:
        data = json.load(f)
    for phase in ["train", "test"]:
        for i, example in enumerate(data[phase]):
            result = solve(example["input"])
            expected = example["output"]
            status = "PASS" if result == expected else "FAIL"
            print(f"{phase} {i}: {status}")
            if status == "FAIL":
                for r in range(len(expected)):
                    if result[r] != expected[r]:
                        print(f"  row {r}: got {result[r]}")
                        print(f"          exp {expected[r]}")
