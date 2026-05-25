from collections import defaultdict


def solve(grid: list[list[int]]) -> list[list[int]]:
    rows = len(grid)
    cols = len(grid[0])

    # Find top-left corners of all 2x2 blocks of 8s
    blocks = []
    for r in range(rows - 1):
        for c in range(cols - 1):
            if (grid[r][c] == 8 and grid[r][c + 1] == 8
                    and grid[r + 1][c] == 8 and grid[r + 1][c + 1] == 8):
                blocks.append((r, c))

    # Group by row — each pair shares the same top-left row
    row_groups = defaultdict(list)
    for r, c in blocks:
        row_groups[r].append(c)

    # For each pair (sorted top-to-bottom), extract content between the blocks
    output = []
    for r in sorted(row_groups.keys()):
        col_pair = sorted(row_groups[r])
        start_c = col_pair[0] + 2   # right edge of left block
        end_c = col_pair[1]          # left edge of right block
        for dr in range(2):
            output.append(grid[r + dr][start_c:end_c])

    return output
