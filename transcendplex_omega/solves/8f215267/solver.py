"""Solver for ARC-AGI task 8f215267.

Pattern:
- Grid has a background color and several rectangular box frames of different colors.
- Small clusters of non-background colors are scattered outside the boxes.
- For each box of color C, count distinct connected components of color C outside all boxes.
- Place that many dots of color C in the middle row of the box's interior,
  at every-other-column positions starting from the right.
- All external clusters are removed in the output.
"""

from collections import Counter, deque


def solve(grid: list[list[int]]) -> list[list[int]]:
    H = len(grid)
    W = len(grid[0])

    # Background = most common color
    bg = Counter(grid[r][c] for r in range(H) for c in range(W)).most_common(1)[0][0]

    # BFS to find connected components of non-background cells
    visited = [[False] * W for _ in range(H)]
    components: list[tuple[int, set[tuple[int, int]]]] = []

    for r in range(H):
        for c in range(W):
            if grid[r][c] != bg and not visited[r][c]:
                color = grid[r][c]
                queue = deque([(r, c)])
                visited[r][c] = True
                cells: set[tuple[int, int]] = set()
                while queue:
                    cr, cc = queue.popleft()
                    cells.add((cr, cc))
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < H and 0 <= nc < W and not visited[nr][nc] and grid[nr][nc] == color:
                            visited[nr][nc] = True
                            queue.append((nr, nc))
                components.append((color, cells))

    # Classify each component as a rectangular frame (box) or a cluster
    boxes: list[tuple[int, int, int, int, int]] = []  # (color, top, bottom, left, right)
    cluster_count: Counter = Counter()

    for color, cells in components:
        min_r = min(r for r, _ in cells)
        max_r = max(r for r, _ in cells)
        min_c = min(c for _, c in cells)
        max_c = max(c for _, c in cells)

        # Build the set of cells expected for a rectangular frame
        expected: set[tuple[int, int]] = set()
        for c2 in range(min_c, max_c + 1):
            expected.add((min_r, c2))
            expected.add((max_r, c2))
        for r2 in range(min_r + 1, max_r):
            expected.add((r2, min_c))
            expected.add((r2, max_c))

        if cells == expected and (max_r - min_r >= 2) and (max_c - min_c >= 2):
            boxes.append((color, min_r, max_r, min_c, max_c))
        else:
            cluster_count[color] += 1

    # Build output: background everywhere, then draw boxes with interior dots
    output = [[bg] * W for _ in range(H)]

    for color, top, bottom, left, right in boxes:
        # Draw frame border
        for c2 in range(left, right + 1):
            output[top][c2] = color
            output[bottom][c2] = color
        for r2 in range(top + 1, bottom):
            output[r2][left] = color
            output[r2][right] = color

        # Interior bounds
        int_left = left + 1
        int_right = right - 1
        int_top = top + 1
        int_bottom = bottom - 1
        int_height = int_bottom - int_top + 1
        mid_row = int_top + int_height // 2

        # Place N dots from the right, every other column
        n_dots = cluster_count.get(color, 0)
        for i in range(n_dots):
            col = int_right - 1 - 2 * i
            if col >= int_left:
                output[mid_row][col] = color

    return output


if __name__ == "__main__":
    import json, pathlib

    task_path = pathlib.Path(__file__).resolve().parent.parent.parent / "dataset" / "tasks" / "8f215267.json"
    with open(task_path) as f:
        task = json.load(f)

    all_pass = True
    for i, pair in enumerate(task["train"]):
        result = solve(pair["input"])
        status = "PASS" if result == pair["output"] else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"Train pair {i}: {status}")

    for i, pair in enumerate(task.get("test", [])):
        if "output" in pair:
            result = solve(pair["input"])
            status = "PASS" if result == pair["output"] else "FAIL"
            if status == "FAIL":
                all_pass = False
            print(f"Test pair {i}: {status}")
        else:
            print(f"Test pair {i}: (no expected output to verify)")

    if all_pass:
        print("\nAll pairs PASSED!")
