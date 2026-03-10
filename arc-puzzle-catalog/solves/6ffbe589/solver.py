"""
Solver for ARC-AGI task 6ffbe589.

Rule: The grid contains a multi-layered pattern and isolated "hint" pixels.
Each color layer in the pattern is independently rotated by (hint_count × 90°) CW,
where hint_count is the number of hint pixels of that color found outside the pattern.
"""

from collections import deque


def solve(grid: list[list[int]]) -> list[list[int]]:
    rows, cols = len(grid), len(grid[0])

    # Find 8-connected components of non-zero cells
    visited = [[False] * cols for _ in range(rows)]
    components: list[list[tuple[int, int]]] = []
    for sr in range(rows):
        for sc in range(cols):
            if grid[sr][sc] != 0 and not visited[sr][sc]:
                q = deque([(sr, sc)])
                visited[sr][sc] = True
                cells = [(sr, sc)]
                while q:
                    r, c = q.popleft()
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            if dr == 0 and dc == 0:
                                continue
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] != 0:
                                visited[nr][nc] = True
                                q.append((nr, nc))
                                cells.append((nr, nc))
                components.append(cells)

    # Find the pattern region: start from the largest component's bbox,
    # expand by a margin, and include all non-zero cells within that region.
    # Hint pixels are always far from the main pattern, so they stay outside.
    MARGIN = 3
    components.sort(key=lambda c: -len(c))
    main_comp = components[0]
    pr1 = min(r for r, c in main_comp)
    pr2 = max(r for r, c in main_comp)
    pc1 = min(c for r, c in main_comp)
    pc2 = max(c for r, c in main_comp)

    er1 = max(0, pr1 - MARGIN)
    er2 = min(rows - 1, pr2 + MARGIN)
    ec1 = max(0, pc1 - MARGIN)
    ec2 = min(cols - 1, pc2 + MARGIN)

    # Collect non-zero cells in the expanded region
    pattern_cells: set[tuple[int, int]] = set()
    for r in range(er1, er2 + 1):
        for c in range(ec1, ec2 + 1):
            if grid[r][c] != 0:
                pattern_cells.add((r, c))

    # Pattern bounding box (tight fit around discovered cells)
    min_r = min(r for r, c in pattern_cells)
    max_r = max(r for r, c in pattern_cells)
    min_c = min(c for r, c in pattern_cells)
    max_c = max(c for r, c in pattern_cells)
    R = max_r - min_r + 1
    C = max_c - min_c + 1

    # Extract the pattern
    pattern = [[grid[min_r + r][min_c + c] for c in range(C)] for r in range(R)]

    # Count hint pixels per color (non-zero cells outside the bounding box)
    hints: dict[int, int] = {}
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0 and (r < min_r or r > max_r or c < min_c or c > max_c):
                color = grid[r][c]
                hints[color] = hints.get(color, 0) + 1

    # Collect positions per color
    color_positions: dict[int, list[tuple[int, int]]] = {}
    for r in range(R):
        for c in range(C):
            if pattern[r][c] != 0:
                color_positions.setdefault(pattern[r][c], []).append((r, c))

    # Rotate each layer and assemble the result
    result = [[0] * C for _ in range(R)]
    for color, positions in color_positions.items():
        rot_count = hints.get(color, 0) % 4
        rotated = positions
        for _ in range(rot_count):
            rotated = [(c, R - 1 - r) for r, c in rotated]
        for r, c in rotated:
            result[r][c] = color

    return result


if __name__ == "__main__":
    import json, os

    task_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "dataset", "tasks", "6ffbe589.json"
    )
    with open(task_path) as f:
        data = json.load(f)

    all_pass = True
    for i, pair in enumerate(data["train"] + data.get("test", [])):
        label = f"train[{i}]" if i < len(data["train"]) else f"test[{i - len(data['train'])}]"
        output = solve(pair["input"])
        expected = pair["output"]
        match = output == expected
        print(f"{label}: {'PASS' if match else 'FAIL'}")
        if not match:
            all_pass = False
            for r in range(max(len(output), len(expected))):
                if r < len(output) and r < len(expected) and output[r] != expected[r]:
                    print(f"  row {r}: got {output[r]}")
                    print(f"       exp {expected[r]}")

    print(f"\n{'ALL PASS' if all_pass else 'SOME FAILED'}")
