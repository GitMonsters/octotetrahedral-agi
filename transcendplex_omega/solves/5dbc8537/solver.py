"""
Solver for ARC puzzle 5dbc8537

The input grid has two regions separated along one axis:
1. TEMPLATE: A shape defined by a border color with interior holes (bg/hole color)
2. CONTENT: Colored rectangular blocks on a background (same color as the holes)

The output is the template region with its holes filled by the content blocks,
matched by dimensions (height × width). Blocks are placed greedily by area (largest first).
"""
import json
from collections import defaultdict


def solve(grid: list[list[int]]) -> list[list[int]]:
    R, C = len(grid), len(grid[0])

    # Find the template and content regions by trying all 4 possible splits
    result = None
    for split_type in ["left", "right", "top", "bottom"]:
        res = try_split(grid, R, C, split_type)
        if res is not None:
            result = res
            break

    if result is None:
        raise ValueError("Could not find valid split")
    return result


def try_split(grid, R, C, split_type):
    """Try a specific split orientation and return the solved output or None."""
    # Find where to split by looking for a region with exactly 2 distinct values
    best = None
    for size in range(2, max(R, C)):
        if split_type == "left" and size < C:
            t_cells = [(r, c) for r in range(R) for c in range(size)]
            c_cells = [(r, c) for r in range(R) for c in range(size, C)]
            out_shape = (R, size)
        elif split_type == "right" and size < C:
            t_cells = [(r, c) for r in range(R) for c in range(C - size, C)]
            c_cells = [(r, c) for r in range(R) for c in range(C - size)]
            out_shape = (R, size)
        elif split_type == "top" and size < R:
            t_cells = [(r, c) for r in range(size) for c in range(C)]
            c_cells = [(r, c) for r in range(size, R) for c in range(C)]
            out_shape = (size, C)
        elif split_type == "bottom" and size < R:
            t_cells = [(r, c) for r in range(R - size, R) for c in range(C)]
            c_cells = [(r, c) for r in range(R - size) for c in range(C)]
            out_shape = (size, C)
        else:
            continue

        t_vals = set(grid[r][c] for r, c in t_cells)
        if len(t_vals) == 2:
            c_vals = set(grid[r][c] for r, c in c_cells)
            shared = t_vals & c_vals
            border_only = t_vals - c_vals
            if len(border_only) == 1 and len(shared) >= 1:
                border_color = list(border_only)[0]
                hole_color = list(shared)[0] if len(shared) == 1 else max(
                    shared, key=lambda v: sum(1 for r, c in t_cells if grid[r][c] == v)
                )
                best = (split_type, size, border_color, hole_color, out_shape, t_cells, c_cells)
                break

    if best is None:
        return None

    split_type, size, border_color, hole_color, out_shape, t_cells, c_cells = best
    oR, oC = out_shape

    # Determine template origin in the grid
    if split_type == "left":
        t_r0, t_c0 = 0, 0
    elif split_type == "right":
        t_r0, t_c0 = 0, C - size
    elif split_type == "top":
        t_r0, t_c0 = 0, 0
    elif split_type == "bottom":
        t_r0, t_c0 = R - size, 0

    # Build the output grid (start with template as-is)
    output = []
    for i in range(oR):
        row = []
        for j in range(oC):
            row.append(grid[t_r0 + i][t_c0 + j])
        output.append(row)

    # Find hole positions in the template (relative to output)
    holes = set()
    for i in range(oR):
        for j in range(oC):
            if output[i][j] == hole_color:
                holes.add((i, j))

    # Extract content blocks from content region
    content_cells = set(c_cells)
    blocks = extract_blocks(grid, content_cells, hole_color)

    # Sort blocks by area (largest first) for efficient backtracking
    blocks.sort(key=lambda b: b[0] * b[1], reverse=True)

    # Use backtracking to find the unique valid tiling
    remaining = set(holes)
    placement = [None] * len(blocks)

    def backtrack(idx):
        if idx == len(blocks):
            return len(remaining) == 0
        h, w, pattern = blocks[idx]
        # Find the top-left-most remaining cell to constrain search
        for i in range(oR):
            for j in range(oC):
                if can_place(remaining, i, j, h, w):
                    # Try placing here
                    placed_cells = set()
                    for di in range(h):
                        for dj in range(w):
                            placed_cells.add((i + di, j + dj))
                    remaining.difference_update(placed_cells)
                    placement[idx] = (i, j)
                    if backtrack(idx + 1):
                        return True
                    remaining.update(placed_cells)
                    placement[idx] = None
        return False

    if not backtrack(0):
        # Try without strict "all holes filled" constraint
        # Just place blocks greedily
        remaining = set(holes)
        for idx, (h, w, pattern) in enumerate(blocks):
            for i in range(oR):
                for j in range(oC):
                    if can_place(remaining, i, j, h, w):
                        for di in range(h):
                            for dj in range(w):
                                remaining.discard((i + di, j + dj))
                        placement[idx] = (i, j)
                        break
                if placement[idx] is not None:
                    break

    # Apply placements to output
    for idx, (h, w, pattern) in enumerate(blocks):
        if placement[idx] is not None:
            pi, pj = placement[idx]
            for di in range(h):
                for dj in range(w):
                    output[pi + di][pj + dj] = pattern[di][dj]

    return output


def can_place(remaining, r, c, h, w):
    """Check if a h×w rectangle at (r,c) fits entirely within remaining holes."""
    for di in range(h):
        for dj in range(w):
            if (r + di, c + dj) not in remaining:
                return False
    return True


def extract_blocks(grid, content_cells, bg_color):
    """Extract rectangular blocks from the content region."""
    R = max(r for r, c in content_cells) + 1
    C = max(c for r, c in content_cells) + 1

    # Find non-bg cells
    non_bg = set()
    for r, c in content_cells:
        if grid[r][c] != bg_color:
            non_bg.add((r, c))

    # Find connected components (4-connected)
    visited = set()
    blocks = []

    for start in non_bg:
        if start in visited:
            continue
        # BFS
        component = set()
        queue = [start]
        while queue:
            cell = queue.pop()
            if cell in visited:
                continue
            visited.add(cell)
            component.add(cell)
            r, c = cell
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (nr, nc) in non_bg and (nr, nc) not in visited:
                    queue.append((nr, nc))

        # Get bounding box
        min_r = min(r for r, c in component)
        max_r = max(r for r, c in component)
        min_c = min(c for r, c in component)
        max_c = max(c for r, c in component)
        h = max_r - min_r + 1
        w = max_c - min_c + 1

        # Extract pattern
        pattern = []
        for i in range(h):
            row = []
            for j in range(w):
                row.append(grid[min_r + i][min_c + j])
            pattern.append(row)

        blocks.append((h, w, pattern))

    return blocks


if __name__ == "__main__":
    with open("/Users/evanpieser/arc-puzzle-catalog/dataset/tasks/5dbc8537.json") as f:
        data = json.load(f)

    all_pass = True
    for split in ["train", "test"]:
        for i, ex in enumerate(data[split]):
            result = solve(ex["input"])
            expected = ex["output"]
            if result == expected:
                print(f"{split} {i}: PASS")
            else:
                print(f"{split} {i}: FAIL")
                all_pass = False
                # Show first difference
                for r in range(min(len(result), len(expected))):
                    for c in range(min(len(result[0]), len(expected[0]))):
                        if result[r][c] != expected[r][c]:
                            print(f"  First diff at ({r},{c}): got {result[r][c]}, expected {expected[r][c]}")
                            break
                    else:
                        continue
                    break
                if len(result) != len(expected) or len(result[0]) != len(expected[0]):
                    print(f"  Size mismatch: got {len(result)}x{len(result[0])}, expected {len(expected)}x{len(expected[0])}")

    print(f"\n{'ALL PASS' if all_pass else 'SOME FAILED'}")
