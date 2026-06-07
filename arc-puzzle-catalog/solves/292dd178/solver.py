from collections import Counter


def solve(grid: list[list[int]]) -> list[list[int]]:
    # Rule: each shape is a rectangular frame of 1s with exactly one gap in the
    # border.  Fill the interior with 2, fill the gap with 2, then extend a beam
    # of 2s outward from the gap until a 1 is hit or the grid edge is reached.
    rows, cols = len(grid), len(grid[0])
    flat = [v for row in grid for v in row]
    bg = Counter(flat).most_common(1)[0][0]

    result = [row[:] for row in grid]
    visited = [[False] * cols for _ in range(rows)]

    def get_component(sr: int, sc: int) -> list[tuple[int, int]]:
        stack = [(sr, sc)]
        cells: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            if r < 0 or r >= rows or c < 0 or c >= cols:
                continue
            if visited[r][c] or grid[r][c] == bg:
                continue
            visited[r][c] = True
            cells.append((r, c))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                stack.append((r + dr, c + dc))
        return cells

    for r in range(rows):
        for c in range(cols):
            if not visited[r][c] and grid[r][c] != bg:
                comp = get_component(r, c)
                cell_set = set(comp)
                r_vals = [x for x, _ in comp]
                c_vals = [y for _, y in comp]
                min_r, max_r = min(r_vals), max(r_vals)
                min_c, max_c = min(c_vals), max(c_vals)

                # Identify gaps: border positions of the bounding box that lack a frame cell
                gaps: list[tuple[int, int, str]] = []
                for ci in range(min_c, max_c + 1):
                    if (min_r, ci) not in cell_set:
                        gaps.append((min_r, ci, "top"))
                    if (max_r, ci) not in cell_set:
                        gaps.append((max_r, ci, "bottom"))
                for ri in range(min_r + 1, max_r):
                    if (ri, min_c) not in cell_set:
                        gaps.append((ri, min_c, "left"))
                    if (ri, max_c) not in cell_set:
                        gaps.append((ri, max_c, "right"))

                # Fill interior
                for ri in range(min_r + 1, max_r):
                    for ci in range(min_c + 1, max_c):
                        result[ri][ci] = 2

                # Fill each gap and extend a beam outward until hitting a 1 or edge
                for gr, gc, side in gaps:
                    result[gr][gc] = 2
                    if side == "top":
                        for er in range(gr - 1, -1, -1):
                            if grid[er][gc] != bg:
                                break
                            result[er][gc] = 2
                    elif side == "bottom":
                        for er in range(gr + 1, rows):
                            if grid[er][gc] != bg:
                                break
                            result[er][gc] = 2
                    elif side == "left":
                        for ec in range(gc - 1, -1, -1):
                            if grid[gr][ec] != bg:
                                break
                            result[gr][ec] = 2
                    elif side == "right":
                        for ec in range(gc + 1, cols):
                            if grid[gr][ec] != bg:
                                break
                            result[gr][ec] = 2

    return result