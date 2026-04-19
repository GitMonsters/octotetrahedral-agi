from collections import Counter
import math


def _monochrome_components(grid):
    h, w = len(grid), len(grid[0])
    seen = set()
    components = []
    for r in range(h):
        for c in range(w):
            if (r, c) in seen:
                continue
            color = grid[r][c]
            stack = [(r, c)]
            seen.add((r, c))
            cells = []
            while stack:
                x, y = stack.pop()
                cells.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and (nx, ny) not in seen and grid[nx][ny] == color:
                        seen.add((nx, ny))
                        stack.append((nx, ny))
            components.append((color, cells))
    return components


def _infer_blocks(grid):
    dominant = Counter(value for row in grid for value in row).most_common(1)[0][0]
    components = _monochrome_components(grid)

    heights = []
    widths = []
    row_starts = []
    col_starts = []
    for color, cells in components:
        if color == dominant:
            continue
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        heights.append(max(rows) - min(rows) + 1)
        widths.append(max(cols) - min(cols) + 1)
        row_starts.append(min(rows))
        col_starts.append(min(cols))

    block_h = math.gcd(*heights)
    block_w = math.gcd(*widths)

    row_starts = sorted(set(row_starts))
    col_starts = sorted(set(col_starts))
    row_step = math.gcd(*[b - a for a, b in zip(row_starts, row_starts[1:])]) if len(row_starts) > 1 else block_h + 1
    col_step = math.gcd(*[b - a for a, b in zip(col_starts, col_starts[1:])]) if len(col_starts) > 1 else block_w + 1

    coarse_h = (len(grid) - block_h) // row_step + 1
    coarse_w = (len(grid[0]) - block_w) // col_step + 1
    coarse = [[grid[r * row_step][c * col_step] for c in range(coarse_w)] for r in range(coarse_h)]
    return block_h, block_w, row_step, col_step, coarse


def _components(grid, background):
    h, w = len(grid), len(grid[0])
    seen = set()
    components = []
    for r in range(h):
        for c in range(w):
            if (r, c) in seen or grid[r][c] == background:
                continue
            stack = [(r, c)]
            seen.add((r, c))
            cells = []
            while stack:
                x, y = stack.pop()
                cells.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and (nx, ny) not in seen and grid[nx][ny] != background:
                        seen.add((nx, ny))
                        stack.append((nx, ny))
            components.append(cells)
    return components


def transform(grid):
    block_h, block_w, row_step, col_step, coarse = _infer_blocks(grid)
    background = Counter(value for row in coarse for value in row).most_common(1)[0][0]
    parts = _components(coarse, background)

    template = max(parts, key=len)
    markers = [part for part in parts if part != template]
    marker_color = Counter(coarse[r][c] for part in markers for r, c in part).most_common(1)[0][0]
    marker_cells = [part[0] for part in markers if len(part) == 1 and coarse[part[0][0]][part[0][1]] == marker_color]

    candidates = [(r, c) for r, c in template if coarse[r][c] == marker_color]
    anchor_r, anchor_c = min(
        candidates,
        key=lambda cell: sum(abs(cell[0] - r) + abs(cell[1] - c) for r, c in template),
    )

    coarse_out = [row[:] for row in coarse]
    for marker_r, marker_c in marker_cells:
        dr = marker_r - anchor_r
        dc = marker_c - anchor_c
        for r, c in template:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < len(coarse_out) and 0 <= nc < len(coarse_out[0]):
                coarse_out[nr][nc] = coarse[r][c]

    out = [row[:] for row in grid]
    for r, row in enumerate(coarse_out):
        for c, value in enumerate(row):
            if value == coarse[r][c]:
                continue
            raw_r = r * row_step
            raw_c = c * col_step
            for rr in range(raw_r, raw_r + block_h):
                for cc in range(raw_c, raw_c + block_w):
                    out[rr][cc] = value
    return out
