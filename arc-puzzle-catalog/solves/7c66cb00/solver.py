"""
Solver for ARC-AGI task 7c66cb00.

Pattern:
- The grid has a background color, small "stamp" shapes in the background area,
  and horizontal color bands below.
- Each band has a border color (left/right edges) and a fill color (interior).
- Each stamp contains one or two non-background colors.
- For each non-bg color C in a stamp, find the band whose fill == C, then
  bottom-align the bounding box of C-pixels into that band, rendering
  C-pixels as the band's border color (other pixels stay as band fill).
- Stamps are erased from the background area in the output.
"""

import json
from collections import Counter


def solve(grid: list[list[int]]) -> list[list[int]]:
    rows = len(grid)
    cols = len(grid[0])

    bg = grid[0][0]

    # Find horizontal bands (contiguous row groups with non-bg edge color)
    bands = []
    i = 0
    while i < rows:
        if grid[i][0] != bg and grid[i][0] == grid[i][-1]:
            start = i
            while i < rows and grid[i][0] != bg and grid[i][0] == grid[i][-1]:
                i += 1
            bands.append({
                'start': start,
                'end': i - 1,
                'border': grid[start][0],
                'fill': grid[start][1],
            })
        else:
            i += 1

    bg_end = bands[0]['start'] - 1 if bands else rows - 1

    # Find stamps via connected components of non-bg cells in background area
    non_bg_cells = set()
    for r in range(bg_end + 1):
        for c in range(cols):
            if grid[r][c] != bg:
                non_bg_cells.add((r, c))

    visited: set[tuple[int, int]] = set()
    stamps: list[list[tuple[int, int]]] = []
    for r, c in sorted(non_bg_cells):
        if (r, c) in visited:
            continue
        queue = [(r, c)]
        visited.add((r, c))
        component = [(r, c)]
        while queue:
            cr, cc = queue.pop(0)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = cr + dr, cc + dc
                if (nr, nc) in non_bg_cells and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
                    component.append((nr, nc))
        stamps.append(component)

    # Build output
    output = [row[:] for row in grid]

    # Erase stamps from background
    for r in range(bg_end + 1):
        for c in range(cols):
            output[r][c] = bg

    # Place each stamp's color-specific shape into the matching band
    for stamp_pixels in stamps:
        stamp_colors = set(grid[r][c] for r, c in stamp_pixels)

        for color in stamp_colors:
            color_pixels = [(r, c) for r, c in stamp_pixels if grid[r][c] == color]

            min_r = min(r for r, c in color_pixels)
            max_r = max(r for r, c in color_pixels)
            min_c = min(c for r, c in color_pixels)
            max_c = max(c for r, c in color_pixels)

            for band in bands:
                if band['fill'] == color:
                    row_offset = band['end'] - max_r
                    for r in range(min_r, max_r + 1):
                        target_r = r + row_offset
                        if target_r < band['start'] or target_r > band['end']:
                            continue
                        for c in range(min_c, max_c + 1):
                            if grid[r][c] == color:
                                output[target_r][c] = band['border']
                    break

    return output


if __name__ == '__main__':
    with open('/Users/evanpieser/arc-puzzle-catalog/dataset/tasks/7c66cb00.json') as f:
        data = json.load(f)

    all_pass = True
    for i, pair in enumerate(data['train']):
        result = solve(pair['input'])
        ok = result == pair['output']
        print(f'Train {i}: {"PASS" if ok else "FAIL"}')
        if not ok:
            all_pass = False
            for r in range(len(result)):
                for c in range(len(result[0])):
                    if result[r][c] != pair['output'][r][c]:
                        print(f'  Mismatch at ({r},{c}): got {result[r][c]}, expected {pair["output"][r][c]}')

    for i, pair in enumerate(data['test']):
        result = solve(pair['input'])
        if 'output' in pair:
            ok = result == pair['output']
            print(f'Test  {i}: {"PASS" if ok else "FAIL"}')
            if not ok:
                all_pass = False
                for r in range(len(result)):
                    for c in range(len(result[0])):
                        if result[r][c] != pair['output'][r][c]:
                            print(f'  Mismatch at ({r},{c}): got {result[r][c]}, expected {pair["output"][r][c]}')
        else:
            print(f'Test  {i}: (no expected output to compare)')

    print(f'\n{"ALL PASS" if all_pass else "SOME FAILURES"}')
