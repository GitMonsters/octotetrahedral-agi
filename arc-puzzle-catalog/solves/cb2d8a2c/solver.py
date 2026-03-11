"""
ARC-AGI puzzle cb2d8a2c solver.

Pattern: A single 3-marker and multiple line segments (1s and 2s).
- All 1s become 2s in output.
- A continuous frame of 3s is drawn from the marker, zigzagging around each segment.
- For each segment, the number of 1s determines the offset (num_1s + 1).
- The 1s indicate the "open" end; the frame wraps on the opposite side.
- The frame extends perpendicular to the segment by offset, and the turn-row/col
  between marker and each segment is at (segment_pos - offset) toward the marker.
"""
from copy import deepcopy


def transform(grid):
    result = [row[:] for row in grid]
    rows, cols = len(grid), len(grid[0])

    # Find marker (3)
    marker = None
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 3:
                marker = (r, c)
                break
        if marker:
            break
    mr, mc = marker

    # Find segments via BFS on 1/2 cells
    visited = [[False] * cols for _ in range(rows)]
    segments = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] in (1, 2) and not visited[r][c]:
                comp = []
                stack = [(r, c)]
                visited[r][c] = True
                while stack:
                    cr, cc = stack.pop()
                    comp.append((cr, cc))
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] in (1, 2):
                            visited[nr][nc] = True
                            stack.append((nr, nc))
                if len(comp) < 2:
                    continue
                row_set = set(cr for cr, _ in comp)
                col_set = set(cc for _, cc in comp)
                if len(row_set) == 1:
                    segments.append(('h', min(row_set), min(col_set), max(col_set)))
                elif len(col_set) == 1:
                    segments.append(('v', min(col_set), min(row_set), max(row_set)))

    # Determine orientation
    h_segs = [s for s in segments if s[0] == 'h']
    v_segs = [s for s in segments if s[0] == 'v']
    orientation = 'h' if len(h_segs) >= len(v_segs) else 'v'
    segs = h_segs if orientation == 'h' else v_segs

    # Compute per-segment frame data
    seg_data = []
    if orientation == 'h':
        for _, seg_row, seg_left, seg_right in segs:
            ones = [c for c in range(seg_left, seg_right + 1) if grid[seg_row][c] == 1]
            off = len(ones) + 1
            ones_avg = sum(ones) / len(ones)
            mid = (seg_left + seg_right) / 2
            # Frame wraps opposite to where the 1s are concentrated
            if ones_avg >= mid:
                turn_col = seg_left - off      # 1s at right → wrap LEFT
            else:
                turn_col = seg_right + off     # 1s at left → wrap RIGHT
            # Turn row toward the marker
            turn_row = seg_row - off if mr < seg_row else seg_row + off
            seg_data.append({'pos': seg_row, 'tc': turn_col, 'tr': turn_row})
        seg_data.sort(key=lambda s: abs(s['pos'] - mr))
        edge = rows - 1 if mr < min(s['pos'] for s in seg_data) else 0
    else:
        for _, seg_col, seg_top, seg_bot in segs:
            ones = [r for r in range(seg_top, seg_bot + 1) if grid[r][seg_col] == 1]
            off = len(ones) + 1
            ones_avg = sum(ones) / len(ones)
            mid = (seg_top + seg_bot) / 2
            if ones_avg >= mid:
                turn_row = seg_top - off       # 1s at bottom → wrap TOP
            else:
                turn_row = seg_bot + off       # 1s at top → wrap BOTTOM
            # Turn col toward the marker
            turn_col = seg_col - off if mc < seg_col else seg_col + off
            seg_data.append({'pos': seg_col, 'tc': turn_col, 'tr': turn_row})
        seg_data.sort(key=lambda s: abs(s['pos'] - mc))
        edge = cols - 1 if mc < min(s['pos'] for s in seg_data) else 0

    # Draw line helper (only overwrites background=8 cells)
    def draw(r1, c1, r2, c2):
        if r1 == r2:
            for c in range(min(c1, c2), max(c1, c2) + 1):
                if 0 <= c < cols and result[r1][c] == 8:
                    result[r1][c] = 3
        elif c1 == c2:
            for r in range(min(r1, r2), max(r1, r2) + 1):
                if 0 <= r < rows and result[r][c1] == 8:
                    result[r][c1] = 3

    # Draw frame path
    r, c = mr, mc
    if orientation == 'h':
        for s in seg_data:
            draw(r, c, s['tr'], c); r = s['tr']
            draw(r, c, r, s['tc']); c = s['tc']
        draw(r, c, edge, c)
    else:
        for s in seg_data:
            draw(r, c, r, s['tc']); c = s['tc']
            draw(r, c, s['tr'], c); r = s['tr']
        draw(r, c, r, edge)

    # Convert all 1s to 2s
    for r in range(rows):
        for c in range(cols):
            if result[r][c] == 1:
                result[r][c] = 2

    return result


if __name__ == "__main__":
    import json
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI-2/data/evaluation/cb2d8a2c.json"
    with open(path) as f:
        data = json.load(f)

    all_pass = True
    for split in ('train', 'test'):
        for i, ex in enumerate(data[split]):
            inp = ex['input']
            expected = ex['output']
            got = transform(inp)
            ok = got == expected
            if not ok:
                all_pass = False
            print(f"{split.capitalize()} {i}: {'PASS' if ok else 'FAIL'}")
            if not ok:
                for r in range(len(expected)):
                    if expected[r] != got[r]:
                        print(f"  row {r} exp: {expected[r]}")
                        print(f"  row {r} got: {got[r]}")

    print(f"\n{'All passed!' if all_pass else 'Some failed.'}")
