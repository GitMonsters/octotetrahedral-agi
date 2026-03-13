"""
Solver for ARC-AGI task 692cd3b6.

Pattern: Two 3x3 boxes (border of 2s, center 5, one gap where a 2 is 0).
Each gap indicates a direction (which face is open). A rectangle of 4s
connects the two boxes through their gaps. The box "shadows" (blocks) 4s
from appearing behind it (opposite the gap direction) within the rectangle.
"""

import json
from typing import List

Grid = List[List[int]]


def solve(grid: Grid) -> Grid:
    rows = len(grid)
    cols = len(grid[0])
    result = [row[:] for row in grid]

    # Find box centers (cells with value 5)
    centers = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 5:
                centers.append((r, c))
    assert len(centers) == 2

    boxes = []
    for cr, cc in centers:
        bt, bb = cr - 1, cr + 1  # box top/bottom rows
        bl, br = cc - 1, cc + 1  # box left/right cols

        # Find gap (0 cell at an edge center of the 3x3 border)
        gap = None
        gap_dir = None
        for dr, dc, d in [(-1, 0, 'UP'), (1, 0, 'DOWN'), (0, -1, 'LEFT'), (0, 1, 'RIGHT')]:
            nr, nc = cr + dr, cc + dc
            if grid[nr][nc] == 0:
                gap = (nr, nc)
                gap_dir = d
                break
        assert gap is not None

        # Exit point: one cell outside the gap in the gap direction
        gr, gc = gap
        exit_deltas = {'UP': (-1, 0), 'DOWN': (1, 0), 'LEFT': (0, -1), 'RIGHT': (0, 1)}
        edr, edc = exit_deltas[gap_dir]
        exit_pt = (gr + edr, gc + edc)

        boxes.append({
            'center': (cr, cc),
            'top': bt, 'bottom': bb, 'left': bl, 'right': br,
            'gap': gap, 'gap_dir': gap_dir, 'exit': exit_pt
        })

    # Main rectangle: bounding box of the two exit points
    e1, e2 = boxes[0]['exit'], boxes[1]['exit']
    rect_top = min(e1[0], e2[0])
    rect_bot = max(e1[0], e2[0])
    rect_left = min(e1[1], e2[1])
    rect_right = max(e1[1], e2[1])

    # Compute shadow regions — each box blocks 4s behind it (opposite gap dir)
    shadow = set()
    for box in boxes:
        gd = box['gap_dir']
        bt, bb, bl, br = box['top'], box['bottom'], box['left'], box['right']

        if gd == 'UP':
            # Shadow goes DOWN from below the box
            sr_start, sr_end = bb + 1, rect_bot
            sc_start = max(bl, rect_left)
            sc_end = min(br, rect_right)
        elif gd == 'DOWN':
            # Shadow goes UP from above the box
            sr_start, sr_end = rect_top, bt - 1
            sc_start = max(bl, rect_left)
            sc_end = min(br, rect_right)
        elif gd == 'LEFT':
            # Shadow goes RIGHT from right of the box
            sc_start, sc_end = br + 1, rect_right
            sr_start = max(bt, rect_top)
            sr_end = min(bb, rect_bot)
        elif gd == 'RIGHT':
            # Shadow goes LEFT from left of the box
            sc_start, sc_end = rect_left, bl - 1
            sr_start = max(bt, rect_top)
            sr_end = min(bb, rect_bot)

        for r in range(sr_start, sr_end + 1):
            for c in range(sc_start, sc_end + 1):
                shadow.add((r, c))

    # Fill rectangle with 4s (only overwrite 0s, skip shadows)
    for r in range(rect_top, rect_bot + 1):
        for c in range(rect_left, rect_right + 1):
            if result[r][c] == 0 and (r, c) not in shadow:
                result[r][c] = 4

    # Fill gap cells with 4
    for box in boxes:
        gr, gc = box['gap']
        result[gr][gc] = 4

    return result


if __name__ == '__main__':
    task = json.load(open('/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/692cd3b6.json'))

    all_pass = True
    for i, pair in enumerate(task['train']):
        predicted = solve(pair['input'])
        expected = pair['output']
        match = predicted == expected
        all_pass = all_pass and match
        if not match:
            print(f"Train {i}: FAIL")
            for r in range(len(expected)):
                for c in range(len(expected[0])):
                    if predicted[r][c] != expected[r][c]:
                        print(f"  ({r},{c}): got {predicted[r][c]}, expected {expected[r][c]}")
        else:
            print(f"Train {i}: PASS")

    for i, pair in enumerate(task['test']):
        predicted = solve(pair['input'])
        print(f"\nTest {i} output:")
        for row in predicted:
            print(row)
        if 'output' in pair:
            match = predicted == pair['output']
            print(f"Test {i}: {'PASS' if match else 'FAIL'}")
            all_pass = all_pass and match

    print(f"\n{'ALL PASS' if all_pass else 'SOME FAILED'}")
