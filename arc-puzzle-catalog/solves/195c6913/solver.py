"""Solver for ARC puzzle 195c6913 — Staircase Snake Pattern Propagation.

Rule: Two background colors form diagonal regions with a dark band between them.
A template row of 2x2 blocks defines a repeating color sequence. Single marker
cells on the grid edge launch snake paths through the dark band, alternating
RIGHT then UP, filling dark cells with the repeating pattern and placing a
terminator color at each boundary crossing.
"""

from collections import Counter


def solve(grid):
    H, W = len(grid), len(grid[0])
    out = [row[:] for row in grid]

    cnt = Counter(v for row in grid for v in row)
    bg1, bg2 = cnt.most_common(2)[0][0], cnt.most_common(2)[1][0]
    light = grid[0][0]
    dark = bg2 if bg1 == light else bg1

    # Find all 2x2 blocks of rare colors
    blocks = []
    used = set()
    for r in range(H - 1):
        for c in range(W - 1):
            v = grid[r][c]
            if v in (bg1, bg2) or (r, c) in used:
                continue
            if grid[r][c+1] == v and grid[r+1][c] == v and grid[r+1][c+1] == v:
                blocks.append((v, r, c))
                used.update([(r,c), (r,c+1), (r+1,c), (r+1,c+1)])

    # Template: row with most 2x2 blocks (near corner)
    by_row = {}
    for v, r, c in blocks:
        by_row.setdefault(r, []).append((v, r, c))
    template_row = max(by_row, key=lambda r: len(by_row[r]))
    template_blocks = sorted(by_row[template_row], key=lambda x: x[2])

    template_cells = set()
    for v, r, c in template_blocks:
        template_cells.update([(r,c), (r,c+1), (r+1,c), (r+1,c+1)])

    # Terminator: 2x2 block not in template
    term_color = None
    term_block = None
    for v, r, c in blocks:
        cells = {(r,c), (r,c+1), (r+1,c), (r+1,c+1)}
        if not cells & template_cells:
            term_block = (v, r, c)
            term_color = v
            break

    pattern = [v for v, _, _ in template_blocks]

    # Find markers: single cells of first template color on grid edge
    marker_color = pattern[0]
    markers = []
    for r in range(H):
        for c in range(W):
            if grid[r][c] == marker_color and (r, c) not in template_cells:
                if r == 0 or r == H - 1 or c == 0 or c == W - 1:
                    markers.append((r, c))

    # Erase template and terminator blocks
    for v, r, c in template_blocks:
        out[r][c] = out[r][c+1] = out[r+1][c] = out[r+1][c+1] = light
    if term_block:
        _, r, c = term_block
        out[r][c] = out[r][c+1] = out[r+1][c] = out[r+1][c+1] = light

    # Trace snake path from each marker
    for mr, mc in markers:
        seq_idx = 1  # marker cell = position 0, stays unchanged

        # Determine directions based on which edge the marker is on
        if mc == 0:
            dirs = [(0, 1), (-1, 0)]     # RIGHT, UP
        elif mc == W - 1:
            dirs = [(0, -1), (-1, 0)]    # LEFT, UP
        elif mr == 0:
            dirs = [(1, 0), (0, 1)]      # DOWN, RIGHT
        else:
            dirs = [(-1, 0), (0, 1)]     # UP, RIGHT

        cr, cc = mr, mc
        di = 0

        while True:
            dr, dc = dirs[di % 2]
            filled = False

            while True:
                nr, nc = cr + dr, cc + dc
                if nr < 0 or nr >= H or nc < 0 or nc >= W:
                    break
                if grid[nr][nc] != dark:
                    if filled:
                        out[nr][nc] = term_color
                    break
                out[nr][nc] = pattern[seq_idx % len(pattern)]
                seq_idx += 1
                cr, cc = nr, nc
                filled = True

            # Stop if grid edge or no dark cells filled before boundary
            if nr < 0 or nr >= H or nc < 0 or nc >= W:
                break
            if not filled:
                break

            di += 1

    return out


def validate():
    import json
    with open('/Users/evanpieser/arc-puzzle-catalog/dataset/tasks/195c6913.json') as f:
        data = json.load(f)

    ok = True
    for idx, pair in enumerate(data['train']):
        result = solve(pair['input'])
        expected = pair['output']
        diffs = sum(1 for r in range(len(expected)) for c in range(len(expected[0]))
                    if result[r][c] != expected[r][c])
        status = '✅' if diffs == 0 else '❌'
        print(f'Train {idx}: {status} {diffs} diffs')
        if diffs > 0:
            ok = False

    for idx, pair in enumerate(data['test']):
        result = solve(pair['input'])
        expected = pair['output']
        diffs = sum(1 for r in range(len(expected)) for c in range(len(expected[0]))
                    if result[r][c] != expected[r][c])
        status = '✅' if diffs == 0 else '❌'
        print(f'Test  {idx}: {status} {diffs} diffs')
        if diffs > 0:
            ok = False
            for r in range(len(expected)):
                for c in range(len(expected[0])):
                    if result[r][c] != expected[r][c]:
                        print(f'  ({r},{c}): got {result[r][c]}, expected {expected[r][c]}')

    return ok


if __name__ == '__main__':
    validate()
