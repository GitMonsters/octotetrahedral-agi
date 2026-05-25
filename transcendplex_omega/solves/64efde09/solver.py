"""Solver for ARC-AGI puzzle 64efde09.

Each grid contains several 2-wide "bars" (vertical or horizontal) and isolated
single-cell "markers". Each bar has a "cap" end identifiable by transition
patterns, and an "open" side determined by which track's transition value matches
its body value. Markers sit in the extension of a bar's open direction and map a
structural slot (distance from cap) to a fill color. All bars share the same
slot->color mapping, projecting colored lines from the bar to the grid edge.
"""
import json
import sys
from collections import deque

BG = 8


def solve(grid: list[list[int]]) -> list[list[int]]:
    rows = len(grid)
    cols = len(grid[0])

    # Find connected components of non-background cells
    non_bg = set()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != BG:
                non_bg.add((r, c))

    visited: set[tuple[int, int]] = set()
    components: list[set[tuple[int, int]]] = []
    for cell in non_bg:
        if cell in visited:
            continue
        comp: set[tuple[int, int]] = set()
        q = deque([cell])
        while q:
            cur = q.popleft()
            if cur in visited:
                continue
            visited.add(cur)
            comp.add(cur)
            cr, cc = cur
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nb = (cr + dr, cc + dc)
                if nb in non_bg and nb not in visited:
                    q.append(nb)
        components.append(comp)

    # Classify into bars (2-wide rectangles) and markers (single cells)
    bars: list[dict] = []
    markers: list[tuple[int, int]] = []
    for comp in components:
        if len(comp) == 1:
            markers.append(next(iter(comp)))
            continue
        rs = [r for r, c in comp]
        cs = [c for r, c in comp]
        min_r, max_r = min(rs), max(rs)
        min_c, max_c = min(cs), max(cs)
        h = max_r - min_r + 1
        w = max_c - min_c + 1
        if h * w == len(comp) and ((h == 2 and w > 2) or (w == 2 and h > 2)):
            bars.append({
                'min_r': min_r, 'max_r': max_r,
                'min_c': min_c, 'max_c': max_c,
                'orient': 'h' if h == 2 else 'v',
            })

    # Analyse each bar: find cap end and open (fill) direction
    for bar in bars:
        orient = bar['orient']
        if orient == 'v':
            c1, c2 = bar['min_c'], bar['max_c']
            rs, re = bar['min_r'], bar['max_r']
            tA = [grid[r][c1] for r in range(rs, re + 1)]
            tB = [grid[r][c2] for r in range(rs, re + 1)]
        else:
            r1, r2 = bar['min_r'], bar['max_r']
            cs, ce = bar['min_c'], bar['max_c']
            tA = [grid[r1][c] for c in range(cs, ce + 1)]
            tB = [grid[r2][c] for c in range(cs, ce + 1)]

        n = len(tA)
        cap_idx = _find_cap(tA, tB, n)

        # Transition is position 1 from cap; body is position 2
        if cap_idx == 0:
            ti, bi = 1, 2
        else:
            ti, bi = n - 2, n - 3

        # Wall track: the one whose transition value == body value
        wall_is_A = (tA[ti] == tA[bi])

        if orient == 'v':
            bar['open'] = 'left' if wall_is_A else 'right'
            bar['cap'] = rs if cap_idx == 0 else re
        else:
            bar['open'] = 'up' if wall_is_A else 'down'
            bar['cap'] = cs if cap_idx == 0 else ce

    # Associate each marker with a bar; build slot->color map
    slot_map: dict[int, int] = {}
    for mr, mc in markers:
        color = grid[mr][mc]
        for bar in bars:
            matched = False
            if bar['orient'] == 'v':
                if not (bar['min_r'] <= mr <= bar['max_r']):
                    continue
                if bar['open'] == 'left' and mc < bar['min_c']:
                    matched = True
                elif bar['open'] == 'right' and mc > bar['max_c']:
                    matched = True
                if matched:
                    slot_map[abs(mr - bar['cap'])] = color
                    break
            else:
                if not (bar['min_c'] <= mc <= bar['max_c']):
                    continue
                if bar['open'] == 'up' and mr < bar['min_r']:
                    matched = True
                elif bar['open'] == 'down' and mr > bar['max_r']:
                    matched = True
                if matched:
                    slot_map[abs(mc - bar['cap'])] = color
                    break

    # Apply fills
    out = [row[:] for row in grid]
    for bar in bars:
        for pos, color in slot_map.items():
            if bar['orient'] == 'v':
                cap = bar['cap']
                fill_r = (cap + pos) if cap == bar['min_r'] else (cap - pos)
                if not (bar['min_r'] <= fill_r <= bar['max_r']):
                    continue
                if bar['open'] == 'left':
                    for c in range(bar['min_c'] - 1, -1, -1):
                        if out[fill_r][c] == BG:
                            out[fill_r][c] = color
                else:
                    for c in range(bar['max_c'] + 1, cols):
                        if out[fill_r][c] == BG:
                            out[fill_r][c] = color
            else:
                cap = bar['cap']
                fill_c = (cap + pos) if cap == bar['min_c'] else (cap - pos)
                if not (bar['min_c'] <= fill_c <= bar['max_c']):
                    continue
                if bar['open'] == 'up':
                    for r in range(bar['min_r'] - 1, -1, -1):
                        if out[r][fill_c] == BG:
                            out[r][fill_c] = color
                else:
                    for r in range(bar['max_r'] + 1, rows):
                        if out[r][fill_c] == BG:
                            out[r][fill_c] = color
    return out


def _find_cap(tA: list[int], tB: list[int], n: int) -> int:
    """Return 0 if cap is at start of the tracks, n-1 if at end."""
    start_ok = _has_transition(tA, tB, at_start=True)
    end_ok = _has_transition(tA, tB, at_start=False)

    if start_ok and not end_ok:
        return 0
    if end_ok and not start_ok:
        return n - 1

    # Fallback: double-4 pattern (consecutive 4s at one end)
    if n >= 2:
        d4_start = (tA[0] == 4 and tA[1] == 4) or (tB[0] == 4 and tB[1] == 4)
        d4_end = (tA[-1] == 4 and tA[-2] == 4) or (tB[-1] == 4 and tB[-2] == 4)
        if d4_start and not d4_end:
            return 0
        if d4_end and not d4_start:
            return n - 1
    return 0  # default


def _has_transition(tA: list[int], tB: list[int], at_start: bool) -> bool:
    """Check if the given end has a cap transition: both tracks match at
    the end position, and one track retains the end color at the adjacent pos."""
    if at_start:
        if tA[0] != tB[0]:
            return False
        end_color = tA[0]
        return tA[1] == end_color or tB[1] == end_color
    else:
        if tA[-1] != tB[-1]:
            return False
        end_color = tA[-1]
        return tA[-2] == end_color or tB[-2] == end_color


if __name__ == '__main__':
    path = sys.argv[1]
    data = json.load(open(path))
    pairs = data.get('train', []) + data.get('test', [])
    all_pass = True
    for i, pair in enumerate(pairs):
        result = solve(pair['input'])
        expected = pair.get('output')
        if expected is None:
            print(f'Pair {i}: no expected output (test)')
            continue
        if result == expected:
            print(f'Pair {i}: PASS')
        else:
            all_pass = False
            print(f'Pair {i}: FAIL')
            for r in range(len(expected)):
                for c in range(len(expected[0])):
                    if result[r][c] != expected[r][c]:
                        print(f'  ({r},{c}): got {result[r][c]}, expected {expected[r][c]}')
    if all_pass:
        print('ALL PASS')
