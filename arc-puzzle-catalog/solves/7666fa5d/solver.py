import json
import sys
from collections import defaultdict


def solve(grid):
    rows = len(grid)
    cols = len(grid[0])
    bg = 8

    # Group colored cells by anti-diagonal (r + c)
    antidiag_cells = defaultdict(list)
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg:
                antidiag_cells[r + c].append(r)

    if not antidiag_cells:
        return [row[:] for row in grid]

    # Build contiguous row-segments per anti-diagonal
    antidiag_segments = {}
    for k, rlist in antidiag_cells.items():
        rlist.sort()
        segs = []
        s, e = rlist[0], rlist[0]
        for rv in rlist[1:]:
            if rv == e + 1:
                e = rv
            else:
                segs.append((s, e))
                s, e = rv, rv
        segs.append((s, e))
        antidiag_segments[k] = segs

    result = [row[:] for row in grid]
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg:
                continue
            s = r + c
            has_left = False
            has_right = False
            for k, segs in antidiag_segments.items():
                if k == s:
                    continue
                rp = r + (k - s) / 2.0
                for seg_s, seg_e in segs:
                    if seg_s <= rp <= seg_e:
                        if k < s:
                            has_left = True
                        else:
                            has_right = True
                        break
                if has_left and has_right:
                    break
            if has_left and has_right:
                result[r][c] = 2
    return result


if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        task = json.load(f)
    ok = True
    for i, ex in enumerate(task.get("train", []) + task.get("test", [])):
        out = solve(ex["input"])
        if out == ex["output"]:
            print(f"Example {i}: PASS")
        else:
            print(f"Example {i}: FAIL")
            ok = False
    if not ok:
        sys.exit(1)