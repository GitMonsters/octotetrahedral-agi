"""
Puzzle 3e6067c3: Connect adjacent boxes based on key sequence.

Grid has bordered boxes with colored centers, plus a key row at the bottom.
The key sequence defines consecutive color pairs (A,B) meaning "extend A toward B".
For each pair, fill the gap between the A and B boxes with color A.
"""
from collections import Counter, deque
import json
import sys


def solve(grid: list[list[int]]) -> list[list[int]]:
    rows, cols = len(grid), len(grid[0])
    bg = grid[0][0]
    output = [row[:] for row in grid]

    # Detect border color as 2nd most common color after bg
    counts = Counter(grid[r][c] for r in range(rows) for c in range(cols))
    border_color = [c for c, _ in counts.most_common() if c != bg][0]

    # Find all bordered boxes via connected components
    border_cells = {(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == border_color}
    visited: set[tuple[int, int]] = set()
    boxes: list[dict] = []

    for start in sorted(border_cells):
        if start in visited:
            continue
        comp: set[tuple[int, int]] = set()
        q = deque([start])
        visited.add(start)
        while q:
            r, c = q.popleft()
            comp.add((r, c))
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (nr, nc) in border_cells and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    q.append((nr, nc))

        min_r = min(r for r, c in comp)
        max_r = max(r for r, c in comp)
        min_c = min(c for r, c in comp)
        max_c = max(c for r, c in comp)

        center_color = None
        center_rows: set[int] = set()
        center_cols: set[int] = set()
        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                v = grid[r][c]
                if v != border_color and v != bg:
                    center_color = v
                    center_rows.add(r)
                    center_cols.add(c)

        if center_color is not None:
            boxes.append({
                "min_r": min_r, "max_r": max_r,
                "min_c": min_c, "max_c": max_c,
                "color": center_color,
                "center_rows": sorted(center_rows),
                "center_cols": sorted(center_cols),
            })

    # Find key sequence (row with alternating bg and colors at odd columns)
    key_colors: list[int] = []
    for r in range(rows - 1, -1, -1):
        colors = []
        valid = True
        for c in range(cols):
            v = grid[r][c]
            if v != bg and v != border_color:
                if c % 2 != 1:
                    valid = False
                    break
                colors.append(v)
        if valid and len(colors) >= 2:
            key_colors = colors
            break

    def gap_is_clear(r_start: int, r_end: int, c_start: int, c_end: int) -> bool:
        for r in range(r_start, r_end + 1):
            for c in range(c_start, c_end + 1):
                if grid[r][c] != bg:
                    return False
        return True

    def find_aligned(prev: dict, color: int, used: set) -> dict | None:
        """Find a box of given color aligned with prev that has a clear gap."""
        for b in boxes:
            if b["color"] != color or id(b) in used:
                continue
            row_match = b["min_r"] == prev["min_r"] and b["max_r"] == prev["max_r"]
            col_match = b["min_c"] == prev["min_c"] and b["max_c"] == prev["max_c"]
            if row_match:
                if prev["max_c"] < b["min_c"]:
                    gs, ge = prev["max_c"] + 1, b["min_c"] - 1
                elif b["max_c"] < prev["min_c"]:
                    gs, ge = b["max_c"] + 1, prev["min_c"] - 1
                else:
                    continue
                if gap_is_clear(prev["min_r"], prev["max_r"], gs, ge):
                    return b
            elif col_match:
                if prev["max_r"] < b["min_r"]:
                    gs, ge = prev["max_r"] + 1, b["min_r"] - 1
                elif b["max_r"] < prev["min_r"]:
                    gs, ge = b["max_r"] + 1, prev["min_r"] - 1
                else:
                    continue
                if gap_is_clear(gs, ge, prev["min_c"], prev["max_c"]):
                    return b
        return None

    # Build box sequence: key defines a path through specific boxes
    used: set[int] = set()
    box_seq: list[dict] = []

    first_candidates = sorted(
        [b for b in boxes if b["color"] == key_colors[0]],
        key=lambda b: (b["min_r"], b["min_c"]),
    )
    if first_candidates:
        box_seq.append(first_candidates[0])
        used.add(id(first_candidates[0]))

    for i in range(1, len(key_colors)):
        if not box_seq:
            break
        nxt = find_aligned(box_seq[-1], key_colors[i], used)
        if nxt:
            box_seq.append(nxt)
            used.add(id(nxt))

    # Connect consecutive boxes in the sequence
    for i in range(len(box_seq) - 1):
        ab, bb = box_seq[i], box_seq[i + 1]
        color_a = ab["color"]
        row_match = ab["min_r"] == bb["min_r"] and ab["max_r"] == bb["max_r"]

        if row_match:
            if ab["max_c"] < bb["min_c"]:
                gs, ge = ab["max_c"] + 1, bb["min_c"] - 1
            else:
                gs, ge = bb["max_c"] + 1, ab["min_c"] - 1
            for r in ab["center_rows"]:
                for c in range(gs, ge + 1):
                    output[r][c] = color_a
        else:
            if ab["max_r"] < bb["min_r"]:
                gs, ge = ab["max_r"] + 1, bb["min_r"] - 1
            else:
                gs, ge = bb["max_r"] + 1, ab["min_r"] - 1
            for r in range(gs, ge + 1):
                for c in ab["center_cols"]:
                    output[r][c] = color_a

    return output


if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        data = json.load(f)

    for i, ex in enumerate(data["train"]):
        result = solve(ex["input"])
        if result == ex["output"]:
            print(f"Train {i}: PASS")
        else:
            print(f"Train {i}: FAIL")
            for r in range(len(result)):
                for c in range(len(result[0])):
                    if result[r][c] != ex["output"][r][c]:
                        print(f"  ({r},{c}): got {result[r][c]}, expected {ex['output'][r][c]}")
