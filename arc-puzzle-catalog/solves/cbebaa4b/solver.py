"""
Solver for ARC-AGI puzzle cbebaa4b.

Pattern: Shapes (connected components of non-zero cells) are scattered across
the grid, each with color-2 "connector" cells at their borders. The color-4
shape is the anchor (stays in place). All other shapes are translated (no
rotation) so their connector-2 cells overlap with free connector-2 cells of
already-placed shapes, assembling everything into one connected structure.
"""
import json
import sys
from collections import deque


def solve(grid):
    rows = len(grid)
    cols = len(grid[0])

    # Find connected components of non-zero cells (4-connected)
    visited = [[False] * cols for _ in range(rows)]
    components = []

    for r in range(rows):
        for c in range(cols):
            if not visited[r][c] and grid[r][c] != 0:
                queue = deque([(r, c)])
                visited[r][c] = True
                cells = []
                while queue:
                    cr, cc = queue.popleft()
                    cells.append((cr, cc, grid[cr][cc]))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] != 0:
                            visited[nr][nc] = True
                            queue.append((nr, nc))
                if cells:
                    components.append(cells)

    # Build piece info
    pieces = []
    anchor_idx = -1
    for i, comp in enumerate(components):
        cell_map = {}
        twos = set()
        for r, c, v in comp:
            cell_map[(r, c)] = v
            if v == 2:
                twos.add((r, c))
        color_counts = {}
        for v in cell_map.values():
            if v != 2:
                color_counts[v] = color_counts.get(v, 0) + 1
        main_color = max(color_counts, key=color_counts.get) if color_counts else 2
        pieces.append({
            'cell_map': cell_map,
            'twos': twos,
            'color': main_color,
        })
        if main_color == 4:
            anchor_idx = i

    n = len(pieces)

    def find_candidates(placed_set, output_grid, free_twos_set):
        """Find all valid (piece_idx, dr, dc, match_count) placements."""
        cands = []
        for j in range(n):
            if j in placed_set:
                continue
            piece = pieces[j]
            tried = set()
            for (fr, fc) in free_twos_set:
                for (pr, pc) in piece['twos']:
                    ddr, ddc = fr - pr, fc - pc
                    if (ddr, ddc) in tried:
                        continue
                    tried.add((ddr, ddc))
                    matches = 0
                    valid = True
                    for (cr, cc), cv in piece['cell_map'].items():
                        nr, nc = cr + ddr, cc + ddc
                        if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                            valid = False
                            break
                        if cv == 2:
                            if (nr, nc) in free_twos_set:
                                matches += 1
                            elif output_grid[nr][nc] == 0:
                                pass
                            else:
                                valid = False
                                break
                        else:
                            if output_grid[nr][nc] != 0:
                                valid = False
                                break
                    if valid and matches >= 1:
                        cands.append((matches, j, ddr, ddc))
        cands.sort(key=lambda x: -x[0])
        return cands

    def apply_placement(piece_idx, dr, dc, output_grid, free_twos_set):
        """Place piece and return new (output, free_twos)."""
        piece = pieces[piece_idx]
        new_out = [row[:] for row in output_grid]
        new_free = set(free_twos_set)
        matched = set()
        added = set()
        for (cr, cc), cv in piece['cell_map'].items():
            nr, nc = cr + dr, cc + dc
            if cv == 2 and (nr, nc) in new_free:
                matched.add((nr, nc))
            else:
                new_out[nr][nc] = cv
                if cv == 2:
                    added.add((nr, nc))
        new_free -= matched
        new_free |= added
        return new_out, new_free

    def backtrack(placed_set, output_grid, free_twos_set):
        """Recursively place pieces; return (free_count, grid) with fewest free 2s."""
        if len(placed_set) == n:
            return (len(free_twos_set), output_grid)
        cands = find_candidates(placed_set, output_grid, free_twos_set)
        best = None
        for _, j, dr, dc in cands:
            new_out, new_free = apply_placement(j, dr, dc, output_grid, free_twos_set)
            new_placed = placed_set | {j}
            result = backtrack(new_placed, new_out, new_free)
            if result is not None:
                if best is None or result[0] < best[0]:
                    best = result
                    if best[0] == 0:
                        return best
        return best

    # Initialize with anchor
    output = [[0] * cols for _ in range(rows)]
    free_twos = set()
    for (r, c), v in pieces[anchor_idx]['cell_map'].items():
        output[r][c] = v
        if v == 2:
            free_twos.add((r, c))

    result = backtrack(frozenset([anchor_idx]), output, frozenset(free_twos))
    if result is not None:
        return result[1]
    return output


if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        data = json.load(f)

    all_pass = True
    for kind in ["train", "test"]:
        for i, pair in enumerate(data.get(kind, [])):
            result = solve(pair["input"])
            expected = pair["output"]
            if result == expected:
                print(f"{kind} {i}: PASS")
            else:
                all_pass = False
                print(f"{kind} {i}: FAIL")
                diffs = 0
                for r in range(len(expected)):
                    for c in range(len(expected[0])):
                        got = result[r][c] if r < len(result) and c < len(result[0]) else None
                        exp = expected[r][c]
                        if got != exp:
                            if diffs < 15:
                                print(f"  ({r},{c}): got {got}, expected {exp}")
                            diffs += 1
                if diffs > 15:
                    print(f"  ... and {diffs - 15} more differences")

    if all_pass:
        print("\nAll PASS!")
