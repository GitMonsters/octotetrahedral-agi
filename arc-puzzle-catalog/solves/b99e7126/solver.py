"""Solver for ARC puzzle b99e7126.

The grid has a repeating tile pattern with border lines. Some tiles are
modified (one color replaced with a new color). The stencil of new-color
positions within a modified tile is applied at the tile level around the
center of the modified region. The input tiles' positions are matched
to the stencil to determine the correct offset.
"""
import json


def solve(grid: list[list[int]]) -> list[list[int]]:
    H, W = len(grid), len(grid[0])
    border_color = grid[0][0]

    period = 0
    for p in range(2, min(H, 20)):
        if all(grid[p][c] == border_color for c in range(W)):
            period = p
            break
    if period == 0:
        return [row[:] for row in grid]

    tile_size = period - 1
    n_tile_rows = (H - 1) // period
    n_tile_cols = (W - 1) // period

    def get_tile(tr: int, tc: int) -> list[list[int]]:
        r0 = tr * period + 1
        c0 = tc * period + 1
        return [[grid[r0 + dr][c0 + dc] for dc in range(tile_size)]
                for dr in range(tile_size)]

    base_tile = get_tile(0, 0)
    base_colors = {v for row in base_tile for v in row}

    modified_positions: list[tuple[int, int]] = []
    mod_tile = None
    new_color = None

    for tr in range(n_tile_rows):
        for tc in range(n_tile_cols):
            tile = get_tile(tr, tc)
            if tile != base_tile:
                modified_positions.append((tr, tc))
                if mod_tile is None:
                    mod_tile = tile
                    for dr in range(tile_size):
                        for dc in range(tile_size):
                            if tile[dr][dc] not in base_colors:
                                new_color = tile[dr][dc]

    if not modified_positions or mod_tile is None or new_color is None:
        return [row[:] for row in grid]

    # Build stencil: positions where new color appears in modified tile
    stencil_ones: list[tuple[int, int]] = []
    for dr in range(tile_size):
        for dc in range(tile_size):
            if mod_tile[dr][dc] == new_color:
                stencil_ones.append((dr, dc))

    stencil_set = set(stencil_ones)
    input_set = set(modified_positions)

    # Find offset by matching input tiles to stencil 1-positions
    # Try each (stencil_pos, input_pos) pair to get a candidate offset
    best_offset = None
    for sr, sc in stencil_ones:
        for ir, ic in modified_positions:
            off_r, off_c = ir - sr, ic - sc
            # Check if ALL input tiles map to stencil 1-positions
            if all((iir - off_r, iic - off_c) in stencil_set
                   for iir, iic in modified_positions):
                best_offset = (off_r, off_c)
                break
        if best_offset is not None:
            break

    if best_offset is None:
        return [row[:] for row in grid]

    off_r, off_c = best_offset

    # Apply stencil with offset to get all output modified tiles
    output_modified: set[tuple[int, int]] = set()
    for sr, sc in stencil_ones:
        tr, tc = sr + off_r, sc + off_c
        if 0 <= tr < n_tile_rows and 0 <= tc < n_tile_cols:
            output_modified.add((tr, tc))

    # Build output
    out = [row[:] for row in grid]
    for tr in range(n_tile_rows):
        for tc in range(n_tile_cols):
            r0 = tr * period + 1
            c0 = tc * period + 1
            tile = mod_tile if (tr, tc) in output_modified else base_tile
            for dr in range(tile_size):
                for dc in range(tile_size):
                    out[r0 + dr][c0 + dc] = tile[dr][dc]

    return out


if __name__ == "__main__":
    import os
    task_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "dataset", "tasks", "b99e7126.json"
    )
    with open(task_path) as f:
        data = json.load(f)

    all_pass = True
    for i, ex in enumerate(data["train"]):
        result = solve(ex["input"])
        expected = ex["output"]
        ok = result == expected
        print(f"Train {i}: {'PASS' if ok else 'FAIL'}")
        if not ok:
            all_pass = False

    for i, ex in enumerate(data["test"]):
        result = solve(ex["input"])
        if "output" in ex:
            ok = result == ex["output"]
            print(f"Test {i}: {'PASS' if ok else 'FAIL'}")
            if not ok:
                all_pass = False
        else:
            print(f"Test {i}: produced {len(result)}x{len(result[0])} output")

    if all_pass:
        print("\nALL PASS")
