"""
ARC-AGI Puzzle 4ff4c9da Solver

Pattern: The grid is a self-similar tiled pattern with separator lines.
The value 8 marks certain cells, replacing one of the tile values (V).
The transformation propagates 8 to all structurally equivalent positions.
"""

import json
from typing import List


def solve(grid: List[List[int]]) -> List[List[int]]:
    H, W = len(grid), len(grid[0])
    grid = [list(row) for row in grid]

    eights = set((r, c) for r in range(H) for c in range(W) if grid[r][c] == 8)
    if not eights:
        return grid

    sep_val = None
    sep_rows = set()
    for r in range(H):
        vals = set(grid[r])
        if len(vals) == 1 and 8 not in vals:
            sep_rows.add(r)
            sep_val = grid[r][0]

    sep_cols = set()
    for c in range(W):
        vals = set(grid[r][c] for r in range(H))
        if len(vals) == 1 and 8 not in vals:
            sep_cols.add(c)

    def get_bands(sep_set, total):
        bands: list[list[int]] = []
        i = 0
        while i < total:
            if i in sep_set:
                i += 1
                continue
            j = i
            while j < total and j not in sep_set:
                j += 1
            bands.append(list(range(i, j)))
            i = j
        return bands

    row_bands = get_bands(sep_rows, H)
    col_bands = get_bands(sep_cols, W)
    row_to_band = {}
    for bi, band in enumerate(row_bands):
        for r in band:
            row_to_band[r] = bi
    col_to_band = {}
    for bi, band in enumerate(col_bands):
        for c in band:
            col_to_band[c] = bi
    n_rb = len(row_bands)
    n_cb = len(col_bands)

    all_vals = set()
    for r in range(H):
        for c in range(W):
            if grid[r][c] != 8:
                all_vals.add(grid[r][c])
    non_sep_vals = sorted(all_vals - {sep_val})

    best_output = None
    best_score = (-1, -1)

    for V in non_sep_vals:
        base = [list(row) for row in grid]
        for r, c in eights:
            base[r][c] = V

        def get_base_block(rb, cb):
            return tuple(
                tuple(base[r][c] for c in col_bands[cb]) for r in row_bands[rb]
            )

        eight_blocks = {}
        for r, c in eights:
            rb = row_to_band[r]
            cb = col_to_band[c]
            lr = row_bands[rb].index(r)
            lc = col_bands[cb].index(c)
            eight_blocks.setdefault((rb, cb), []).append((lr, lc))

        output = [list(row) for row in base]
        for r, c in eights:
            output[r][c] = 8

        propagated = 0
        blocks_with_matches = set()

        # Phase 1: exact block content matching
        for (erb, ecb), local_8s in eight_blocks.items():
            e_base = get_base_block(erb, ecb)
            for rb in range(n_rb):
                for cb in range(n_cb):
                    if (rb, cb) == (erb, ecb):
                        continue
                    if get_base_block(rb, cb) == e_base:
                        blocks_with_matches.add((erb, ecb))
                        for lr, lc in local_8s:
                            if lr < len(row_bands[rb]) and lc < len(col_bands[cb]):
                                r = row_bands[rb][lr]
                                c = col_bands[cb][lc]
                                if output[r][c] != 8:
                                    propagated += 1
                                output[r][c] = 8

        # Phase 2: V-distribution matching for blocks where ALL V-positions are 8
        for (erb, ecb), local_8s in eight_blocks.items():
            block = get_base_block(erb, ecb)
            bh = len(block)
            bw = len(block[0]) if bh > 0 else 0

            v_positions = set(
                (lr, lc)
                for lr in range(bh)
                for lc in range(bw)
                if block[lr][lc] == V
            )
            if v_positions != set(local_8s):
                continue

            v_per_col = tuple(
                sum(1 for lr in range(bh) if block[lr][lc] == V)
                for lc in range(bw)
            )
            v_per_row = tuple(
                sum(1 for lc in range(bw) if block[lr][lc] == V)
                for lr in range(bh)
            )

            # Phase 2a: match on BOTH V-per-col AND V-per-row (all blocks)
            for rb in range(n_rb):
                for cb in range(n_cb):
                    if (rb, cb) == (erb, ecb):
                        continue
                    if len(row_bands[rb]) != bh or len(col_bands[cb]) != bw:
                        continue
                    other = get_base_block(rb, cb)
                    ovpc = tuple(
                        sum(1 for lr in range(bh) if other[lr][lc] == V)
                        for lc in range(bw)
                    )
                    ovpr = tuple(
                        sum(1 for lc in range(bw) if other[lr][lc] == V)
                        for lr in range(bh)
                    )
                    if ovpc == v_per_col and ovpr == v_per_row:
                        for lr in range(bh):
                            for lc in range(bw):
                                if other[lr][lc] == V:
                                    r = row_bands[rb][lr]
                                    c = col_bands[cb][lc]
                                    if output[r][c] != 8:
                                        propagated += 1
                                    output[r][c] = 8

            # Phase 2b: for unmatched blocks, looser match within same band
            if (erb, ecb) in blocks_with_matches:
                continue

            for rb in range(n_rb):
                if rb == erb or len(row_bands[rb]) != bh:
                    continue
                other = get_base_block(rb, ecb)
                ovpc = tuple(
                    sum(1 for lr in range(bh) if other[lr][lc] == V)
                    for lc in range(bw)
                )
                if ovpc == v_per_col:
                    for lr in range(bh):
                        for lc in range(bw):
                            if other[lr][lc] == V:
                                r = row_bands[rb][lr]
                                c = col_bands[ecb][lc]
                                if output[r][c] != 8:
                                    propagated += 1
                                output[r][c] = 8

            for cb in range(n_cb):
                if cb == ecb or len(col_bands[cb]) != bw:
                    continue
                other = get_base_block(erb, cb)
                cbw = len(col_bands[cb])
                ovpr = tuple(
                    sum(1 for lc in range(cbw) if other[lr][lc] == V)
                    for lr in range(bh)
                )
                if ovpr == v_per_row:
                    for lr in range(bh):
                        for lc in range(cbw):
                            if other[lr][lc] == V:
                                r = row_bands[erb][lr]
                                c = col_bands[cb][lc]
                                if output[r][c] != 8:
                                    propagated += 1
                                output[r][c] = 8

        score = (len(blocks_with_matches), propagated)
        if score > best_score:
            best_score = score
            best_output = output

    return best_output


if __name__ == "__main__":
    import sys

    task_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/arc_task_4ff4c9da.json"
    with open(task_path) as f:
        task = json.load(f)

    all_pass = True
    for i, ex in enumerate(task["train"]):
        result = solve(ex["input"])
        match = result == ex["output"]
        print(f"Train {i}: {'PASS' if match else 'FAIL'}")
        if not match:
            all_pass = False

    for i, ex in enumerate(task["test"]):
        result = solve(ex["input"])
        print(f"\nTest {i} result:")
        if "output" in ex:
            match = result == ex["output"]
            print(f"Test {i}: {'PASS' if match else 'FAIL'}")
            if not match:
                H, W = len(ex["output"]), len(ex["output"][0])
                diffs = [
                    (r, c, result[r][c], ex["output"][r][c])
                    for r in range(H)
                    for c in range(W)
                    if result[r][c] != ex["output"][r][c]
                ]
                print(f"  {len(diffs)} diffs")
                for r, c, g, e in diffs[:5]:
                    print(f"    ({r},{c}): got {g}, expected {e}")
        print(json.dumps(result))
