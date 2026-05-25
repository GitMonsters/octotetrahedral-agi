"""
ARC-AGI Task 85fa5666 Solver

Pattern: Each 2x2 block of 2s has 4 corner markers at diagonal positions.
1. Corner values rotate clockwise (TL←BL, TR←TL, BL←BR, BR←TR).
2. Each corner's new value extends diagonally outward (away from block center)
   until hitting the grid edge or a non-zero input cell.
3. A diagonal stops when the next cell has a 2-cell immediately to its right
   AND a corner of the same other block immediately above, AND the diagonal's
   value matches that corner's rotated value.
"""

from typing import List


def solve(grid: List[List[int]]) -> List[List[int]]:
    rows = len(grid)
    cols = len(grid[0])

    # Find all 2x2 blocks of 2s
    blocks = []
    two_cells: set = set()
    for r in range(rows - 1):
        for c in range(cols - 1):
            if (
                (r, c) not in two_cells
                and grid[r][c] == 2
                and grid[r][c + 1] == 2
                and grid[r + 1][c] == 2
                and grid[r + 1][c + 1] == 2
            ):
                blocks.append((r, c))
                two_cells.update([(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)])

    # Build block metadata with rotated corner values
    block_data = []
    corner_info: dict = {}  # (r,c) -> (block_idx, new_value)
    for bi, (br, bc) in enumerate(blocks):
        tl = (br - 1, bc - 1)
        tr = (br - 1, bc + 2)
        bl = (br + 2, bc - 1)
        bro = (br + 2, bc + 2)
        twos = {(br, bc), (br, bc + 1), (br + 1, bc), (br + 1, bc + 1)}

        new_tl = grid[bl[0]][bl[1]]
        new_tr = grid[tl[0]][tl[1]]
        new_bl = grid[bro[0]][bro[1]]
        new_br = grid[tr[0]][tr[1]]

        corner_info[tl] = (bi, new_tl)
        corner_info[tr] = (bi, new_tr)
        corner_info[bl] = (bi, new_bl)
        corner_info[bro] = (bi, new_br)

        block_data.append({
            "tl": tl, "tr": tr, "bl": bl, "br": bro,
            "new_tl": new_tl, "new_tr": new_tr,
            "new_bl": new_bl, "new_br": new_br,
            "twos": twos,
        })

    out = [row[:] for row in grid]

    for bi, bd in enumerate(block_data):
        out[bd["tl"][0]][bd["tl"][1]] = bd["new_tl"]
        out[bd["tr"][0]][bd["tr"][1]] = bd["new_tr"]
        out[bd["bl"][0]][bd["bl"][1]] = bd["new_bl"]
        out[bd["br"][0]][bd["br"][1]] = bd["new_br"]

        diags = [
            (bd["tl"], bd["new_tl"], (-1, -1)),
            (bd["tr"], bd["new_tr"], (-1, +1)),
            (bd["bl"], bd["new_bl"], (+1, -1)),
            (bd["br"], bd["new_br"], (+1, +1)),
        ]

        for (cr, cc), val, (dr, dc) in diags:
            r, c = cr + dr, cc + dc
            while 0 <= r < rows and 0 <= c < cols:
                if grid[r][c] != 0:
                    break

                # Stop when: cell has a 2-cell to its right AND a corner
                # of the same other block above it, with matching value
                blocked = False
                if c + 1 < cols and (r, c + 1) in two_cells:
                    if r - 1 >= 0 and (r - 1, c) in corner_info:
                        other_bi, corner_val = corner_info[(r - 1, c)]
                        if (
                            other_bi != bi
                            and (r, c + 1) in block_data[other_bi]["twos"]
                            and val == corner_val
                        ):
                            blocked = True

                if blocked:
                    break

                out[r][c] = val
                r += dr
                c += dc

    return out


if __name__ == "__main__":
    import json
    import sys

    path = "/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/85fa5666.json"
    with open(path) as f:
        task = json.load(f)

    # Verify all training examples
    all_pass = True
    for i, pair in enumerate(task["train"]):
        pred = solve(pair["input"])
        if pred == pair["output"]:
            print(f"Train {i}: PASS")
        else:
            print(f"Train {i}: FAIL")
            all_pass = False
            for r in range(len(pred)):
                for c in range(len(pred[0])):
                    if pred[r][c] != pair["output"][r][c]:
                        print(f"  ({r},{c}): got={pred[r][c]}, want={pair['output'][r][c]}")

    # Verify test examples
    for i, pair in enumerate(task["test"]):
        pred = solve(pair["input"])
        if "output" in pair:
            if pred == pair["output"]:
                print(f"Test {i}: PASS")
            else:
                print(f"Test {i}: FAIL")
                all_pass = False
        else:
            print(f"Test {i}: (no expected output)")
            for row in pred:
                print(row)

    if all_pass:
        print("\nAll training examples passed!")
