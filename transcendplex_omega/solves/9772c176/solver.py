"""
ARC-AGI task 9772c176 solver.

Pattern: Each input has octagonal shapes made of 8s. The output adds 4s
extending outward from each flat edge of each octagon, forming diamond-shaped
protrusions. The diagonals of the octagon corners are continued beyond the
flat edges to determine the extent of the 4-region.
"""
import json
import copy
from collections import deque


def solve(grid: list[list[int]]) -> list[list[int]]:
    H = len(grid)
    W = len(grid[0])
    out = copy.deepcopy(grid)

    # Find connected components of 8s
    visited = [[False] * W for _ in range(H)]
    shapes = []

    for r in range(H):
        for c in range(W):
            if grid[r][c] == 8 and not visited[r][c]:
                component = []
                q = deque([(r, c)])
                visited[r][c] = True
                while q:
                    cr, cc = q.popleft()
                    component.append((cr, cc))
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < H and 0 <= nc < W and not visited[nr][nc] and grid[nr][nc] == 8:
                            visited[nr][nc] = True
                            q.append((nr, nc))
                shapes.append(component)

    for shape in shapes:
        # Get row-wise left and right extents
        row_ext: dict[int, list[int]] = {}
        for r, c in shape:
            if r not in row_ext:
                row_ext[r] = [c, c]
            else:
                row_ext[r][0] = min(row_ext[r][0], c)
                row_ext[r][1] = max(row_ext[r][1], c)

        min_row = min(row_ext)
        max_row = max(row_ext)
        left = {r: row_ext[r][0] for r in row_ext}
        right = {r: row_ext[r][1] for r in row_ext}

        # Find flat left section (rows where left is at its minimum)
        min_left = min(left.values())
        flat_ls = min(r for r in left if left[r] == min_left)
        flat_le = max(r for r in left if left[r] == min_left)

        # Find flat right section (rows where right is at its maximum)
        max_right = max(right.values())
        flat_rs = min(r for r in right if right[r] == max_right)
        flat_re = max(r for r in right if right[r] == max_right)

        # Left extension: 4s to the left of the flat left side
        for r in range(flat_ls, flat_le + 1):
            d = min(r - flat_ls, flat_le - r)
            if d >= 1:
                for c in range(max(0, min_left - d), min_left):
                    out[r][c] = 4

        # Right extension: 4s to the right of the flat right side
        for r in range(flat_rs, flat_re + 1):
            d = min(r - flat_rs, flat_re - r)
            if d >= 1:
                for c in range(max_right + 1, min(W, max_right + d + 1)):
                    out[r][c] = 4

        # Top extension: 4s above the shape
        for k in range(1, W):
            r = min_row - k
            if r < 0:
                break
            l = left[min_row] + k
            ri = right[min_row] - k
            if l > ri:
                break
            for c in range(max(0, l), min(W, ri + 1)):
                out[r][c] = 4

        # Bottom extension: 4s below the shape
        for k in range(1, W):
            r = max_row + k
            if r >= H:
                break
            l = left[max_row] + k
            ri = right[max_row] - k
            if l > ri:
                break
            for c in range(max(0, l), min(W, ri + 1)):
                out[r][c] = 4

    return out


def verify():
    task = json.load(open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/9772c176.json"))
    all_pass = True

    for i, pair in enumerate(task["train"]):
        result = solve(pair["input"])
        match = result == pair["output"]
        print(f"Train {i}: {'PASS' if match else 'FAIL'}")
        if not match:
            all_pass = False
            for r in range(len(result)):
                if result[r] != pair["output"][r]:
                    print(f"  Row {r}: got    {result[r]}")
                    print(f"  Row {r}: expect {pair['output'][r]}")

    for i, pair in enumerate(task["test"]):
        result = solve(pair["input"])
        if "output" in pair:
            match = result == pair["output"]
            print(f"Test  {i}: {'PASS' if match else 'FAIL'}")
            if not match:
                all_pass = False
                for r in range(len(result)):
                    if result[r] != pair["output"][r]:
                        print(f"  Row {r}: got    {result[r]}")
                        print(f"  Row {r}: expect {pair['output'][r]}")
        else:
            print(f"Test  {i}: (no expected output)")

    print(f"\n{'ALL PASS' if all_pass else 'SOME FAILED'}")
    return all_pass


if __name__ == "__main__":
    verify()
