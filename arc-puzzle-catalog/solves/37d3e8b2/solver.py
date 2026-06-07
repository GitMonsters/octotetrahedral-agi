#!/usr/bin/env python3
"""
ARC-AGI task 37d3e8b2 solver.

Pattern: Each connected blob of color 8 has N enclosed holes (connected
components of 0 inside its bounding box that don't touch the bounding box
border). Map N → output color: {1→1, 2→2, 3→3, 4→7}.
"""

import sys
import json


HOLES_TO_COLOR = {1: 1, 2: 2, 3: 3, 4: 7}


def _find_blobs(grid: list[list[int]]) -> list[list[tuple[int, int]]]:
    h, w = len(grid), len(grid[0])
    visited: set[tuple[int, int]] = set()
    blobs = []
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 8 and (r, c) not in visited:
                blob: list[tuple[int, int]] = []
                stack = [(r, c)]
                while stack:
                    rr, cc = stack.pop()
                    if (rr, cc) in visited:
                        continue
                    if not (0 <= rr < h and 0 <= cc < w):
                        continue
                    if grid[rr][cc] != 8:
                        continue
                    visited.add((rr, cc))
                    blob.append((rr, cc))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        stack.append((rr + dr, cc + dc))
                blobs.append(blob)
    return blobs


def _count_holes(grid: list[list[int]], blob: list[tuple[int, int]]) -> int:
    rs = [r for r, c in blob]
    cs = [c for r, c in blob]
    minr, maxr, minc, maxc = min(rs), max(rs), min(cs), max(cs)
    blob_set = set(blob)
    vis: dict[tuple[int, int], bool] = {}
    holes = 0
    for r in range(minr, maxr + 1):
        for c in range(minc, maxc + 1):
            if (r, c) not in blob_set and (r, c) not in vis and grid[r][c] == 0:
                stack = [(r, c)]
                escapes = False
                while stack:
                    rr, cc = stack.pop()
                    if (rr, cc) in vis:
                        continue
                    if not (minr <= rr <= maxr and minc <= cc <= maxc):
                        escapes = True
                        continue
                    if (rr, cc) in blob_set:
                        continue
                    vis[(rr, cc)] = True
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        stack.append((rr + dr, cc + dc))
                if not escapes:
                    holes += 1
    return holes


def solve(grid: list[list[int]]) -> list[list[int]]:
    result = [row[:] for row in grid]
    blobs = _find_blobs(grid)
    for blob in blobs:
        n = _count_holes(grid, blob)
        color = HOLES_TO_COLOR.get(n, 1)
        for r, c in blob:
            result[r][c] = color
    return result


if __name__ == "__main__":
    # Load task JSON
    task_path = sys.argv[1] if len(sys.argv) > 1 else "~/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/37d3e8b2.json"
    task_path = task_path.replace("~", "/Users/evanpieser")
    
    with open(task_path) as f:
        task = json.load(f)
    
    # Test on all training examples
    all_pass = True
    for idx, example in enumerate(task['train']):
        inp = example['input']
        expected = example['output']
        result = solve(inp)
        
        match = result == expected
        status = "PASS" if match else "FAIL"
        print(f"Train {idx}: {status}")
        
        if not match:
            all_pass = False
            # Show first diff
            for r in range(len(result)):
                for c in range(len(result[0])):
                    if result[r][c] != expected[r][c]:
                        print(f"  Diff at ({r},{c}): got {result[r][c]}, expected {expected[r][c]}")
                        break
    
    if all_pass:
        print("\nAll training examples passed!")
    else:
        print("\nSome training examples failed.")
        sys.exit(1)
