"""
ARC-AGI Task d56f2372 Solver

Rule: The input grid contains multiple colored shapes. Exactly one shape
has left-right (vertical axis) symmetry. The output is that symmetric
shape extracted to its bounding box, with all other cells set to 0.
"""

import json
import numpy as np
from typing import List


def solve(grid: List[List[int]]) -> List[List[int]]:
    g = np.array(grid)
    colors = set(g.flatten()) - {0}

    for color in colors:
        mask = (g == color).astype(int)
        rows, cols = np.where(mask)
        if len(rows) == 0:
            continue
        r0, r1 = rows.min(), rows.max()
        c0, c1 = cols.min(), cols.max()
        shape = mask[r0:r1+1, c0:c1+1]

        # Check left-right symmetry
        if np.array_equal(shape, np.fliplr(shape)):
            # Build output with original color values
            out = np.zeros_like(shape)
            out[shape == 1] = color
            return out.tolist()

    # Fallback: return empty grid (should not happen for valid tasks)
    return [[0]]


def verify():
    with open('/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/d56f2372.json') as f:
        task = json.load(f)

    for i, ex in enumerate(task['train']):
        result = solve(ex['input'])
        expected = ex['output']
        match = result == expected
        print(f"Train {i}: {'PASS' if match else 'FAIL'}")
        if not match:
            print(f"  Expected: {expected}")
            print(f"  Got:      {result}")

    # Run on test input
    test_result = solve(task['test'][0]['input'])
    print(f"\nTest 0 output ({len(test_result)}x{len(test_result[0])}):")
    for row in test_result:
        print(''.join(str(c) if c else '.' for c in row))


if __name__ == '__main__':
    verify()
