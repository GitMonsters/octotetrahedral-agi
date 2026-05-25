# ARC-AGI Task 551d5bf1 Solution

## Problem Summary

This task involves transforming grids containing rectangles outlined with 1s.

## Transformation Rule

The solver implements the following deterministic transformation:

1. **Identify Rectangles**: Find all connected components of 1s that form rectangular borders. Each rectangle is defined by its bounding box (min_r, max_r, min_c, max_c).

2. **Fill Interiors**: Fill the interior of each rectangle (cells strictly between the borders) with color 8.

3. **Handle Edge Gaps**: For any position along a rectangle's edge where a 1 is expected but missing (a "gap"):
   - Fill the gap cell itself with 8
   - Extend outward from the gap in the perpendicular direction until hitting:
     - A 1 (boundary), or
     - The grid edge

## Algorithm Details

- **Top/Bottom edges**: Check if (min_r, c) or (max_r, c) exists in the rectangle's component for c in [min_c, max_c]. If missing, fill the gap and extend vertically.
- **Left/Right edges**: Check if (r, min_c) or (r, max_c) exists in the rectangle's component for r in [min_r, max_r]. If missing, fill the gap and extend horizontally.

## Results

- **Training Example 0**: ✓ PASS
- **Training Example 1**: ✓ PASS
- **Test Output**: Generated successfully (170 cells filled with 8)

## Files

- `solver.py`: Complete solver with `solve(grid) -> grid` function
- `output.json`: Test output
- `README.md`: This file
