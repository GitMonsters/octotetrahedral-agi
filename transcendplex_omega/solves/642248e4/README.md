# ARC-AGI Puzzle 642248e4 Solution

## Problem Summary
For each cell with value 1 in the grid, fill an adjacent cell with a border color, based on which border the 1 is closest to.

## Pattern Discovery

### Key Observations
1. The grid always has uniform borders on at least two opposite edges
2. Border colors are different from the interior (usually 0s and 1s)
3. For each cell containing 1:
   - Calculate distances to all four borders
   - Fill the adjacent cell in the direction of the nearest border
   - Use that border's color as the fill value

### Border Detection
- **Horizontal borders**: Check if top row is uniform → top_color; bottom row is uniform → bottom_color
- **Vertical borders**: Check if left column is uniform → left_color; right column is uniform → right_color

### Fill Logic
For each 1 at position (i, j):
- If horizontal borders exist (top/bottom):
  - If closer to top (i ≤ h/2): fill (i-1, j) with top_color
  - If closer to bottom (i > h/2): fill (i+1, j) with bottom_color
- If vertical borders exist (left/right):
  - If closer to left (j ≤ w/2): fill (i, j-1) with left_color
  - If closer to right (j > w/2): fill (i, j+1) with right_color

## Examples

### Example 0
- Top border: color 8, Bottom border: color 3
- 1s in top half → fill cell above with 8
- 1s in bottom half → fill cell below with 3
- Result: ✓ PASS

### Example 1
- Top border: color 2, Bottom border: color 4
- Same logic as Example 0
- Result: ✓ PASS

### Example 2
- Left border: color 3, Right border: color 4
- 1s in left half → fill cell left with 3
- 1s in right half → fill cell right with 4
- Result: ✓ PASS

## Testing Results
All 3 training examples pass validation.
