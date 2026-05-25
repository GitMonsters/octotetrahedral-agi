# ARC-AGI Task 42918530 Solution

## Rule
For each color that appears in multiple rectangular boxes:
1. Find the box with a non-zero interior pattern (i.e., the color appears in the interior)
2. Copy that interior pattern to all other boxes of the same color that have hollow interiors (all zeros)

## Key Implementation Details
- Uses connected components (DFS) to identify distinct rectangular boxes for each color
- Filters out artifact boxes that are too small (< 3x3) to have a meaningful interior
- Extracts the interior of each box (excluding borders)
- Identifies "pattern boxes" (any non-zero cells in interior) vs "hollow boxes" (all zeros)
- Copies the pattern from the pattern box to all hollow boxes of the same color

## Examples
- Example 1: Color 4 appears in 2 boxes (top-right and bottom-left). Top-right has pattern, bottom-left is hollow. Pattern is copied.
- Example 2: Color 8 appears in 2 main boxes (top-right and bottom-left). Similar pattern copying.
- All 4 training examples pass.
