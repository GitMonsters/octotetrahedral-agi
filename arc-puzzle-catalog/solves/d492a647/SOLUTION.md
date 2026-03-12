# ARC-AGI Puzzle d492a647 Solution

## Pattern Discovery

**Rule:** Find the special marker (any color value not 0 or 5), then replace all 0s that have the same row and column parity (even/odd position) as the marker with the marker color.

### Detailed Explanation

1. **Identify the Marker:** Search for any cell value that is neither 0 (empty) nor 5 (background). This is the marker value.

2. **Calculate Marker Parity:** Determine if the marker's row index and column index are even or odd.

3. **Transform 0s:** For every 0 in the grid:
   - Calculate its row and column parity
   - If both row and column parity match the marker's parity, replace the 0 with the marker color
   - Otherwise, leave it unchanged

### Examples

**Training Example 1:**
- Marker: 3 at position (5, 7)
- Marker parity: row 5 (odd), column 7 (odd)
- All 0s at positions where both row and column are odd get replaced with 3

**Training Example 2:**
- Marker: 1 at position (4, 5)
- Marker parity: row 4 (even), column 5 (odd)
- All 0s at positions where row is even AND column is odd get replaced with 1

## Test Results

- ✓ Training Example 1: PASS
- ✓ Training Example 2: PASS
- ✓ Test Example: PASS

## Implementation

The solver is implemented in `solver.py` with a `solve(grid)` function that:
1. Creates a copy of the input grid
2. Finds the special marker
3. Applies the parity-based transformation
4. Returns the transformed grid
