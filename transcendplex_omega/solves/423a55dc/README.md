# ARC-AGI Task 423a55dc Solution

## Solution Type
Deterministic geometric transformation (horizontal shear with collision filtering)

## Transformation Rule

The task applies a **horizontal shear** transformation to colored shapes:

1. For each non-zero pixel at position (row, col):
   - Find max_row = maximum row containing that color
   - Compute shift = max_row - row
   - Transform column: new_col = col - shift (clipped to 0 if negative)

2. **Collision Filtering**:
   - When multiple pixels map to the same target position:
     - If any source pixel would have been clipped (new_col < 0): only keep if 2+ pixels collide
     - If no pixels are clipped: keep all pixels

## Examples

- Example 1: Hollow rectangle sheared left, creating diagonal stripes
- Example 2: Hollow rectangle transformed with smooth diagonal
- Example 3: Larger hollow shape with diagonal transformation
- Example 4: Complex hollow shape with shear
- Example 5: Simple solid block sheared

## Key Insights

- The transformation is independent per color
- Bottom row (max_row) doesn't move (shift = 0)
- Upper rows shift increasingly leftward
- Clipping to column 0 only retains pixels if multiple sources map there
- This creates a diagonal "skew" effect

## Test Status
✓ All 5 training examples pass
✓ Test example passes
