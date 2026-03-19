"""
ARC puzzle 4b410a8c solver.

Pattern: Each marker pixel (non-background color) is replaced with a 4-pixel
diagonal cross pattern:
- (row-1, col-1): 0 (black)
- (row-1, col+1): 2 (red)
- (row+1, col-1): 7 (orange)
- (row+1, col+1): 1 (blue)
The marker itself is removed (replaced with background).
"""
import copy
from collections import Counter


def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    
    # Find background color (most common)
    flat = [c for row in grid for c in row]
    background = Counter(flat).most_common(1)[0][0]
    
    # Find all marker positions
    markers = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != background:
                markers.append((r, c))
    
    # Create output grid (copy of input)
    output = copy.deepcopy(grid)
    
    # Replace each marker with the cross pattern
    for r, c in markers:
        # Remove the marker (replace with background)
        output[r][c] = background
        
        # Place the pattern (if within bounds)
        pattern = [
            (r-1, c-1, 0),  # black top-left
            (r-1, c+1, 2),  # red top-right
            (r+1, c-1, 7),  # orange bottom-left
            (r+1, c+1, 1),  # blue bottom-right
        ]
        
        for pr, pc, color in pattern:
            if 0 <= pr < rows and 0 <= pc < cols:
                output[pr][pc] = color
    
    return output
