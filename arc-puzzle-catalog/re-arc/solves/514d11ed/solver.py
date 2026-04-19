"""
Solver for RE-ARC task 514d11ed

Pattern: Identity transformation - output is identical to input.
The grid contains a background color (2) with some foreground pattern,
and the output simply returns the input unchanged.
"""

def transform(grid):
    """
    Identity transformation - returns a copy of the input grid.
    
    Args:
        grid: 2D list of integers representing the input grid
        
    Returns:
        2D list identical to the input
    """
    return [row[:] for row in grid]
