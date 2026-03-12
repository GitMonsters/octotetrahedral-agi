# Solver for ARC-AGI Puzzle fafd9572

## Quick Start

```bash
python3 solver.py
```

This runs the test cases against the training examples.

## Usage

Import and use the solver:

```python
from solver import solve
import numpy as np

# Your input grid (2D list or numpy array)
input_grid = [[1, 1, 0, ...], [...], ...]

# Get the solved output
output_grid = solve(input_grid)

# Convert to numpy array for easier inspection
output_array = np.array(output_grid)
print(output_array)
```

## How It Works

The puzzle involves:
1. Finding a pattern (grid of values > 1) that acts as a color legend
2. Finding all connected components of 1's in the grid (the blocks)
3. Mapping each block to a tile based on its spatial position
4. Coloring each block according to the pattern color for its tile

See `SOLUTION.md` for detailed explanation.

## Files

- `solver.py` - Main solver implementation with test harness
- `SOLUTION.md` - Detailed explanation of the algorithm
- `README.md` - This file

## Status

✓ All training examples pass
✓ Test example solved
