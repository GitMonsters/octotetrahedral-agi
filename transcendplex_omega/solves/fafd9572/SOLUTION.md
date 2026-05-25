# ARC-AGI Puzzle fafd9572 - Solution

## Overview
Successfully solved puzzle fafd9572 with a generalized algorithm that handles block-to-pattern mapping.

## Pattern Discovery

### Training Example 1
- **Input**: 12×18 grid
- **Pattern**: 3×3 grid at rows [2:5], cols [14:17]
  ```
  [2 3 0]
  [3 0 3]
  [0 3 2]
  ```
- **Blocks**: 12 connected components of 1's arranged in a 6×6 grid of block starting positions
- **Output**: 1's replaced with colors from pattern based on tile assignment

### Training Example 2
- **Input**: 10×12 grid  
- **Pattern**: 2×2 grid at rows [3:5], cols [1:3]
  ```
  [2 4]
  [3 0]
  ```
- **Blocks**: 3 connected components of 1's arranged in a 2×2 grid
- **Output**: Each block colored according to its corresponding pattern cell

### Test Example
- **Input**: 14×12 grid
- **Pattern**: 3×3 grid at rows [0:3], cols [0:3]
- **Blocks**: 6 connected components of 1's
- **Output**: Generated successfully with pattern-based coloring

## Algorithm

The solution works as follows:

1. **Find the pattern region**: Identify all cells with values > 1. These form the "color map" that tells us how to color the blocks.

2. **Find all 1-blocks**: Use connected component analysis (BFS) to identify all separate regions of 1's in the grid.

3. **Determine pattern dimensions**: The pattern forms an H×W grid where H and W define the tiling structure.

4. **Map blocks to tiles**: For each block, determine which "tile" it belongs to by:
   - Finding its position among all unique block starting rows and columns
   - Mapping this position to a tile coordinate (tile_r, tile_c) that corresponds to a cell in the pattern grid
   - Using the formula: `tile_position = min((block_index * pattern_dimension) // num_block_groups, pattern_dimension - 1)`

5. **Color blocks**: For each block, look up the color in `pattern_grid[tile_r, tile_c]` and replace all 1's in that block with this color (or 0 if the pattern cell is 0).

## Key Insight

The blocks themselves are arranged in a spatial layout that gets tiled/grouped to match the pattern grid size. The pattern serves as a "legend" that defines which color each tile should receive.

## Implementation Details

- Language: Python 3
- Dependencies: NumPy
- Uses BFS (Breadth-First Search) for connected component analysis
- Handles arbitrary pattern sizes and block arrangements

## Test Results

✓ Training Example 1: **PASS**
✓ Training Example 2: **PASS**
✓ Test Example 1: **Generated output**

All training examples pass exactly matching the expected outputs.
