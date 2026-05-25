import numpy as np
from collections import deque


def solve(grid):
    """
    ARC puzzle fafd9572 solver.
    
    The puzzle works as follows:
    1. The input contains a grid of 1-valued blocks scattered in a rectangular region
    2. Somewhere in the input is a "pattern" - a small grid with values > 1
    3. The block region is tiled into a grid that matches the pattern grid size
    4. The pattern encodes which color each tile should be
    5. The solver replaces each block within a tile with the corresponding pattern color
    
    Algorithm:
    - Find the pattern region (values > 1)
    - Find all connected components of 1's (these are the blocks)
    - Group blocks into tiles based on their spatial arrangement
    - Determine which pattern cell each tile corresponds to
    - Replace 1's in each block with the mapped color from the pattern
    """
    
    grid = np.array(grid)
    result = grid.copy()
    
    # Find the pattern region (values > 1)
    pattern_mask = (grid > 1)
    pattern_positions = np.argwhere(pattern_mask)
    
    if len(pattern_positions) == 0:
        # No pattern found, return as is
        return result.tolist()
    
    # Get pattern bounds and extract it
    pattern_min_r = pattern_positions[:, 0].min()
    pattern_max_r = pattern_positions[:, 0].max()
    pattern_min_c = pattern_positions[:, 1].min()
    pattern_max_c = pattern_positions[:, 1].max()
    
    pattern_h = pattern_max_r - pattern_min_r + 1
    pattern_w = pattern_max_c - pattern_min_c + 1
    pattern_grid = grid[pattern_min_r:pattern_max_r + 1, pattern_min_c:pattern_max_c + 1]
    
    # Find all 1-blocks using connected components
    visited = set()
    blocks = []
    
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if grid[r, c] == 1 and (r, c) not in visited:
                block = []
                queue = deque([(r, c)])
                
                while queue:
                    cr, cc = queue.popleft()
                    if (cr, cc) in visited or cr < 0 or cr >= grid.shape[0] or cc < 0 or cc >= grid.shape[1]:
                        continue
                    if grid[cr, cc] != 1:
                        continue
                    
                    visited.add((cr, cc))
                    block.append((cr, cc))
                    
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        queue.append((cr + dr, cc + dc))
                
                if block:
                    blocks.append(block)
    
    # Get the bounds of each block
    block_info = []
    for block in blocks:
        min_r = min(p[0] for p in block)
        max_r = max(p[0] for p in block)
        min_c = min(p[1] for p in block)
        max_c = max(p[1] for p in block)
        block_info.append({
            'positions': block,
            'min_r': min_r,
            'max_r': max_r,
            'min_c': min_c,
            'max_c': max_c
        })
    
    if not block_info:
        return result.tolist()
    
    # Find unique row and column boundaries of blocks
    row_boundaries = sorted(set(b['min_r'] for b in block_info))
    col_boundaries = sorted(set(b['min_c'] for b in block_info))
    
    # Calculate tile dimensions by grouping rows/cols
    # We need to find which pattern cell each block belongs to
    # Blocks form clusters that correspond to pattern cells
    
    # Map each block to a tile based on its position
    # Tile grid dimensions should match pattern dimensions
    tiles = {}  # (tile_r, tile_c) -> [(block_info), ...]
    
    # Determine tile assignment for each block
    # Blocks are grouped based on their row/col neighborhoods
    for block in block_info:
        # Find which tier this block is in
        # Sort blocks by row, then determine which group of pattern_h it belongs to
        row_idx = row_boundaries.index(block['min_r'])
        col_idx = col_boundaries.index(block['min_c'])
        
        # Assign to tile based on how many unique rows/cols we have
        # For a pattern_h x pattern_w, we distribute blocks across tiles
        num_row_groups = len(row_boundaries)
        num_col_groups = len(col_boundaries)
        
        # Tile position: map num_row_groups x num_col_groups to pattern_h x pattern_w
        tile_r = min((row_idx * pattern_h) // num_row_groups, pattern_h - 1)
        tile_c = min((col_idx * pattern_w) // num_col_groups, pattern_w - 1)
        
        if (tile_r, tile_c) not in tiles:
            tiles[(tile_r, tile_c)] = []
        tiles[(tile_r, tile_c)].append(block)
    
    # Color each block based on its tile's pattern color
    for (tile_r, tile_c), blocks_in_tile in tiles.items():
        # Get color from pattern
        if tile_r < pattern_h and tile_c < pattern_w:
            color = pattern_grid[tile_r, tile_c]
        else:
            color = 0
        
        # Update all blocks in this tile
        for block in blocks_in_tile:
            if color > 0:
                for r, c in block['positions']:
                    result[r, c] = color
            else:
                for r, c in block['positions']:
                    result[r, c] = 0
    
    return result.tolist()


if __name__ == '__main__':
    import json
    
    # Load and test on training examples
    with open('/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/fafd9572.json') as f:
        task = json.load(f)
    
    print("Testing solver on training examples:")
    all_pass = True
    
    for idx, example in enumerate(task['train']):
        input_grid = example['input']
        expected = np.array(example['output'])
        result = np.array(solve(input_grid))
        
        passed = np.array_equal(result, expected)
        all_pass = all_pass and passed
        
        print(f"\nTraining Example {idx + 1}: {'PASS' if passed else 'FAIL'}")
        if not passed:
            print(f"Expected shape: {expected.shape}, Got shape: {result.shape}")
            diff_positions = np.argwhere(result != expected)
            if len(diff_positions) > 0:
                print(f"Number of differences: {len(diff_positions)}")
                for pos in diff_positions[:10]:
                    r, c = pos
                    print(f"  ({r}, {c}): expected {expected[r, c]}, got {result[r, c]}")
    
    print(f"\n{'='*60}")
    print(f"Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
