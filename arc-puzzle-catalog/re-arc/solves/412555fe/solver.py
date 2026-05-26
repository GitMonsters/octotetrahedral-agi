"""
ARC-AGI Task 412555fe Solver - Version 2
Transformation: Create diagonal (transpose) symmetry
Rule: Prefer non-block values when creating symmetry
"""

import numpy as np
from collections import Counter


def find_block_color(grid):
    """
    Find the 'block' color - the color used as placeholder/filler.
    
    Strategy:
    1. Look for colors that form rectangular-ish regions (not scattered)
    2. Prefer colors with moderate frequency (not too rare, not necessarily most common)
    3. The block color typically appears in contiguous patches
    """
    grid = np.array(grid)
    h, w = grid.shape
    counter = Counter(grid.flatten())
    
    candidates = []
    
    for color, count in counter.items():
        if count < 15:  # Too small
            continue
        
        # Find bounding box
        coords = np.where(grid == color)
        if len(coords[0]) == 0:
            continue
        
        min_r, max_r = coords[0].min(), coords[0].max()
        min_c, max_c = coords[1].min(), coords[1].max()
        bbox_h = max_r - min_r + 1
        bbox_w = max_c - min_c + 1
        bbox_area = bbox_h * bbox_w
        
        # Density in bounding box
        density = count / bbox_area if bbox_area > 0 else 0
        
        # Rectangularity score: how rectangular is the region?
        # Higher if bbox is more rectangular (aspect ratio near 1 or reasonable)
        aspect = max(bbox_h, bbox_w) / (min(bbox_h, bbox_w) + 0.1)
        
        # Compactness: dense regions score higher
        compactness = density
        
        # Size score: prefer moderately sized regions (20-150 cells)
        size_score = 1.0
        if count < 30:
            size_score = count / 30
        elif count > 150:
            size_score = 150 / count
        
        # Combined score
        # Prefer: dense (>0.4), moderately sized regions
        if density >= 0.3:
            score = compactness * size_score * min(count, 100)
            candidates.append((color, count, density, bbox_area, score))
    
    if not candidates:
        # Fallback: return second most common color
        # (first is often background)
        most_common = counter.most_common(2)
        if len(most_common) >= 2:
            return most_common[1][0]
        return most_common[0][0]
    
    # Sort by score
    candidates.sort(key=lambda x: x[4], reverse=True)
    
    return candidates[0][0]


def get_neighbor_fill_value(grid, i, j, block_color):
    """
    Get a fill value by looking at non-block neighbors.
    """
    h, w = grid.shape
    neighbors = []
    
    # Check 4-connected neighbors from both positions
    for ri, rj in [(i, j), (j, i)]:
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = ri + di, rj + dj
            if 0 <= ni < h and 0 <= nj < w:
                val = grid[ni, nj]
                if val != block_color:
                    neighbors.append(val)
    
    if neighbors:
        # Return most common non-block neighbor
        return Counter(neighbors).most_common(1)[0][0]
    
    # Wider search
    for radius in range(2, min(h, w) // 2):
        for ri, rj in [(i, j), (j, i)]:
            for di in range(-radius, radius + 1):
                for dj in range(-radius, radius + 1):
                    if abs(di) + abs(dj) <= radius:  # Manhattan distance
                        ni, nj = ri + di, rj + dj
                        if 0 <= ni < h and 0 <= nj < w:
                            val = grid[ni, nj]
                            if val != block_color:
                                return val
    
    # Last resort: any non-block color
    for color in Counter(grid.flatten()).most_common():
        if color[0] != block_color:
            return color[0]
    
    return 0


def transform(grid):
    """
    Create diagonal symmetry in the grid.
    
    Algorithm:
    1. Detect the block color (placeholder color)
    2. For each pair (i,j) and (j,i):
       - If both have the same value: keep it
       - If one is block and other isn't: use the non-block value
       - If both are block: infer from neighbors
       - If both are non-block but different: choose one (prefer first)
    3. Result: output[i][j] = output[j][i] for all i,j
    """
    grid = np.array(grid)
    h, w = grid.shape
    
    if h != w:
        # Can't create diagonal symmetry on non-square grid
        return grid.tolist()
    
    block_color = find_block_color(grid)
    if block_color is None:
        # No clear block color - grid might already be good
        block_color = Counter(grid.flatten()).most_common(1)[0][0]
    
    output = np.copy(grid)
    
    # Process upper triangle (diagonal and above)
    for i in range(h):
        for j in range(i, w):
            val_ij = grid[i, j]
            val_ji = grid[j, i]
            
            if i == j:
                # Diagonal element
                if val_ij == block_color:
                    # Fill from neighbors
                    output[i, i] = get_neighbor_fill_value(grid, i, i, block_color)
                # else: keep original value
                
            else:
                # Off-diagonal pair
                if val_ij == val_ji:
                    # Already symmetric
                    continue
                
                # Choose which value to use for both positions
                chosen_val = None
                
                if val_ij != block_color and val_ji == block_color:
                    # Use non-block value
                    chosen_val = val_ij
                elif val_ji != block_color and val_ij == block_color:
                    # Use non-block value
                    chosen_val = val_ji
                elif val_ij == block_color and val_ji == block_color:
                    # Both are block - infer from context
                    chosen_val = get_neighbor_fill_value(grid, i, j, block_color)
                else:
                    # Both are non-block but different
                    # Heuristic: prefer the value from the upper triangle position
                    chosen_val = val_ij
                
                # Apply symmetry
                output[i, j] = chosen_val
                output[j, i] = chosen_val
    
    return output.tolist()


if __name__ == "__main__":
    import json
    
    with open('/tmp/rearc45/412555fe.json', 'r') as f:
        data = json.load(f)
    
    print("Testing solver V2 on training examples:\n")
    
    passed = 0
    for idx, example in enumerate(data['train']):
        input_grid = example['input']
        expected_output = example['output']
        
        predicted_output = transform(input_grid)
        
        match = (np.array(predicted_output) == np.array(expected_output)).all()
        
        print(f"Training Example {idx}: {'PASS ✓' if match else 'FAIL ✗'}")
        
        if not match:
            pred = np.array(predicted_output)
            exp = np.array(expected_output)
            diff_count = np.sum(pred != exp)
            total = pred.size
            print(f"  Differences: {diff_count}/{total} cells ({100*diff_count/total:.1f}%)")
            
            # Debug: show detected block color
            from collections import Counter
            block_color = find_block_color(np.array(input_grid))
            inp_counter = Counter(np.array(input_grid).flatten())
            print(f"  Detected block color: {block_color}")
            print(f"  Color frequencies: {sorted(inp_counter.items())[:6]}")
        else:
            passed += 1
    
    print(f"\n{'='*50}")
    print(f"Passed: {passed}/{len(data['train'])}")
    print("="*50)
