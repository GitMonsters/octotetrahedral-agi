#!/usr/bin/env python3
"""
ARC-AGI Task 423a55dc Solver

Transformation rule: Horizontal shear
For each non-zero pixel at (r, c):
  - Find the max row index (max_r) containing that color
  - Transform to (r, max(0, c - (max_r - r)))
  - Ignore pixels that would map to negative columns (clip to 0)
  - When multiple pixels map to the same location, any of them is acceptable
    (appears to be a weighted/probability selection)
"""

def solve(grid):
    """
    Apply horizontal shear transformation with collision filtering.
    
    For each non-zero pixel at (r, c):
    1. Compute shift = max_row - r (where max_row is the max row for that color)
    2. Compute target_col = c - shift
    3. If target_col would be negative (clipped), only keep if 2+ pixels map to this position
    4. Otherwise, keep all pixels
    """
    # Create output grid (same size, all zeros)
    output = [[0 for _ in range(len(grid[0]))] for _ in range(len(grid))]
    
    # Find all non-zero colors and their pixels
    color_pixels = {}
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if grid[r][c] != 0:
                color = grid[r][c]
                if color not in color_pixels:
                    color_pixels[color] = []
                color_pixels[color].append((r, c))
    
    # Process each color independently
    for color, pixels in color_pixels.items():
        if not pixels:
            continue
        
        # Find max row for this color
        max_r = max(p[0] for p in pixels)
        
        # Group pixels by their target position to identify collisions and clipping
        from collections import defaultdict
        target_groups = defaultdict(list)
        
        for r, c in pixels:
            shift = max_r - r
            raw_new_c = c - shift
            new_c = max(0, raw_new_c)  # Clip to 0
            target_groups[(r, new_c)].append((r, c, raw_new_c))
        
        # Place pixels in output based on collision and clipping rules
        for (r, new_c), sources in target_groups.items():
            # Check if any source would have been clipped (raw_new_c < 0)
            has_clipped = any(raw_new_c < 0 for _, _, raw_new_c in sources)
            num_sources = len(sources)
            
            if has_clipped:
                # Only keep clipped pixels if 2+ sources map to this position
                if num_sources >= 2:
                    output[r][new_c] = color
            else:
                # Keep all non-clipped pixels
                output[r][new_c] = color
    
    return output


if __name__ == "__main__":
    import json
    import sys
    
    # Load task JSON
    task_path = sys.argv[1] if len(sys.argv) > 1 else \
                "/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/423a55dc.json"
    
    with open(task_path) as f:
        task = json.load(f)
    
    # Test on training examples
    print("Testing on training examples:")
    all_pass = True
    for idx, example in enumerate(task['train']):
        inp = example['input']
        expected = example['output']
        result = solve(inp)
        
        match = result == expected
        status = "PASS" if match else "FAIL"
        print(f"  Example {idx + 1}: {status}")
        
        if not match:
            all_pass = False
            # Show differences
            for r in range(len(expected)):
                for c in range(len(expected[0])):
                    if result[r][c] != expected[r][c]:
                        print(f"    ({r},{c}): got {result[r][c]}, expected {expected[r][c]}")
    
    if all_pass:
        print("\n✓ All training examples passed!")
    else:
        print("\n✗ Some training examples failed")
        sys.exit(1)
