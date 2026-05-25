import json
from typing import List

def solve(grid: List[List[int]]) -> List[List[int]]:
    """
    Transform grid by dividing rectangular regions of 8s into quadrants
    and filling each with colors: 6 (TL), 1 (TR), 2 (BL), 3 (BR).
    The innermost area gets color 4.
    """
    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    
    # Find all rectangular regions of 8s
    visited = [[False] * cols for _ in range(rows)]
    
    def flood_fill(start_r: int, start_c: int) -> List[tuple]:
        """Find all cells in a connected region of 8s."""
        region = []
        stack = [(start_r, start_c)]
        while stack:
            r, c = stack.pop()
            if r < 0 or r >= rows or c < 0 or c >= cols:
                continue
            if visited[r][c] or grid[r][c] != 8:
                continue
            visited[r][c] = True
            region.append((r, c))
            stack.extend([(r+1, c), (r-1, c), (r, c+1), (r, c-1)])
        return region
    
    # Process each region
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 8 and not visited[r][c]:
                region = flood_fill(r, c)
                
                if not region:
                    continue
                
                # Get bounding box
                min_r = min(rr for rr, cc in region)
                max_r = max(rr for rr, cc in region)
                min_c = min(cc for rr, cc in region)
                max_c = max(cc for rr, cc in region)
                
                height = max_r - min_r + 1
                width = max_c - min_c + 1
                
                # Fill the region with colors based on position
                for rr, cc in region:
                    # Calculate position within region
                    rel_r = rr - min_r
                    rel_c = cc - min_c
                    
                    # Determine distance from edges
                    dist_from_top = rel_r
                    dist_from_bottom = height - 1 - rel_r
                    dist_from_left = rel_c
                    dist_from_right = width - 1 - rel_c
                    
                    # Find minimum distance to any edge
                    min_dist = min(dist_from_top, dist_from_bottom, dist_from_left, dist_from_right)
                    
                    # Determine which corner this is closest to
                    is_top = dist_from_top < dist_from_bottom
                    is_left = dist_from_left < dist_from_right
                    
                    # Color based on which corner it's in
                    if min_dist < 2:  # Outer 2 layers
                        if is_top and is_left:
                            result[rr][cc] = 6
                        elif is_top and not is_left:
                            result[rr][cc] = 1
                        elif not is_top and is_left:
                            result[rr][cc] = 2
                        else:
                            result[rr][cc] = 3
                    else:
                        # Inner area
                        result[rr][cc] = 4
    
    return result


if __name__ == "__main__":
    # Load and test
    with open('/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/639f5a19.json') as f:
        data = json.load(f)
    
    all_pass = True
    for i, example in enumerate(data['train']):
        output = solve(example['input'])
        expected = example['output']
        
        match = output == expected
        all_pass = all_pass and match
        
        status = "PASS" if match else "FAIL"
        print(f"Training example {i+1}: {status}")
        
        if not match:
            print(f"  Expected shape: {len(expected)}x{len(expected[0])}")
            print(f"  Got shape: {len(output)}x{len(output[0])}")
            # Show first mismatch
            for r in range(len(expected)):
                for c in range(len(expected[0])):
                    if output[r][c] != expected[r][c]:
                        print(f"  First mismatch at ({r},{c}): got {output[r][c]}, expected {expected[r][c]}")
                        break
    
    print(f"\nAll training examples passed: {all_pass}")
