import json
import sys
from typing import List

def solve(grid: List[List[int]]) -> List[List[int]]:
    """
    Solve ARC task 3ed85e70.
    
    Pattern:
    1. Find all contiguous regions of colored cells (non-0, non-3)
    2. For each region, check its expanded bounding box (±1 cell):
       - If contains 0s and only 0s (no other colors), fill those 0s with a border color
       - Border color: 8 gets 1, 4 gets 8, 2 gets 1
    3. For color 2 regions that form 3x3 perfect rectangles:
       - Also change the center cell from 2 to 4
    """
    
    output = [row[:] for row in grid]
    
    # Determine border colors for each interior color
    border_color = {8: 1, 4: 8, 2: 1}
    
    # Find all colored regions via flood fill
    visited = set()
    regions = []  # List of (color, set of cells)
    
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] not in (0, 3) and (i, j) not in visited:
                # Flood fill this region
                color = grid[i][j]
                cells = set()
                queue = [(i, j)]
                visited.add((i, j))
                
                while queue:
                    r, c = queue.pop(0)
                    cells.add((r, c))
                    
                    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        nr, nc = r + dr, c + dc
                        if (0 <= nr < len(grid) and 0 <= nc < len(grid[0]) and
                            (nr, nc) not in visited and grid[nr][nc] == color):
                            visited.add((nr, nc))
                            queue.append((nr, nc))
                
                if cells:
                    regions.append((color, cells))
    
    # Process each region
    for color, cells in regions:
        min_r = min(r for r, c in cells)
        max_r = max(r for r, c in cells)
        min_c = min(c for r, c in cells)
        max_c = max(c for r, c in cells)
        
        h = max_r - min_r + 1
        w = max_c - min_c + 1
        
        # Handle color 2 regions: change center to 4 if perfect 3x3
        if color == 2:
            if h == 3 and w == 3 and len(cells) == 9:
                center_r = min_r + 1
                center_c = min_c + 1
                output[center_r][center_c] = 4
        
        # For colors that have border filling: expand and check for 0s
        if color in border_color:
            # Expand bounding box by 1 in all directions
            expand_min_r = max(0, min_r - 1)
            expand_max_r = min(len(grid) - 1, max_r + 1)
            expand_min_c = max(0, min_c - 1)
            expand_max_c = min(len(grid[0]) - 1, max_c + 1)
            
            # Check what's in the expanded region
            has_zeros = False
            has_other_colors = False
            
            for r in range(expand_min_r, expand_max_r + 1):
                for c in range(expand_min_c, expand_max_c + 1):
                    if (r, c) not in cells:  # Outside the original region
                        if grid[r][c] == 0:
                            has_zeros = True
                        elif grid[r][c] not in (0, 3, color):
                            has_other_colors = True
            
            # Only fill if we have 0s and no other colors (except 3 which is background)
            if has_zeros and not has_other_colors:
                # Fill 0s with border color
                for r in range(expand_min_r, expand_max_r + 1):
                    for c in range(expand_min_c, expand_max_c + 1):
                        if grid[r][c] == 0:
                            output[r][c] = border_color[color]
    
    return output


def main():
    if len(sys.argv) > 1:
        task_path = sys.argv[1]
    else:
        task_path = os.path.expanduser("~/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/3ed85e70.json")
    
    with open(task_path) as f:
        task = json.load(f)
    
    all_pass = True
    for idx, example in enumerate(task['train']):
        result = solve(example['input'])
        expected = example['output']
        
        if result == expected:
            print(f"Train example {idx}: PASS")
        else:
            print(f"Train example {idx}: FAIL")
            all_pass = False
            # Debug: show first difference
            for i in range(len(result)):
                for j in range(len(result[0])):
                    if result[i][j] != expected[i][j]:
                        print(f"  First diff at ({i},{j}): got {result[i][j]}, expected {expected[i][j]}")
                        break
                if result[i] != expected[i]:
                    break
    
    if all_pass:
        print("\nAll training examples PASS!")
    else:
        print("\nSome training examples FAILED")
    
    return 0 if all_pass else 1


if __name__ == "__main__":
    import os
    sys.exit(main())
