import json
import sys
import os
from typing import List

def solve(grid: List[List[int]]) -> List[List[int]]:
    """
    Solve ARC task 3ed85e70.
    
    Pattern:
    1. For color 2 regions that form 3x3 perfect rectangles:
       - Change the center cell from 2 to 4
    2. For color 8 regions:
       - Expand the bounding box by 1 cell in all directions
       - If the expanded box contains only 0s (no other non-bg colors),
         fill those 0s with 1s
    """
    
    output = [row[:] for row in grid]
    
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
    
    # First, check if there are any 3x3 4-regions that will expand
    # This determines what value 2-regions' centers change to
    has_expanding_4_regions = False
    for color, cells in regions:
        if color == 4:
            min_r = min(r for r, c in cells)
            max_r = max(r for r, c in cells)
            min_c = min(c for r, c in cells)
            max_c = max(c for r, c in cells)
            h = max_r - min_r + 1
            w = max_c - min_c + 1
            
            if h == 3 and w == 3 and len(cells) == 9:
                # Check if it will expand (has 0s around it, no other colors)
                expand_min_r = max(0, min_r - 1)
                expand_max_r = min(len(grid) - 1, max_r + 1)
                expand_min_c = max(0, min_c - 1)
                expand_max_c = min(len(grid[0]) - 1, max_c + 1)
                
                other_colors = False
                for r in range(expand_min_r, expand_max_r + 1):
                    for c in range(expand_min_c, expand_max_c + 1):
                        if (r, c) not in cells and grid[r][c] not in (0, 3):
                            other_colors = True
                            break
                    if other_colors:
                        break
                
                if not other_colors:
                    has_expanding_4_regions = True
                    break
    
    for color, cells in regions:
        min_r = min(r for r, c in cells)
        max_r = max(r for r, c in cells)
        min_c = min(c for r, c in cells)
        max_c = max(c for r, c in cells)
        
        h = max_r - min_r + 1
        w = max_c - min_c + 1
        
        # Handle color 2 regions: change center based on whether 4-regions expand
        if color == 2:
            if h == 3 and w == 3 and len(cells) == 9:
                center_r = min_r + 1
                center_c = min_c + 1
                # If there are expanding 4-regions, center becomes 1; otherwise 4
                center_val = 1 if has_expanding_4_regions else 4
                output[center_r][center_c] = center_val
        
        # Handle color 8 regions: expand by 1 and fill 0s with 1s
        elif color == 8:
            # Expand bounding box by 1 in all directions
            expand_min_r = max(0, min_r - 1)
            expand_max_r = min(len(grid) - 1, max_r + 1)
            expand_min_c = max(0, min_c - 1)
            expand_max_c = min(len(grid[0]) - 1, max_c + 1)
            
            # Check if expanded region contains only 0s (and the 8s themselves)
            other_colors = False
            for r in range(expand_min_r, expand_max_r + 1):
                for c in range(expand_min_c, expand_max_c + 1):
                    if (r, c) not in cells and grid[r][c] not in (0, 3):
                        other_colors = True
                        break
                if other_colors:
                    break
            
            # If no other colors in expansion, fill 0s with 1
            if not other_colors:
                for r in range(expand_min_r, expand_max_r + 1):
                    for c in range(expand_min_c, expand_max_c + 1):
                        if grid[r][c] == 0:
                            output[r][c] = 1
        
        # Handle color 4 regions: expand by 1 and fill 0s with 8s (only for 3x3)
        elif color == 4:
            if h == 3 and w == 3 and len(cells) == 9:
                # Expand bounding box by 1 in all directions
                expand_min_r = max(0, min_r - 1)
                expand_max_r = min(len(grid) - 1, max_r + 1)
                expand_min_c = max(0, min_c - 1)
                expand_max_c = min(len(grid[0]) - 1, max_c + 1)
                
                # Check if expanded region contains only 0s (and the 4s themselves)
                other_colors = False
                for r in range(expand_min_r, expand_max_r + 1):
                    for c in range(expand_min_c, expand_max_c + 1):
                        if (r, c) not in cells and grid[r][c] not in (0, 3):
                            other_colors = True
                            break
                    if other_colors:
                        break
                
                # If no other colors in expansion, fill 0s with 8
                if not other_colors:
                    for r in range(expand_min_r, expand_max_r + 1):
                        for c in range(expand_min_c, expand_max_c + 1):
                            if grid[r][c] == 0:
                                output[r][c] = 8
        
        # Handle color 1 regions: change center based on whether 4-regions expand
        elif color == 1:
            if h == 3 and w == 3 and len(cells) == 9:
                center_r = min_r + 1
                center_c = min_c + 1
                # If there are expanding 4-regions, center becomes 2; otherwise stays 1
                if has_expanding_4_regions:
                    output[center_r][center_c] = 2
                    
                    # Also expand by 1 and fill 0s with 1
                    expand_min_r = max(0, min_r - 1)
                    expand_max_r = min(len(grid) - 1, max_r + 1)
                    expand_min_c = max(0, min_c - 1)
                    expand_max_c = min(len(grid[0]) - 1, max_c + 1)
                    
                    # Check if expanded region contains only 0s (and the 1s themselves)
                    other_colors = False
                    for r in range(expand_min_r, expand_max_r + 1):
                        for c in range(expand_min_c, expand_max_c + 1):
                            if (r, c) not in cells and grid[r][c] not in (0, 3):
                                other_colors = True
                                break
                        if other_colors:
                            break
                    
                    # If no other colors in expansion, fill 0s with 1
                    if not other_colors:
                        for r in range(expand_min_r, expand_max_r + 1):
                            for c in range(expand_min_c, expand_max_c + 1):
                                if grid[r][c] == 0:
                                    output[r][c] = 1
    
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
    
    if all_pass:
        print("\nAll training examples PASS!")
    else:
        print("\nSome training examples FAILED")
    
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
