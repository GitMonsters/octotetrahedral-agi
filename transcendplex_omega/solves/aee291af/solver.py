#!/usr/bin/env python3
"""
ARC puzzle aee291af solver.

Rule: Find the rectangular region where:
1. The border is entirely made of 8s
2. The interior contains 2s and 8s
3. The pattern of 2-positions is UNIQUE (appears only once among all such rectangles)
"""

def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Solve the ARC puzzle by finding the unique 8-bordered rectangle.
    
    Args:
        grid: A 2D list of integers representing the input grid
        
    Returns:
        A 2D list representing the extracted rectangle
    """
    h_grid = len(grid)
    w_grid = len(grid[0])
    
    # Find all rectangles with 8-borders and 2s inside
    rectangles = []
    
    for start_row in range(h_grid):
        for start_col in range(w_grid):
            # Try all possible rectangle sizes
            for h in range(3, min(h_grid - start_row, w_grid - start_col) + 1):
                for w in range(3, min(h_grid - start_row, w_grid - start_col) + 1):
                    if start_row + h > h_grid or start_col + w > w_grid:
                        continue
                    
                    # Check if all borders are 8
                    all_border_8 = True
                    
                    # Check top and bottom rows
                    for c in range(start_col, start_col + w):
                        if grid[start_row][c] != 8 or grid[start_row + h - 1][c] != 8:
                            all_border_8 = False
                            break
                    
                    # Check left and right columns
                    if all_border_8:
                        for r in range(start_row, start_row + h):
                            if grid[r][start_col] != 8 or grid[r][start_col + w - 1] != 8:
                                all_border_8 = False
                                break
                    
                    if all_border_8:
                        # Extract the rectangle
                        rect = [grid[r][start_col:start_col + w] for r in range(start_row, start_row + h)]
                        
                        # Check if it contains at least one 2
                        if any(2 in row for row in rect):
                            # Get positions of all 2s
                            pos_2s = set()
                            for ri, row in enumerate(rect):
                                for ci, val in enumerate(row):
                                    if val == 2:
                                        pos_2s.add((ri, ci))
                            
                            rectangles.append((h, w, start_row, start_col, pos_2s, rect))
    
    # Group rectangles by their 2-position pattern
    pattern_map = {}
    for h, w, sr, sc, pos_2s, rect in rectangles:
        key = frozenset(pos_2s)
        if key not in pattern_map:
            pattern_map[key] = []
        pattern_map[key].append((h, w, sr, sc, rect))
    
    # Find the pattern that appears exactly once
    for pattern, rects in pattern_map.items():
        if len(rects) == 1:
            return rects[0][4]  # Return the rectangle
    
    # Fallback: return the largest rectangle (shouldn't reach here if puzzle is valid)
    rectangles.sort(key=lambda x: -x[0] * x[1])
    return rectangles[0][5]


def main():
    """Test the solver on all training examples."""
    import json
    
    with open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/aee291af.json") as f:
        data = json.load(f)
    
    print("Testing solver on training examples:")
    all_pass = True
    
    for idx, example in enumerate(data['train']):
        input_grid = example['input']
        expected_output = example['output']
        
        result = solve(input_grid)
        
        if result == expected_output:
            print(f"  ✓ Example {idx + 1} PASS")
        else:
            print(f"  ✗ Example {idx + 1} FAIL")
            print(f"    Expected:")
            for row in expected_output:
                print(f"      {row}")
            print(f"    Got:")
            for row in result:
                print(f"      {row}")
            all_pass = False
    
    if all_pass:
        print("\n✓ All training examples passed!")
    else:
        print("\n✗ Some examples failed")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
