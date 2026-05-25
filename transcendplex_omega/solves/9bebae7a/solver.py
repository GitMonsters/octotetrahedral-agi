import json
from copy import deepcopy

def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Rule: Mirror the 4-shape across an axis at its boundary.
    The 6-shape determines which axis (row or column).
    The mirror direction is chosen to use available space.
    """
    result = deepcopy(grid)
    
    # Find all 4s and 6s
    fours = set()
    sixes = set()
    
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if grid[i][j] == 4:
                fours.add((i, j))
            elif grid[i][j] == 6:
                sixes.add((i, j))
    
    if not fours or not sixes:
        return result
    
    # Find bounding boxes
    four_rows = sorted([r for r, c in fours])
    four_cols = sorted([c for r, c in fours])
    six_rows = sorted([r for r, c in sixes])
    six_cols = sorted([c for r, c in sixes])
    
    four_min_r, four_max_r = four_rows[0], four_rows[-1]
    four_min_c, four_max_c = four_cols[0], four_cols[-1]
    six_min_r, six_max_r = six_rows[0], six_rows[-1]
    six_min_c, six_max_c = six_cols[0], six_cols[-1]
    
    six_h = six_max_r - six_min_r
    six_w = six_max_c - six_min_c
    
    # Calculate the axis of reflection based on 6-shape orientation
    new_fours = set(fours)
    
    # Determine which direction based on 6-shape dimensions
    if six_w > six_h:
        # Horizontal cross: mirror vertically (across row axis)
        # Choose direction based on available space (considering 6-shape position)
        if four_max_r < six_min_r:
            # 4s above 6s
            space_above = four_min_r
            space_below = six_min_r - four_max_r - 1
        else:
            # 4s below 6s
            space_above = four_min_r - six_max_r - 1
            space_below = len(result) - four_max_r - 1
        
        if space_above > space_below:
            # Mirror upward
            axis_r = four_min_r - 0.5
        else:
            # Mirror downward
            axis_r = four_max_r + 0.5
        
        for r, c in fours:
            mirror_r_float = 2 * axis_r - r
            mirror_r = int(round(mirror_r_float))
            if 0 <= mirror_r < len(result):
                new_fours.add((mirror_r, c))
    
    elif six_h > six_w:
        # Vertical cross: mirror horizontally (across column axis)
        # Choose direction based on available space or position
        if four_max_c < six_min_c:
            # 4s left of 6s: axis at right edge of 4s, extending rightward
            axis_c = four_max_c + 0.5
        else:
            # 4s right of 6s: axis at left edge of 4s, extending leftward
            axis_c = four_min_c - 0.5
        
        for r, c in fours:
            mirror_c_float = 2 * axis_c - c
            mirror_c = int(round(mirror_c_float))
            if 0 <= mirror_c < len(result[0]):
                new_fours.add((r, mirror_c))
    
    else:
        # Square cross: determine based on distance
        dist_r = min(abs(four_max_r - six_min_r), abs(four_min_r - six_max_r))
        dist_c = min(abs(four_max_c - six_min_c), abs(four_min_c - six_max_c))
        
        # Tiebreaker: if distances are equal, look at which direction the 6-shape is relative to 4-shape
        if dist_r < dist_c:
            # Clearly closer vertically: mirror across row axis
            space_above = four_min_r
            space_below = len(result) - four_max_r - 1
            
            if space_above > space_below:
                axis_r = four_min_r - 0.5
            else:
                axis_r = four_max_r + 0.5
            
            for r, c in fours:
                mirror_r_float = 2 * axis_r - r
                mirror_r = int(round(mirror_r_float))
                if 0 <= mirror_r < len(result):
                    new_fours.add((mirror_r, c))
        elif dist_c < dist_r:
            # Clearly closer horizontally: mirror across column axis
            if four_max_c < six_min_c:
                axis_c = four_max_c + 0.5
            else:
                axis_c = four_max_c + 0.5
            
            for r, c in fours:
                mirror_c_float = 2 * axis_c - c
                mirror_c = int(round(mirror_c_float))
                if 0 <= mirror_c < len(result[0]):
                    new_fours.add((r, mirror_c))
        else:
            # Equal distance: check which axis the 6-shape is along
            six_w = six_max_c - six_min_c
            six_h = six_max_r - six_min_r
            
            if six_w > six_h:
                # 6 is more horizontal: use row mirroring
                space_above = four_min_r
                space_below = len(result) - four_max_r - 1
                
                if space_above > space_below:
                    axis_r = four_min_r - 0.5
                else:
                    axis_r = four_max_r + 0.5
                
                for r, c in fours:
                    mirror_r_float = 2 * axis_r - r
                    mirror_r = int(round(mirror_r_float))
                    if 0 <= mirror_r < len(result):
                        new_fours.add((mirror_r, c))
            else:
                # 6 is more vertical or square: use column mirroring as tiebreaker
                if four_max_c < six_min_c:
                    # 4s left of 6s: extend rightward
                    axis_c = four_max_c + 0.5
                else:
                    # 4s right of 6s or overlapping: extend leftward
                    axis_c = four_min_c - 0.5
                
                for r, c in fours:
                    mirror_c_float = 2 * axis_c - c
                    mirror_c = int(round(mirror_c_float))
                    if 0 <= mirror_c < len(result[0]):
                        new_fours.add((r, mirror_c))
    
    # Clear all 6s
    for r, c in sixes:
        result[r][c] = 0
    
    # Add all 4s
    for r, c in new_fours:
        if 0 <= r < len(result) and 0 <= c < len(result[r]):
            result[r][c] = 4
    
    return result


if __name__ == "__main__":
    # Load the puzzle
    with open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/9bebae7a.json") as f:
        puzzle = json.load(f)
    
    print("Testing solver on all training examples:")
    all_pass = True
    
    for i, example in enumerate(puzzle["train"]):
        input_grid = example["input"]
        expected = example["output"]
        result = solve(input_grid)
        
        match = result == expected
        status = "✓ PASS" if match else "✗ FAIL"
        print(f"  Example {i+1}: {status}")
        
        if not match:
            all_pass = False
            print(f"    Expected:\n{expected}")
            print(f"    Got:\n{result}")
    
    if all_pass:
        print("\n✓ All training examples passed!")
    else:
        print("\n✗ Some examples failed")
