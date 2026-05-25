import json

def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Rule: Grid is divided into regions (top/bottom or left/right).
    - Region with color 1: 2s move to RIGHT edge
    - Region with color 8: 2s move to LEFT edge
    """
    if not grid:
        return grid
    
    height = len(grid)
    width = len(grid[0])
    result = [row[:] for row in grid]
    
    # Detect divider type
    is_vertical = False
    vertical_divider = None
    
    for c in range(1, width):
        left_colors = set()
        right_colors = set()
        for r in range(min(3, height)):
            if grid[r][c-1] != 2:
                left_colors.add(grid[r][c-1])
            if grid[r][c] != 2:
                right_colors.add(grid[r][c])
        if left_colors and right_colors and left_colors != right_colors:
            is_vertical = True
            vertical_divider = c
            break
    
    horizontal_divider = None
    if not is_vertical:
        for r in range(1, height):
            top_colors = set()
            bot_colors = set()
            for c in range(min(3, width)):
                if grid[r-1][c] != 2:
                    top_colors.add(grid[r-1][c])
                if grid[r][c] != 2:
                    bot_colors.add(grid[r][c])
            if top_colors and bot_colors and top_colors != bot_colors:
                horizontal_divider = r
                break
    
    if is_vertical and vertical_divider:
        # Vertical divider
        for r in range(height):
            # Left region
            two_positions_left = [c for c in range(vertical_divider) if grid[r][c] == 2]
            if two_positions_left:
                # Get bg color for left region
                color_count = {}
                for c in range(vertical_divider):
                    if grid[r][c] != 2:
                        color_count[grid[r][c]] = color_count.get(grid[r][c], 0) + 1
                bg_color = max(color_count, key=color_count.get) if color_count else 1
                
                # Clear 2s
                for c in two_positions_left:
                    result[r][c] = bg_color
                
                # Place based on background color
                if bg_color == 8:
                    for i, c in enumerate(two_positions_left):
                        result[r][i] = 2
                else:  # bg_color == 1
                    for i, c in enumerate(two_positions_left):
                        result[r][vertical_divider - 1 - i] = 2
            
            # Right region
            two_positions_right = [c for c in range(vertical_divider, width) if grid[r][c] == 2]
            if two_positions_right:
                # Get bg color for right region
                color_count = {}
                for c in range(vertical_divider, width):
                    if grid[r][c] != 2:
                        color_count[grid[r][c]] = color_count.get(grid[r][c], 0) + 1
                bg_color = max(color_count, key=color_count.get) if color_count else 8
                
                # Clear 2s
                for c in two_positions_right:
                    result[r][c] = bg_color
                
                # Place based on background color
                if bg_color == 8:
                    for i, c in enumerate(two_positions_right):
                        result[r][vertical_divider + i] = 2
                else:  # bg_color == 1
                    for i, c in enumerate(two_positions_right):
                        result[r][width - 1 - i] = 2
    
    elif horizontal_divider is not None:
        # Horizontal divider
        for r in range(height):
            two_positions = [c for c in range(width) if grid[r][c] == 2]
            if two_positions:
                # Get bg color for this row
                color_count = {}
                for c in range(width):
                    if grid[r][c] != 2:
                        color_count[grid[r][c]] = color_count.get(grid[r][c], 0) + 1
                bg_color = max(color_count, key=color_count.get) if color_count else 1
                
                # Clear 2s
                for c in two_positions:
                    result[r][c] = bg_color
                
                # Place based on background color
                if bg_color == 8:
                    for i, c in enumerate(two_positions):
                        result[r][i] = 2
                else:  # bg_color == 1
                    for i, c in enumerate(two_positions):
                        result[r][width - 1 - i] = 2
    
    return result


if __name__ == "__main__":
    with open("~/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/9b4c17c4.json".replace("~", "/Users/evanpieser"), "r") as f:
        data = json.load(f)
    
    all_passed = True
    for i, example in enumerate(data["train"]):
        input_grid = example["input"]
        expected = example["output"]
        result = solve(input_grid)
        
        if result == expected:
            print(f"✓ Training example {i+1} PASSED")
        else:
            print(f"✗ Training example {i+1} FAILED")
            all_passed = False
    
    if all_passed:
        print("\n✓ ALL TRAINING EXAMPLES PASSED")
    else:
        print("\n✗ SOME EXAMPLES FAILED")
        import sys
        sys.exit(1)
