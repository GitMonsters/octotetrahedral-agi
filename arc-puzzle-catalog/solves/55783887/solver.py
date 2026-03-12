import json

def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Transform rule (two-phase):
    Phase 1:
    - 1s draw diagonal lines in directions (+1, -1) and (-1, +1)
    - Lines are only drawn if there's a non-background target in that direction
    
    Phase 2:
    - 6s draw diagonal lines in directions (+1, +1) and (-1, -1)
    - But only extend if there's a 1 adjacent in the opposite direction
      (meaning a 1 line approached the 6 from that direction)
    """
    result = [row[:] for row in grid]
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    
    # Find background color
    color_count = {}
    for row in grid:
        for cell in row:
            if cell != 1 and cell != 6:
                color_count[cell] = color_count.get(cell, 0) + 1
    background = max(color_count, key=color_count.get) if color_count else 4
    
    # Find all 1s and 6s
    ones = []
    sixes = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1:
                ones.append((r, c))
            elif grid[r][c] == 6:
                sixes.append((r, c))
    
    # Helper function: check if there's a non-background target in a direction
    def has_target(r, c, dr, dc, search_grid):
        curr_r, curr_c = r + dr, c + dc
        while 0 <= curr_r < rows and 0 <= curr_c < cols:
            if search_grid[curr_r][curr_c] != background:
                return True
            curr_r += dr
            curr_c += dc
        return False
    
    # Phase 1: For each 1, draw lines
    for r1, c1 in ones:
        for dr, dc in [(1, -1), (-1, 1)]:
            if has_target(r1, c1, dr, dc, grid):
                curr_r, curr_c = r1 + dr, c1 + dc
                while 0 <= curr_r < rows and 0 <= curr_c < cols:
                    if grid[curr_r][curr_c] != background:
                        break
                    result[curr_r][curr_c] = 1
                    curr_r += dr
                    curr_c += dc
    
    # Phase 2: For each 6, draw lines (only if there's a 1 approaching from that direction)
    for r6, c6 in sixes:
        for dr, dc in [(1, 1), (-1, -1)]:
            # Check if there's a 1 in the opposite direction (meaning 1s approached from this direction)
            opposite_r, opposite_c = r6 - dr, c6 - dc
            if 0 <= opposite_r < rows and 0 <= opposite_c < cols and result[opposite_r][opposite_c] == 1:
                # Extend in this direction
                curr_r, curr_c = r6 + dr, c6 + dc
                while 0 <= curr_r < rows and 0 <= curr_c < cols:
                    if grid[curr_r][curr_c] != background:
                        break
                    result[curr_r][curr_c] = 6
                    curr_r += dr
                    curr_c += dc
    
    return result


if __name__ == "__main__":
    # Load task from JSON
    with open("~/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/55783887.json".replace("~", "/Users/evanpieser"), "r") as f:
        task = json.load(f)
    
    # Test all training examples
    all_passed = True
    for i, example in enumerate(task["train"]):
        input_grid = example["input"]
        expected_output = example["output"]
        predicted_output = solve(input_grid)
        
        passed = predicted_output == expected_output
        status = "PASS" if passed else "FAIL"
        print(f"Training example {i+1}: {status}")
        
        if not passed:
            all_passed = False
            print(f"  Input shape: {len(input_grid)}x{len(input_grid[0])}")
            print(f"  Expected shape: {len(expected_output)}x{len(expected_output[0])}")
            print(f"  Got shape: {len(predicted_output)}x{len(predicted_output[0])}")
    
    print()
    if all_passed:
        print("All training examples passed!")
    else:
        print("Some examples failed. Debugging needed.")
