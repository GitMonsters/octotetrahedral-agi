import json
import math

def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Solve ARC puzzle c97c0139: Color Swap - Diamond Expansion
    
    Rule: For each continuous line of 2s (horizontal or vertical),
    create an isosceles triangle pattern of 8s perpendicular to the line.
    Width at distance d: max(0, line_length - 2*d)
    """
    output = [row[:] for row in grid]
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0
    
    visited = set()
    
    # Horizontal lines
    for r in range(height):
        c = 0
        while c < width:
            if grid[r][c] == 2 and (r, c, 'h') not in visited:
                start_c = c
                while c < width and grid[r][c] == 2:
                    visited.add((r, c, 'h'))
                    c += 1
                end_c = c - 1
                line_length = end_c - start_c + 1
                mid_c = (start_c + end_c) / 2.0
                
                # Triangle above
                d = 1
                while True:
                    target_r = r - d
                    if target_r < 0:
                        break
                    width_at_d = line_length - 2 * d
                    if width_at_d <= 0:
                        break
                    half_width = width_at_d / 2.0
                    left = math.ceil(mid_c - half_width)
                    right = math.floor(mid_c + half_width)
                    for target_c in range(left, right + 1):
                        if 0 <= target_c < width and output[target_r][target_c] == 0:
                            output[target_r][target_c] = 8
                    d += 1
                
                # Triangle below
                d = 1
                while True:
                    target_r = r + d
                    if target_r >= height:
                        break
                    width_at_d = line_length - 2 * d
                    if width_at_d <= 0:
                        break
                    half_width = width_at_d / 2.0
                    left = math.ceil(mid_c - half_width)
                    right = math.floor(mid_c + half_width)
                    for target_c in range(left, right + 1):
                        if 0 <= target_c < width and output[target_r][target_c] == 0:
                            output[target_r][target_c] = 8
                    d += 1
            else:
                c += 1
    
    # Vertical lines
    for c in range(width):
        r = 0
        while r < height:
            if grid[r][c] == 2 and (r, c, 'v') not in visited:
                start_r = r
                while r < height and grid[r][c] == 2:
                    visited.add((r, c, 'v'))
                    r += 1
                end_r = r - 1
                line_length = end_r - start_r + 1
                mid_r = (start_r + end_r) / 2.0
                
                # Triangle left
                d = 1
                while True:
                    target_c = c - d
                    if target_c < 0:
                        break
                    width_at_d = line_length - 2 * d
                    if width_at_d <= 0:
                        break
                    half_width = width_at_d / 2.0
                    top = math.ceil(mid_r - half_width)
                    bottom = math.floor(mid_r + half_width)
                    for target_r in range(top, bottom + 1):
                        if 0 <= target_r < height and output[target_r][target_c] == 0:
                            output[target_r][target_c] = 8
                    d += 1
                
                # Triangle right
                d = 1
                while True:
                    target_c = c + d
                    if target_c >= width:
                        break
                    width_at_d = line_length - 2 * d
                    if width_at_d <= 0:
                        break
                    half_width = width_at_d / 2.0
                    top = math.ceil(mid_r - half_width)
                    bottom = math.floor(mid_r + half_width)
                    for target_r in range(top, bottom + 1):
                        if 0 <= target_r < height and output[target_r][target_c] == 0:
                            output[target_r][target_c] = 8
                    d += 1
            else:
                r += 1
    
    return output


if __name__ == "__main__":
    with open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/c97c0139.json") as f:
        data = json.load(f)
    
    print("Testing solver on training examples...")
    all_passed = True
    
    for idx, example in enumerate(data["train"]):
        result = solve(example["input"])
        expected = example["output"]
        
        if result == expected:
            print(f"✓ Training example {idx + 1} PASSED")
        else:
            print(f"✗ Training example {idx + 1} FAILED")
            print(f"  Expected shape: {len(expected)}x{len(expected[0])}")
            print(f"  Got shape: {len(result)}x{len(result[0])}")
            all_passed = False
    
    if all_passed:
        print("\n✓ ALL TESTS PASSED")
    else:
        print("\n✗ SOME TESTS FAILED")
        exit(1)
