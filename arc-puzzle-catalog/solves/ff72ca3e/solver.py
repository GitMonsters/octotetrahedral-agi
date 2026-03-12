def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    For each color-4, find the nearest color-5. Create a square of color-2 centered
    on the 4 with half-width = max(row_dist, col_dist) - 1. Merge rectangles from
    multiple 4s. Preserve colors 4 and 5.
    """
    output = [row[:] for row in grid]
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0
    
    fours = []
    fives = []
    
    for r in range(height):
        for c in range(width):
            if grid[r][c] == 4:
                fours.append((r, c))
            elif grid[r][c] == 5:
                fives.append((r, c))
    
    # Collect all rectangles to fill
    rects_to_fill = []
    
    for r4, c4 in fours:
        if not fives:
            continue
        
        # Find nearest 5
        closest_five = min(fives, key=lambda p: abs(p[0] - r4) + abs(p[1] - c4))
        r5, c5 = closest_five
        
        # Calculate distances
        row_dist = abs(r4 - r5)
        col_dist = abs(c4 - c5)
        max_dist = max(row_dist, col_dist)
        
        # Square half-width based on max distance
        half = max_dist - 1
        
        # Clamp to grid bounds
        min_r = max(0, r4 - half)
        max_r = min(height - 1, r4 + half)
        min_c = max(0, c4 - half)
        max_c = min(width - 1, c4 + half)
        
        rects_to_fill.append((min_r, max_r, min_c, max_c))
    
    # Fill all rectangles with color-2, merging overlaps
    for min_r, max_r, min_c, max_c in rects_to_fill:
        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                # Preserve 4 and 5
                if output[r][c] != 4 and output[r][c] != 5:
                    output[r][c] = 2
    
    return output


if __name__ == "__main__":
    import json
    
    with open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/ff72ca3e.json") as f:
        data = json.load(f)
    
    passed = 0
    failed = 0
    
    for idx, example in enumerate(data["train"]):
        result = solve(example["input"])
        expected = example["output"]
        
        if result == expected:
            print(f"✓ Training example {idx + 1} PASSED")
            passed += 1
        else:
            print(f"✗ Training example {idx + 1} FAILED")
            failed += 1
            print(f"  Expected rows: {len(expected)}, Got rows: {len(result)}")
            if len(result) == len(expected):
                for r in range(len(result)):
                    if result[r] != expected[r]:
                        print(f"  Row {r} mismatch")
                        print(f"    Expected: {expected[r]}")
                        print(f"    Got:      {result[r]}")
                        break
    
    print(f"\nResults: {passed} passed, {failed} failed")
    if failed == 0:
        print("ALL TRAINING EXAMPLES PASSED! ✓")
