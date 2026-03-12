def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Extend rectangle to include markers.
    Preserve both row and column dividers.
    """
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0
    
    result = [row[:] for row in grid]
    
    # Find shape and markers
    shape_cells = set()
    markers = []
    
    for r in range(height):
        for c in range(width):
            if grid[r][c] == 3:
                shape_cells.add((r, c))
            elif grid[r][c] == 6:
                markers.append((r, c))
    
    if not shape_cells:
        return result
    
    # Get current bounds
    shape_rows = [r for r, c in shape_cells]
    shape_cols = [c for r, c in shape_cells]
    
    old_min_row, old_max_row = min(shape_rows), max(shape_rows)
    old_min_col, old_max_col = min(shape_cols), max(shape_cols)
    
    # Determine new bounds based on markers
    new_min_row = old_min_row
    new_max_row = old_max_row
    new_min_col = old_min_col
    new_max_col = old_max_col
    
    for mr, mc in markers:
        # If marker row is within rect rows, extend horizontally to include it
        if old_min_row <= mr <= old_max_row:
            new_min_col = min(new_min_col, mc + 1)
            new_max_col = max(new_max_col, mc - 1)
        # If marker col is within rect cols, extend vertically to include it
        if old_min_col <= mc <= old_max_col:
            new_min_row = min(new_min_row, mr + 1)
            new_max_row = max(new_max_row, mr - 1)
    
    # Find columns that are completely filled in original rect (dividers)
    filled_cols = set()
    for c in range(old_min_col, old_max_col + 1):
        if all((r, c) in shape_cells for r in range(old_min_row, old_max_row + 1)):
            filled_cols.add(c)
    
    divider_cols = filled_cols - {old_min_col, old_max_col}
    
    # Find rows that are completely filled in original rect (dividers)
    filled_rows = set()
    for r in range(old_min_row, old_max_row + 1):
        if all((r, c) in shape_cells for c in range(old_min_col, old_max_col + 1)):
            filled_rows.add(r)
    
    divider_rows = filled_rows - {old_min_row, old_max_row}
    
    # Draw new rectangle
    for r in range(new_min_row, new_max_row + 1):
        for c in range(new_min_col, new_max_col + 1):
            # Top and bottom edges
            if r == new_min_row or r == new_max_row:
                result[r][c] = 3
            # Left and right edges
            elif c == new_min_col or c == new_max_col:
                result[r][c] = 3
            # Interior dividers (rows)
            elif r in divider_rows:
                result[r][c] = 3
            # Interior dividers (columns)
            elif c in divider_cols:
                result[r][c] = 3
            else:
                # Interior empty space
                result[r][c] = 0
    
    # Preserve 6s
    for mr, mc in markers:
        result[mr][mc] = 6
    
    return result


if __name__ == "__main__":
    import json
    
    # Load and test on training examples
    with open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/baf41dbf.json", "r") as f:
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
            print(f"  Expected shape: {len(expected)}x{len(expected[0])}")
            print(f"  Got shape: {len(result)}x{len(result[0])}")
            for r in range(len(expected)):
                for c in range(len(expected[0])):
                    if result[r][c] != expected[r][c]:
                        print(f"  Mismatch at ({r}, {c}): got {result[r][c]}, expected {expected[r][c]}")
                        break
    
    print(f"\nResults: {passed} passed, {failed} failed")
    if failed == 0:
        print("All training examples passed! ✓")
