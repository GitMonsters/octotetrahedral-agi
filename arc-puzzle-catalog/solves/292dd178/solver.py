def solve(grid: list[list[int]]) -> list[list[int]]:
    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    
    # Look for enclosed rectangular regions
    for top in range(rows):
        for left in range(cols):
            if grid[top][left] == 1:
                # Try to find a rectangle starting from this corner
                for bottom in range(top + 2, rows):
                    for right in range(left + 2, cols):
                        # Check if corners are all 1s
                        if (grid[top][left] == 1 and grid[top][right] == 1 and 
                            grid[bottom][left] == 1 and grid[bottom][right] == 1):
                            
                            # Check if this region is mostly enclosed by 1s
                            # Count 1s on borders
                            border_1s = 0
                            border_total = 0
                            
                            for c in range(left, right + 1):
                                if grid[top][c] == 1:
                                    border_1s += 1
                                border_total += 1
                                if grid[bottom][c] == 1:
                                    border_1s += 1
                                border_total += 1
                                
                            for r in range(top + 1, bottom):
                                if grid[r][left] == 1:
                                    border_1s += 1
                                border_total += 1
                                if grid[r][right] == 1:
                                    border_1s += 1
                                border_total += 1
                            
                            # If mostly enclosed (>= 75% of border is 1s)
                            if border_1s >= border_total * 0.75:
                                # Process interior
                                for rr in range(top + 1, bottom):
                                    for cc in range(left + 1, right):
                                        if grid[rr][cc] != 1:  # This is a hole
                                            result[rr][cc] = 2
                                            
                                            # Extend lines from holes
                                            # Horizontal left
                                            for extend_c in range(cc - 1, -1, -1):
                                                if grid[rr][extend_c] != 1:
                                                    result[rr][extend_c] = 2
                                                else:
                                                    break
                                                    
                                            # Horizontal right  
                                            for extend_c in range(cc + 1, cols):
                                                if grid[rr][extend_c] != 1:
                                                    result[rr][extend_c] = 2
                                                else:
                                                    break
                                            
                                            # Vertical up
                                            for extend_r in range(rr - 1, -1, -1):
                                                if grid[extend_r][cc] != 1:
                                                    result[extend_r][cc] = 2
                                                else:
                                                    break
                                                    
                                            # Vertical down
                                            for extend_r in range(rr + 1, rows):
                                                if grid[extend_r][cc] != 1:
                                                    result[extend_r][cc] = 2
                                                else:
                                                    break
    
    return result

if __name__ == "__main__":
    import json
    with open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/292dd178.json") as f:
        task = json.load(f)
    for i, ex in enumerate(task["train"]):
        result = solve(ex["input"])
        status = "PASS" if result == ex["output"] else "FAIL"
        print(f"Train {i}: {status}")