def solve(grid: list[list[int]]) -> list[list[int]]:
    rows = len(grid)
    cols = len(grid[0])
    result = [row[:] for row in grid]
    
    MIN_DIM = 5
    colors = set(grid[r][c] for r in range(rows) for c in range(cols))
    
    for color in colors:
        # For each pair of rows (r1, r2) with height >= MIN_DIM
        for r1 in range(rows):
            for r2 in range(r1 + MIN_DIM - 1, rows):
                # Find columns where all rows r1..r2 have this color
                valid = []
                for c in range(cols):
                    if all(grid[r][c] == color for r in range(r1, r2 + 1)):
                        valid.append(c)
                
                if not valid:
                    continue
                
                # Find contiguous runs of valid columns with width >= MIN_DIM
                runs = []
                start = valid[0]
                for k in range(1, len(valid)):
                    if valid[k] != valid[k - 1] + 1:
                        runs.append((start, valid[k - 1]))
                        start = valid[k]
                runs.append((start, valid[-1]))
                
                for c1, c2 in runs:
                    if c2 - c1 + 1 < MIN_DIM:
                        continue
                    
                    # Check maximality: can't extend in any direction
                    can_up = r1 > 0 and all(grid[r1 - 1][c] == color for c in range(c1, c2 + 1))
                    can_down = r2 < rows - 1 and all(grid[r2 + 1][c] == color for c in range(c1, c2 + 1))
                    can_left = c1 > 0 and all(grid[r][c1 - 1] == color for r in range(r1, r2 + 1))
                    can_right = c2 < cols - 1 and all(grid[r][c2 + 1] == color for r in range(r1, r2 + 1))
                    
                    if not can_up and not can_down and not can_left and not can_right:
                        for r in range(r1, r2 + 1):
                            for c in range(c1, c2 + 1):
                                result[r][c] = 4
    
    return result


if __name__ == "__main__":
    import json
    with open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/25094a63.json") as f:
        task = json.load(f)
    for i, ex in enumerate(task["train"]):
        result = solve(ex["input"])
        status = "PASS" if result == ex["output"] else "FAIL"
        print(f"Train {i}: {status}")
    for i, ex in enumerate(task["test"]):
        result = solve(ex["input"])
        print(f"Test {i}: produced {len(result)}x{len(result[0])}")
