import json
from typing import List, Set, Tuple
import copy

def solve(grid: List[List[int]]) -> List[List[int]]:
    """
    Solve ARC puzzle 55783887.
    
    Rules:
    1. Find all 1s and all 6s in the input
    2. Find all pairs of 1s that form valid diagonals
    3. Draw 1s diagonals (skipping original 6 positions on the diagonal)
    4. For each 6 that's ON the 1s diagonal path, draw its diagonals stopping at 1s
    5. 6s not on the 1s diagonal are preserved but don't extend
    """
    result = [row[:] for row in grid]
    
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0
    
    # Find positions of 1s and 6s
    ones = []
    sixes = set()
    for r in range(height):
        for c in range(width):
            if grid[r][c] == 1:
                ones.append((r, c))
            elif grid[r][c] == 6:
                sixes.add((r, c))
    
    # Find all valid diagonal pairs of 1s
    pairs = []
    for i in range(len(ones)):
        for j in range(i+1, len(ones)):
            r1, c1 = ones[i]
            r2, c2 = ones[j]
            dr = r2 - r1
            dc = c2 - c1
            if abs(dr) == abs(dc) and dr != 0:
                pairs.append((ones[i], ones[j]))
    
    # Identify which 6s are on the 1s diagonal paths
    sixes_on_diagonal = set()
    for (r1, c1), (r2, c2) in pairs:
        step_r = 1 if r2 > r1 else -1
        step_c = 1 if c2 > c1 else -1
        
        r, c = r1, c1
        while True:
            if (r, c) in sixes:
                sixes_on_diagonal.add((r, c))
            if (r, c) == (r2, c2):
                break
            r += step_r
            c += step_c
    
    # Draw all diagonal pairs
    for (r1, c1), (r2, c2) in pairs:
        dr = 1 if r2 > r1 else -1
        dc = 1 if c2 > c1 else -1
        
        # Draw line from (r1, c1) to (r2, c2)
        r, c = r1, c1
        while (r, c) != (r2, c2):
            if (r, c) not in sixes:  # Don't overwrite original 6s
                result[r][c] = 1
            r += dr
            c += dc
        if (r2, c2) not in sixes:  # Don't overwrite original 6s
            result[r2][c2] = 1
    
    # Draw diagonals from 6s that are ON the 1s diagonal, stopping at 1s
    for r6, c6 in sixes_on_diagonal:
        # Up-left diagonal
        r, c = r6 - 1, c6 - 1
        while r >= 0 and c >= 0 and result[r][c] != 1:
            result[r][c] = 6
            r -= 1
            c -= 1
        
        # Up-right diagonal
        r, c = r6 - 1, c6 + 1
        while r >= 0 and c < width and result[r][c] != 1:
            result[r][c] = 6
            r -= 1
            c += 1
        
        # Down-left diagonal
        r, c = r6 + 1, c6 - 1
        while r < height and c >= 0 and result[r][c] != 1:
            result[r][c] = 6
            r += 1
            c -= 1
        
        # Down-right diagonal
        r, c = r6 + 1, c6 + 1
        while r < height and c < width and result[r][c] != 1:
            result[r][c] = 6
            r += 1
            c += 1
    
    return result


if __name__ == "__main__":
    # Load the task
    task = json.load(open('/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/55783887.json'))
    
    # Test all training examples
    all_pass = True
    for idx, example in enumerate(task['train']):
        result = solve(example['input'])
        expected = example['output']
        
        # Check if result matches expected
        match = result == expected
        if not match:
            all_pass = False
            print(f"Example {idx+1}: FAIL")
            # Show first difference
            for r in range(len(result)):
                for c in range(len(result[0])):
                    if result[r][c] != expected[r][c]:
                        print(f"  Difference at ({r},{c}): got {result[r][c]}, expected {expected[r][c]}")
                        break
                if not match:
                    break
        else:
            print(f"Example {idx+1}: PASS")
    
    if all_pass:
        print("\nAll training examples passed!")
    else:
        print("\nSome training examples failed.")
