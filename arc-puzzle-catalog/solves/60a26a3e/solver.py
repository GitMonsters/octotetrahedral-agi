import json

def solve(grid):
    """
    Find diamond patterns: centered at (r, c) with 0 in center and 2s at:
    (r-1, c), (r+1, c), (r, c-1), (r, c+1)
    
    For diamonds on same row: fill horizontal gaps with 1s
    For diamonds on same column: fill vertical gaps with 1s
    """
    output = [row[:] for row in grid]
    
    twos = set((r, c) for r in range(len(grid)) for c in range(len(grid[0])) if grid[r][c] == 2)
    
    # Find diamonds: cross-shaped with empty center
    diamonds = []
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if grid[r][c] == 0:
                if (r-1, c) in twos and (r+1, c) in twos and (r, c-1) in twos and (r, c+1) in twos:
                    diamonds.append((r, c))
    
    # Group diamonds by row for horizontal filling
    rows_with_diamonds = {}
    for dr, dc in diamonds:
        if dr not in rows_with_diamonds:
            rows_with_diamonds[dr] = []
        rows_with_diamonds[dr].append(dc)
    
    # Fill gaps horizontally
    for row, cols in rows_with_diamonds.items():
        cols = sorted(set(cols))
        for i in range(len(cols) - 1):
            c1, c2 = cols[i], cols[i+1]
            for c in range(c1 + 1, c2):
                if output[row][c] == 0:
                    output[row][c] = 1
    
    # Group diamonds by column for vertical filling
    cols_with_diamonds = {}
    for dr, dc in diamonds:
        if dc not in cols_with_diamonds:
            cols_with_diamonds[dc] = []
        cols_with_diamonds[dc].append(dr)
    
    # Fill gaps vertically
    for col, rows in cols_with_diamonds.items():
        rows = sorted(set(rows))
        for i in range(len(rows) - 1):
            r1, r2 = rows[i], rows[i+1]
            for r in range(r1 + 1, r2):
                if output[r][col] == 0:
                    output[r][col] = 1
    
    return output


if __name__ == '__main__':
    with open('/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/60a26a3e.json') as f:
        task = json.load(f)
    
    all_pass = True
    for i, example in enumerate(task['train']):
        result = solve(example['input'])
        expected = example['output']
        match = result == expected
        print(f"Training {i}: {'PASS' if match else 'FAIL'}")
        if not match:
            all_pass = False
    
    print(f"\n{'All tests passed!' if all_pass else 'Some tests failed!'}")
