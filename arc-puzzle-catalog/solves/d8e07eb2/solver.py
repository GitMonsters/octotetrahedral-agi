import json, copy

def solve(grid):
    R, C = len(grid), len(grid[0])
    out = copy.deepcopy(grid)
    bg = 8  # 🔷

    # Fixed block positions
    block_cols = [2, 7, 12, 17]
    block_rows = [8, 13, 18, 23]

    def get_pattern(r0, c0):
        return tuple(grid[r][c] != bg for r in range(r0, r0+3) for c in range(c0, c0+3))

    # Top patterns (rows 1-3)
    top_patterns = set()
    for c0 in block_cols:
        pat = get_pattern(1, c0)
        if any(pat):
            top_patterns.add(pat)

    # Find matching blocks
    matches = []
    for ri, r0 in enumerate(block_rows):
        for ci, c0 in enumerate(block_cols):
            pat = get_pattern(r0, c0)
            if pat in top_patterns:
                matches.append((ri, ci))

    # Highlight matching blocks with 💚 (5x5 area)
    for ri, ci in matches:
        r0, c0 = block_rows[ri], block_cols[ci]
        for r in range(r0 - 1, r0 + 4):
            for c in range(c0 - 1, c0 + 4):
                if 0 <= r < R and 0 <= c < C and out[r][c] == bg:
                    out[r][c] = 3  # 💚

    # Check for complete row or column of matches
    match_set = set(matches)
    complete_row = any(all((row, col) in match_set for col in range(4)) for row in range(4))
    complete_col = any(all((row, col) in match_set for row in range(4)) for col in range(4))

    if complete_row or complete_col:
        for r in list(range(5)) + list(range(28, 30)):
            for c in range(C):
                if out[r][c] == bg:
                    out[r][c] = 3  # 💚
    else:
        for r in range(28, 30):
            for c in range(C):
                if out[r][c] == bg:
                    out[r][c] = 2  # 🟢

    return out

# Test
DIR = '/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI-2/data/evaluation/'
with open(f'{DIR}d8e07eb2.json') as f:
    task = json.load(f)

for i, ex in enumerate(task['train']):
    result = solve(ex['input'])
    if result == ex['output']:
        print(f"Train {i}: PASS")
    else:
        # Find differences
        diffs = 0
        for r in range(len(result)):
            for c in range(len(result[0])):
                if result[r][c] != ex['output'][r][c]:
                    diffs += 1
        print(f"Train {i}: FAIL ({diffs} diffs)")
