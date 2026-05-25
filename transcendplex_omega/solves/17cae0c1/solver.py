def solve(grid):
    # 3x9 grid split into three 3x3 sections. Each section's 5-pattern determines output color.
    # Pattern lookup: specific 3x3 arrangements of 5s map to specific colors.
    pattern_to_color = {
        ((5,5,5),(5,0,5),(5,5,5)): 3,  # ring
        ((0,0,0),(0,5,0),(0,0,0)): 4,  # center dot
        ((0,0,5),(0,5,0),(5,0,0)): 9,  # anti-diagonal
        ((0,0,0),(0,0,0),(5,5,5)): 1,  # bottom row
        ((5,5,5),(0,0,0),(0,0,0)): 6,  # top row
    }
    result = []
    for r in range(3):
        row = []
        for sec in range(3):
            c_start = sec * 3
            pattern = tuple(
                tuple(grid[rr][c_start + cc] for cc in range(3))
                for rr in range(3)
            )
            color = pattern_to_color[pattern]
            row.extend([color] * 3)
        result.append(row)
    return result

if __name__ == "__main__":
    import sys, json
    with open(sys.argv[1]) as f:
        task = json.load(f)
    for i, ex in enumerate(task['train'] + task['test']):
        result = solve(ex['input'])
        status = "PASS" if result == ex['output'] else "FAIL"
        print(f"{'Train' if i < len(task['train']) else 'Test'} {i if i < len(task['train']) else i - len(task['train'])}: {status}")
