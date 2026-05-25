def solve(grid):
    """Tile 5 times horizontally, alternating original and reversed rows."""
    rows = len(grid)
    result = []
    for r in range(rows):
        original = grid[r]
        reversed_row = original[::-1]
        out_row = []
        for t in range(5):
            if t % 2 == 0:
                out_row.extend(original)
            else:
                out_row.extend(reversed_row)
        result.append(out_row)
    return result


if __name__ == "__main__":
    import sys, json
    with open(sys.argv[1]) as f:
        task = json.load(f)
    for i, ex in enumerate(task['train'] + task['test']):
        result = solve(ex['input'])
        status = "PASS" if result == ex['output'] else "FAIL"
        print(f"{'Train' if i < len(task['train']) else 'Test'} {i if i < len(task['train']) else i - len(task['train'])}: {status}")
