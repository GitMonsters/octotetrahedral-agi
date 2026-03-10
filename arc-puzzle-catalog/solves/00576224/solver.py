def solve(grid: list[list[int]]) -> list[list[int]]:
    """Tile 2x2 input into 6x6. Alternate row-pairs: normal, LR-flipped, normal."""
    rows, cols = len(grid), len(grid[0])
    out = []
    for block_row in range(3):
        for input_row in range(rows):
            if block_row % 2 == 0:
                row = grid[input_row] * 3
            else:
                row = grid[input_row][::-1] * 3
            out.append(row)
    return out


if __name__ == "__main__":
    import sys, json
    with open(sys.argv[1]) as f:
        task = json.load(f)
    for i, ex in enumerate(task['train'] + task['test']):
        result = solve(ex['input'])
        status = "PASS" if result == ex['output'] else "FAIL"
        print(f"{'Train' if i < len(task['train']) else 'Test'} {i if i < len(task['train']) else i - len(task['train'])}: {status}")
