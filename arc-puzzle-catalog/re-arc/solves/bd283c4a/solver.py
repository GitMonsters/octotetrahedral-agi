"""ARC-AGI puzzle bd283c4a solver.
Rule: Count color frequencies, sort descending, fill output column-by-column bottom-to-top.
"""
from collections import Counter

def transform(input_grid: list[list[int]]) -> list[list[int]]:
    rows = len(input_grid)
    cols = len(input_grid[0])
    
    counts: Counter = Counter()
    for row in input_grid:
        for val in row:
            counts[val] += 1
    
    sorted_colors = sorted(counts.items(), key=lambda x: -x[1])
    
    output = [[0] * cols for _ in range(rows)]
    
    color_idx = 0
    remaining = sorted_colors[0][1]
    current_color = sorted_colors[0][0]
    
    for c in range(cols):
        for r in range(rows - 1, -1, -1):
            if remaining == 0:
                color_idx += 1
                current_color = sorted_colors[color_idx][0]
                remaining = sorted_colors[color_idx][1]
            output[r][c] = current_color
            remaining -= 1
    
    return output

if __name__ == "__main__":
    # Training examples
    ex0_in  = [[4,4,8,9,4,9,4,4,9,4],[4,9,4,4,5,4,4,5,9,4],[5,9,9,4,5,9,4,5,9,4],[5,9,4,4,5,9,4,9,4,4],[5,9,4,9,4,4,4,4,5,4],[5,9,4,9,4,4,9,4,5,4],[5,9,4,5,4,5,9,4,4,4],[5,9,4,9,4,5,9,4,4,9],[5,9,4,9,4,4,9,5,4,8],[4,9,4,4,9,4,9,5,4,4]]
    ex0_out = [[4,4,4,4,4,9,9,5,5,8],[4,4,4,4,4,9,9,9,5,8],[4,4,4,4,4,9,9,9,5,5],[4,4,4,4,4,9,9,9,5,5],[4,4,4,4,4,9,9,9,5,5],[4,4,4,4,4,9,9,9,5,5],[4,4,4,4,4,9,9,9,5,5],[4,4,4,4,4,9,9,9,5,5],[4,4,4,4,4,9,9,9,5,5],[4,4,4,4,4,9,9,9,5,5]]
    
    ex1_in  = [[2,6,6,6,6,5,6,6,6,6],[2,6,2,6,6,5,2,6,2,6],[6,5,2,6,2,5,2,5,2,2],[6,6,6,6,2,5,6,5,6,2],[6,2,6,6,2,6,6,6,6,2],[8,2,6,5,6,6,2,8,6,8],[5,2,2,5,6,6,2,6,6,8],[5,2,2,5,2,6,6,6,2,6],[6,2,6,6,2,8,6,5,2,6],[6,2,6,6,2,8,6,5,6,6]]
    ex1_out = [[6,6,6,6,6,2,2,2,5,8],[6,6,6,6,6,2,2,2,5,8],[6,6,6,6,6,2,2,2,5,8],[6,6,6,6,6,2,2,2,5,8],[6,6,6,6,6,2,2,2,5,8],[6,6,6,6,6,2,2,2,5,8],[6,6,6,6,6,2,2,2,5,5],[6,6,6,6,6,2,2,2,5,5],[6,6,6,6,6,2,2,2,5,5],[6,6,6,6,6,6,2,2,5,5]]
    
    test_in  = [[3,3,3,3,9,3,3,8,3,8],[8,2,9,3,3,8,3,8,3,8],[8,2,9,8,9,8,3,3,3,8],[8,3,9,8,3,8,2,8,2,3],[8,3,9,8,3,9,2,8,2,9],[8,3,9,8,3,9,3,3,2,9],[8,3,9,3,3,3,8,3,2,9],[8,3,3,3,9,3,3,3,8,9],[3,3,8,3,9,3,8,3,8,9],[3,3,8,3,9,3,8,3,8,9]]
    test_out = [[3,3,3,3,8,8,8,9,9,2],[3,3,3,3,8,8,8,9,9,2],[3,3,3,3,8,8,8,9,9,2],[3,3,3,3,8,8,8,9,9,2],[3,3,3,3,8,8,8,9,9,2],[3,3,3,3,8,8,8,9,9,2],[3,3,3,3,3,8,8,9,9,2],[3,3,3,3,3,8,8,8,9,2],[3,3,3,3,3,8,8,8,9,9],[3,3,3,3,3,8,8,8,9,9]]
    
    ok = True
    for i, (inp, exp) in enumerate([(ex0_in, ex0_out), (ex1_in, ex1_out), (test_in, test_out)]):
        result = transform(inp)
        label = f"Example {i}" if i < 2 else "Test"
        if result == exp:
            print(f"{label}: PASS")
        else:
            print(f"{label}: FAIL")
            for r in range(len(exp)):
                if result[r] != exp[r]:
                    print(f"  Row {r}: got {result[r]} expected {exp[r]}")
            ok = False
    
    print("SOLVED" if ok else "FAILED")
