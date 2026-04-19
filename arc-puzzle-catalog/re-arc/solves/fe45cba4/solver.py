"""ARC puzzle fe45cba4 solver."""
from collections import Counter

def transform(input_grid: list[list[int]]) -> list[list[int]]:
    rows = len(input_grid)
    cols = len(input_grid[0])

    # Background = most common color
    flat = [c for row in input_grid for c in row]
    bg = Counter(flat).most_common(1)[0][0]

    # Collect non-bg cells by color
    color_cells: dict[int, list[tuple[int,int]]] = {}
    for r in range(rows):
        for c in range(cols):
            v = input_grid[r][c]
            if v != bg:
                color_cells.setdefault(v, []).append((r, c))

    # Connected components via BFS
    def get_components(cells):
        cell_set = set(cells)
        visited = set()
        comps = []
        for cell in cells:
            if cell in visited:
                continue
            comp = []
            queue = [cell]
            visited.add(cell)
            while queue:
                cr, cc = queue.pop(0)
                comp.append((cr, cc))
                for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
                    nb = (cr+dr, cc+dc)
                    if nb in cell_set and nb not in visited:
                        visited.add(nb)
                        queue.append(nb)
            comps.append(comp)
        return comps

    # Identify merge color (2 components) vs keep color (1 component)
    merge_color = keep_color = None
    for color, cells in color_cells.items():
        if len(get_components(cells)) >= 2:
            merge_color = color
        else:
            keep_color = color

    # Build output: start with background
    output = [[bg] * cols for _ in range(rows)]

    # Copy keep-color cells unchanged
    for r, c in color_cells[keep_color]:
        output[r][c] = keep_color

    # For merge color: find main component (rightmost by avg column)
    merge_comps = get_components(color_cells[merge_color])
    main_comp = max(merge_comps, key=lambda comp: sum(c for _, c in comp) / len(comp))

    # Rectangle row span = main component's row span
    min_row = min(r for r, _ in main_comp)
    max_row = max(r for r, _ in main_comp)
    num_rows = max_row - min_row + 1

    # Rectangle width from total area
    total_area = len(color_cells[merge_color])
    num_cols_rect = total_area // num_rows

    # Fill rectangle, right-aligned to grid edge
    for r in range(min_row, max_row + 1):
        for c in range(cols - num_cols_rect, cols):
            output[r][c] = merge_color

    return output


# === Test against all examples ===
if __name__ == "__main__":
    examples = [
        {
            "input": [[7,7,7,7,7,9,9,9],[7,7,7,7,7,7,9,9],[2,7,7,7,7,9,9,9],[2,2,2,7,7,7,7,7],[2,2,2,7,7,2,2,2],[2,7,7,7,7,7,7,2],[7,7,7,7,7,7,7,2],[7,7,7,7,7,2,2,2]],
            "output": [[7,7,7,7,7,9,9,9],[7,7,7,7,7,7,9,9],[7,7,7,7,7,9,9,9],[7,7,7,7,7,7,7,7],[7,7,7,7,2,2,2,2],[7,7,7,7,2,2,2,2],[7,7,7,7,2,2,2,2],[7,7,7,7,2,2,2,2]]
        },
        {
            "input": [[7,7,7,7,7,9,9,9],[9,7,7,7,7,7,9,9],[9,9,7,7,7,9,9,9],[9,7,7,7,7,7,7,7],[7,7,7,7,7,2,2,2],[7,7,7,7,7,7,7,2],[7,7,7,7,7,7,7,2],[7,7,7,7,7,2,2,2]],
            "output": [[7,7,7,7,9,9,9,9],[7,7,7,7,9,9,9,9],[7,7,7,7,9,9,9,9],[7,7,7,7,7,7,7,7],[7,7,7,7,7,2,2,2],[7,7,7,7,7,7,7,2],[7,7,7,7,7,7,7,2],[7,7,7,7,7,2,2,2]]
        }
    ]

    all_pass = True
    for i, ex in enumerate(examples):
        result = transform(ex["input"])
        if result == ex["output"]:
            print(f"Example {i}: PASS")
        else:
            print(f"Example {i}: FAIL")
            for r, (got, exp) in enumerate(zip(result, ex["output"])):
                if got != exp:
                    print(f"  row {r}: got {got} expected {exp}")
            all_pass = False

    # Test input
    test_input = [[7,7,7,7,7,7,7,7,7,7,9,9,9,9,9,9],[7,7,7,7,7,7,7,7,7,7,7,9,9,9,9,9],[7,7,7,7,7,7,7,7,7,7,7,7,9,9,9,9],[7,7,7,7,7,7,7,7,7,7,7,9,9,9,9,9],[7,7,7,7,7,7,7,7,7,7,9,9,9,9,9,9],[8,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7],[8,8,7,8,7,7,7,7,7,7,7,7,7,7,7,7],[8,8,8,8,8,7,7,7,7,7,7,7,7,7,7,7],[8,8,7,8,7,7,7,7,7,7,8,8,8,8,8,8],[8,7,7,7,7,7,7,7,7,7,7,8,7,8,8,8],[7,7,7,7,7,7,7,7,7,7,7,7,7,7,8,8],[7,7,7,7,7,7,7,7,7,7,7,8,7,8,8,8],[7,7,7,7,7,7,7,7,7,7,8,8,8,8,8,8],[7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7],[7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7],[7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7]]
    expected = [[7,7,7,7,7,7,7,7,7,7,9,9,9,9,9,9],[7,7,7,7,7,7,7,7,7,7,7,9,9,9,9,9],[7,7,7,7,7,7,7,7,7,7,7,7,9,9,9,9],[7,7,7,7,7,7,7,7,7,7,7,9,9,9,9,9],[7,7,7,7,7,7,7,7,7,7,9,9,9,9,9,9],[7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7],[7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7],[7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7],[7,7,7,7,7,7,7,7,7,8,8,8,8,8,8,8],[7,7,7,7,7,7,7,7,7,8,8,8,8,8,8,8],[7,7,7,7,7,7,7,7,7,8,8,8,8,8,8,8],[7,7,7,7,7,7,7,7,7,8,8,8,8,8,8,8],[7,7,7,7,7,7,7,7,7,8,8,8,8,8,8,8],[7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7],[7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7],[7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7]]

    test_result = transform(test_input)
    if test_result == expected:
        print("Test: PASS")
    else:
        print("Test: FAIL")
        for r, (got, exp) in enumerate(zip(test_result, expected)):
            if got != exp:
                print(f"  row {r}: got {got} expected {exp}")
        all_pass = False

    print("\nSOLVED" if all_pass else "\nFAILED")
