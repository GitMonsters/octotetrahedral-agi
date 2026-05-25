def solve(grid: list[list[int]]) -> list[list[int]]:
    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]

    # Find all valid 3x3 boxes: contain ≥1 two, center is not 2, all non-2 cells are 5
    valid_boxes = []
    for r in range(rows - 2):
        for c in range(cols - 2):
            sub = [grid[r + i][c + j] for i in range(3) for j in range(3)]
            has_two = any(v == 2 for v in sub)
            center_not_two = sub[4] != 2
            all_non2_are_5 = all(v in (2, 5) for v in sub)
            if has_two and center_not_two and all_non2_are_5:
                two_count = sum(1 for v in sub if v == 2)
                valid_boxes.append((r, c, two_count))

    # For each 2-cell, assign it to the valid box with the most 2s
    selected_boxes = set()
    for tr in range(rows):
        for tc in range(cols):
            if grid[tr][tc] != 2:
                continue
            best_box = None
            best_count = 0
            for r, c, count in valid_boxes:
                if r <= tr <= r + 2 and c <= tc <= c + 2:
                    if count > best_count:
                        best_count = count
                        best_box = (r, c)
            if best_box:
                selected_boxes.add(best_box)

    # Apply transformation: center → 4, other non-2 cells → 7
    for r, c in selected_boxes:
        for i in range(3):
            for j in range(3):
                if grid[r + i][c + j] != 2:
                    if i == 1 and j == 1:
                        result[r + i][c + j] = 4
                    else:
                        result[r + i][c + j] = 7

    return result
