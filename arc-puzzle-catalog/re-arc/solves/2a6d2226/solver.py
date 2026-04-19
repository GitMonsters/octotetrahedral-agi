def transform(grid):
    from collections import Counter

    rows = len(grid)
    cols = len(grid[0])

    flat = [grid[r][c] for r in range(rows) for c in range(cols)]
    bg = Counter(flat).most_common(1)[0][0]
    colors = set(flat) - {bg}

    best_area = 0
    best_rect = None
    best_color = None

    for color in colors:
        heights = [0] * cols
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == color:
                    heights[c] += 1
                else:
                    heights[c] = 0

            stack = []
            for c in range(cols + 1):
                h = heights[c] if c < cols else 0
                while stack and heights[stack[-1]] > h:
                    height = heights[stack.pop()]
                    width = c if not stack else c - stack[-1] - 1
                    area = height * width
                    if area > best_area:
                        best_area = area
                        left = stack[-1] + 1 if stack else 0
                        top = r - height + 1
                        best_rect = (top, left, r, c - 1)
                        best_color = color
                stack.append(c)

    output = [[bg] * cols for _ in range(rows)]
    if best_rect:
        r1, c1, r2, c2 = best_rect
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                output[r][c] = best_color

    return output
