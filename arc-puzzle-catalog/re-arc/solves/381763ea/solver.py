def transform(grid: list[list[int]]) -> list[list[int]]:
    n = len(grid)
    colors = set(c for row in grid for c in row)

    if 9 in colors:
        # Main diagonal: 9 at (i, i)
        return [[9 if i == j else 7 for j in range(n)] for i in range(n)]
    elif 4 in colors:
        # Anti-diagonal: 9 at (i, n-1-i)
        return [[9 if i + j == n - 1 else 7 for j in range(n)] for i in range(n)]
    else:
        # Only color 7 (no 4 or 9)
        if n % 4 == 0:
            # Vertical line at rightmost column
            return [[9 if j == n - 1 else 7 for j in range(n)] for i in range(n)]
        else:
            # Anti-diagonal
            return [[9 if i + j == n - 1 else 7 for j in range(n)] for i in range(n)]
