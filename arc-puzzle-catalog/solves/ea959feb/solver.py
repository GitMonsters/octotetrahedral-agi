def solve(grid: list[list[int]]) -> list[list[int]]:
    """Restore a periodic tiling pattern where rectangular holes were filled with 1s."""
    rows = len(grid)
    cols = len(grid[0])

    def try_period(py: int, px: int) -> list[list[int]] | None:
        """Try to reconstruct grid with given period. Returns grid or None if inconsistent."""
        # For each equivalence class (r%py, c%px), collect non-1 values
        true_val = {}
        for ry in range(py):
            for rx in range(px):
                vals = set()
                for r in range(ry, rows, py):
                    for c in range(rx, cols, px):
                        if grid[r][c] != 1:
                            vals.add(grid[r][c])
                if len(vals) > 1:
                    return None  # Inconsistent non-1 values
                if len(vals) == 1:
                    true_val[(ry, rx)] = vals.pop()
                else:
                    true_val[(ry, rx)] = 1  # All instances are 1, so 1 is the true value

        # Reconstruct
        result = []
        for r in range(rows):
            row = []
            for c in range(cols):
                row.append(true_val[(r % py, c % px)])
            result.append(row)
        return result

    # Try periods from small to large
    for py in range(1, rows + 1):
        for px in range(1, cols + 1):
            result = try_period(py, px)
            if result is not None:
                # Verify: every non-1 cell in original must match
                valid = True
                for r in range(rows):
                    for c in range(cols):
                        if grid[r][c] != 1 and result[r][c] != grid[r][c]:
                            valid = False
                            break
                    if not valid:
                        break
                if valid and result != grid:  # Must actually fix something
                    return result
    return grid
