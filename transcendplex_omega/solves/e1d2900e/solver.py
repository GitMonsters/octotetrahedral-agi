def solve(grid: list[list[int]]) -> list[list[int]]:
    """Move scattered 1s to be adjacent to their nearest 2x2 block of 2s.

    A 1 is attracted to a block if it shares a row or column with that block.
    Multi-block sharers are always attracted. Single-block sharers within
    axis distance <= 6 are attracted. Beyond that, a 1 is only attracted if
    its block would otherwise have zero associated 1s (orphan rescue).
    """
    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]

    # Find 2x2 blocks of 2s (by top-left corner)
    blocks: list[tuple[int, int]] = []
    block_cells: set[tuple[int, int]] = set()
    for r in range(rows - 1):
        for c in range(cols - 1):
            if (r, c) not in block_cells and grid[r][c] == 2 and grid[r][c+1] == 2 and grid[r+1][c] == 2 and grid[r+1][c+1] == 2:
                blocks.append((r, c))
                block_cells.update([(r, c), (r, c+1), (r+1, c), (r+1, c+1)])

    ones = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 1]

    def get_sharing(pr: int, pc: int) -> list[tuple[int, int]]:
        return list({(br, bc) for br, bc in blocks
                     if pr in (br, br + 1) or pc in (bc, bc + 1)})

    def axis_dist(pr: int, pc: int, br: int, bc: int) -> int:
        if pr in (br, br + 1):
            return max(bc - pc, pc - (bc + 1), 0)
        return max(br - pr, pr - (br + 1), 0)

    def nearest_block(pr: int, pc: int, candidates: list[tuple[int, int]]) -> tuple[int, int]:
        return min(candidates, key=lambda b: (pr - b[0] - 0.5)**2 + (pc - b[1] - 0.5)**2)

    def place(pr: int, pc: int, br: int, bc: int) -> tuple[int, int]:
        cx, cy = br + 0.5, bc + 0.5
        dr, dc = pr - cx, pc - cy
        if abs(dr) > abs(dc):
            col = bc if pc <= cy else bc + 1
            return (br - 1 if dr < 0 else br + 2, col)
        row = br if pr <= cx else br + 1
        return (row, bc - 1 if dc < 0 else bc + 2)

    # Phase 1: classify 1s
    attracted: dict[tuple[int, int], tuple[int, int]] = {}
    pending: dict[tuple[int, int], tuple[int, int]] = {}

    for pr, pc in ones:
        sharing = get_sharing(pr, pc)
        if not sharing:
            continue
        if len(sharing) >= 2:
            attracted[(pr, pc)] = nearest_block(pr, pc, sharing)
        else:
            br, bc = sharing[0]
            if axis_dist(pr, pc, br, bc) <= 6:
                attracted[(pr, pc)] = (br, bc)
            else:
                pending[(pr, pc)] = (br, bc)

    # Phase 2: rescue orphaned blocks
    block_counts = {b: 0 for b in blocks}
    for b in attracted.values():
        block_counts[b] += 1

    for block in blocks:
        if block_counts[block] == 0:
            for pos, blk in list(pending.items()):
                if blk == block:
                    attracted[pos] = blk
                    del pending[pos]

    # Phase 3: apply movements
    for (pr, pc), (br, bc) in attracted.items():
        new_r, new_c = place(pr, pc, br, bc)
        result[pr][pc] = 0
        result[new_r][new_c] = 1

    return result
