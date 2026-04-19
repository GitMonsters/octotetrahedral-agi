from collections import Counter

def transform(input_grid: list[list[int]]) -> list[list[int]]:
    H = len(input_grid)
    W = len(input_grid[0])
    
    all_colors = Counter(v for row in input_grid for v in row)
    bg = all_colors.most_common(1)[0][0]
    
    # Find axes
    best_ha, best_ha_score = 0, -1
    for ha2 in range(W * 2 + 1):
        ha = ha2 / 2
        score = 0
        for r in range(H):
            for c in range(W):
                mc = round(2 * ha - c)
                if 0 <= mc < W and mc != c and input_grid[r][c] == input_grid[r][mc]:
                    score += 1
        if score > best_ha_score:
            best_ha_score = score; best_ha = ha
    
    best_va, best_va_score = 0, -1
    for va2 in range(H * 2 + 1):
        va = va2 / 2
        score = 0
        for r in range(H):
            for c in range(W):
                mr = round(2 * va - r)
                if 0 <= mr < H and mr != r and input_grid[r][c] == input_grid[mr][c]:
                    score += 1
        if score > best_va_score:
            best_va_score = score; best_va = va
    
    va, ha = best_va, best_ha
    
    def get_d4_images(r, c):
        dr = r - va
        dc = c - ha
        transforms = [
            (va + dc, ha - dr),
            (va - dr, ha - dc),
            (va - dc, ha + dr),
            (va - dr, ha + dc),
            (va + dr, ha - dc),
            (va + dc, ha + dr),
            (va - dc, ha - dr),
        ]
        result = []
        for tr, tc in transforms:
            tri, tci = round(tr), round(tc)
            if 0 <= tri < H and 0 <= tci < W:
                result.append(input_grid[tri][tci])
        return result
    
    def apply_fill(fill_color):
        out = [row[:] for row in input_grid]
        n_changes = 0
        for r in range(H):
            for c in range(W):
                if input_grid[r][c] != fill_color:
                    continue
                images = get_d4_images(r, c)
                non_fill = [v for v in images if v != fill_color]
                if non_fill:
                    vote = Counter(non_fill).most_common(1)[0][0]
                    out[r][c] = vote
                    if vote != fill_color:
                        n_changes += 1
        return out, n_changes
    
    def score_full_d4(grid):
        """Score how D4-symmetric the result is"""
        score = 0
        total = 0
        for r in range(H):
            for c in range(W):
                dr = r - va
                dc = c - ha
                ops = [
                    (round(va + dc), round(ha - dr)),
                    (round(va - dr), round(ha - dc)),
                    (round(va - dc), round(ha + dr)),
                    (round(va - dr), round(ha + dc)),
                    (round(va + dr), round(ha - dc)),
                    (round(va + dc), round(ha + dr)),
                    (round(va - dc), round(ha - dr)),
                ]
                for mr, mc in ops:
                    if 0 <= mr < H and 0 <= mc < W:
                        total += 1
                        if grid[r][c] == grid[mr][mc]:
                            score += 1
        return score / total if total > 0 else 0
    
    # Try candidate fill colors
    candidates = list(set([bg] + [c for c in all_colors if c != bg]))
    best_out = None
    best_score = -1
    
    for fill in candidates:
        out, n_changes = apply_fill(fill)
        sym = score_full_d4(out)
        # Prefer: high symmetry + fewer changes (normalized)
        total_cells = H * W
        change_penalty = n_changes / total_cells  # smaller is better
        combined = sym - 0.1 * change_penalty
        if combined > best_score:
            best_score = combined
            best_out = out
    
    return best_out
