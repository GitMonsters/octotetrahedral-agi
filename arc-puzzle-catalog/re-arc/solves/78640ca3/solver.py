from collections import Counter
from typing import List, Dict, Set, Tuple

def transform(input_grid: List[List[int]]) -> List[List[int]]:
    rows = len(input_grid)
    cols = len(input_grid[0])
    
    flat = [v for row in input_grid for v in row]
    bg = Counter(flat).most_common(1)[0][0]
    
    # Find horizontal separator rows
    h_seps: Dict[int, int] = {}
    for r in range(rows):
        if len(set(input_grid[r])) == 1 and input_grid[r][0] != bg:
            h_seps[r] = input_grid[r][0]
    
    # Find vertical separator columns
    v_seps: Dict[int, int] = {}
    for c in range(cols):
        col_vals = set(input_grid[r][c] for r in range(rows))
        if len(col_vals) == 1 and input_grid[0][c] != bg:
            v_seps[c] = input_grid[0][c]
    
    sep_rows = set(h_seps.keys())
    sep_cols = set(v_seps.keys())
    all_sep_colors = set(h_seps.values()) | set(v_seps.values())
    
    # Build output: bg + separator lines
    out = [[bg] * cols for _ in range(rows)]
    for r in sep_rows:
        for c in range(cols):
            out[r][c] = input_grid[r][c]
    for c in sep_cols:
        for r in range(rows):
            out[r][c] = input_grid[r][c]
    
    # Collect stray pixels (non-bg, not on sep row/col)
    stray: List[Tuple[int, int, int]] = []
    for r in range(rows):
        if r in sep_rows:
            continue
        for c in range(cols):
            if c in sep_cols:
                continue
            v = input_grid[r][c]
            if v != bg:
                stray.append((r, c, v))
    
    # Build row/col lookup for stray pixels
    stray_in_row: Dict[int, List[Tuple[int, int, int]]] = {}
    stray_in_col: Dict[int, List[Tuple[int, int, int]]] = {}
    for r, c, v in stray:
        stray_in_row.setdefault(r, []).append((r, c, v))
        stray_in_col.setdefault(c, []).append((r, c, v))
    
    def is_middle_h(sep_r: int) -> bool:
        """Check if h-separator has other h-separators on both sides."""
        has_above = any(r < sep_r for r in h_seps if r != sep_r)
        has_below = any(r > sep_r for r in h_seps if r != sep_r)
        return has_above and has_below
    
    def is_middle_v(sep_c: int) -> bool:
        has_left = any(c < sep_c for c in v_seps if c != sep_c)
        has_right = any(c > sep_c for c in v_seps if c != sep_c)
        return has_left and has_right
    
    def companion_filter_h(matching: List[Tuple[int, int]], sep_color: int) -> List[Tuple[int, int]]:
        """For h-sep edge: keep stray pixels that share ROW with different-colored non-bg pixel."""
        result = []
        for r, c in matching:
            row_pixels = stray_in_row.get(r, [])
            has_companion = any(v != sep_color for _, _, v in row_pixels)
            if has_companion:
                result.append((r, c))
        return result
    
    def companion_filter_v(matching: List[Tuple[int, int]], sep_color: int) -> List[Tuple[int, int]]:
        """For v-sep edge: keep stray pixels that share COLUMN with different-colored non-bg pixel."""
        result = []
        for r, c in matching:
            col_pixels = stray_in_col.get(c, [])
            has_companion = any(v != sep_color for _, _, v in col_pixels)
            if has_companion:
                result.append((r, c))
        return result
    
    def middle_filter_h(sep_r: int, sep_color: int):
        """For middle h-sep: find qualifying columns and assign sides."""
        # Get colors per column across all stray pixels
        col_entries: Dict[int, List[Tuple[int, int]]] = {}
        for r, c, v in stray:
            col_entries.setdefault(c, []).append((r, v))
        
        above_cols = set()
        below_cols = set()
        
        for c, entries in col_entries.items():
            colors_present = set(v for _, v in entries)
            if not all_sep_colors.issubset(colors_present):
                continue
            # Check no adjacent differently-colored stray pixels
            sorted_entries = sorted(entries, key=lambda x: x[0])
            has_adj_diff = False
            for i in range(len(sorted_entries) - 1):
                r1, c1 = sorted_entries[i]
                r2, c2 = sorted_entries[i + 1]
                if r2 - r1 == 1 and c1 != c2:
                    has_adj_diff = True
                    break
            if has_adj_diff:
                continue
            
            # Assign side: where are stray pixels of sep_color?
            above_sep = [r for r, v in entries if r < sep_r and v == sep_color]
            below_sep = [r for r, v in entries if r > sep_r and v == sep_color]
            
            if above_sep and not below_sep:
                above_cols.add(c)
            elif below_sep and not above_sep:
                below_cols.add(c)
            else:
                # Both sides have sep_color: use total stray count
                total_above = sum(1 for r, v in entries if r < sep_r)
                total_below = sum(1 for r, v in entries if r > sep_r)
                if total_above > total_below:
                    above_cols.add(c)
                else:
                    below_cols.add(c)
        
        return above_cols, below_cols
    
    def middle_filter_v(sep_c: int, sep_color: int):
        """For middle v-sep: find qualifying rows and assign sides."""
        row_entries: Dict[int, List[Tuple[int, int]]] = {}
        for r, c, v in stray:
            row_entries.setdefault(r, []).append((c, v))
        
        left_rows = set()
        right_rows = set()
        
        for r, entries in row_entries.items():
            colors_present = set(v for _, v in entries)
            if not all_sep_colors.issubset(colors_present):
                continue
            sorted_entries = sorted(entries, key=lambda x: x[0])
            has_adj_diff = False
            for i in range(len(sorted_entries) - 1):
                c1, v1 = sorted_entries[i]
                c2, v2 = sorted_entries[i + 1]
                if c2 - c1 == 1 and v1 != v2:
                    has_adj_diff = True
                    break
            if has_adj_diff:
                continue
            
            left_sep = [c for c, v in entries if c < sep_c and v == sep_color]
            right_sep = [c for c, v in entries if c > sep_c and v == sep_color]
            
            if left_sep and not right_sep:
                left_rows.add(r)
            elif right_sep and not left_sep:
                right_rows.add(r)
            else:
                total_left = sum(1 for c, v in entries if c < sep_c)
                total_right = sum(1 for c, v in entries if c > sep_c)
                if total_left > total_right:
                    left_rows.add(r)
                else:
                    right_rows.add(r)
        
        return left_rows, right_rows
    
    # Process horizontal separators
    for sep_r, sep_color in sorted(h_seps.items()):
        if is_middle_h(sep_r):
            above_cols, below_cols = middle_filter_h(sep_r, sep_color)
        else:
            matching = [(r, c) for r, c, v in stray if v == sep_color]
            matching = companion_filter_h(matching, sep_color)
            above_cols = set(c for r, c in matching if r < sep_r)
            below_cols = set(c for r, c in matching if r > sep_r)
        
        if sep_r > 0:
            for c in above_cols:
                out[sep_r - 1][c] = sep_color
        if sep_r < rows - 1:
            for c in below_cols:
                out[sep_r + 1][c] = sep_color
    
    # Process vertical separators
    for sep_c, sep_color in sorted(v_seps.items()):
        if is_middle_v(sep_c):
            left_rows, right_rows = middle_filter_v(sep_c, sep_color)
        else:
            matching = [(r, c) for r, c, v in stray if v == sep_color]
            matching = companion_filter_v(matching, sep_color)
            left_rows = set(r for r, c in matching if c < sep_c)
            right_rows = set(r for r, c in matching if c > sep_c)
        
        if sep_c > 0:
            for r in left_rows:
                out[r][sep_c - 1] = sep_color
        if sep_c < cols - 1:
            for r in right_rows:
                out[r][sep_c + 1] = sep_color
    
    return out
