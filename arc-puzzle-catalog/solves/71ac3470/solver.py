import json
from collections import Counter

# Learn separator color from training data
def _detect_separator():
    with open('/tmp/rearc45/71ac3470.json') as f:
        task = json.load(f)
    p = task['train'][0]
    inp, out = p['input'], p['output']
    H, W = len(inp), len(inp[0])
    flat = [c for row in inp for c in row]
    bg = Counter(flat).most_common(1)[0][0]
    
    # Check orientation
    top_bottom = any(inp[0][c] != bg for c in range(W)) or any(inp[H-1][c] != bg for c in range(W))
    left_right = any(inp[r][0] != bg for r in range(H)) or any(inp[r][W-1] != bg for r in range(H))
    
    if top_bottom:
        center_row = H // 2
        return out[center_row][0]  # center row value (uniform)
    elif left_right:
        center_col = W // 2
        for r in range(H):
            if inp[r][0] != bg or inp[r][W-1] != bg:
                return out[r][center_col]
    return bg

SEP = _detect_separator()

def transform(grid):
    H = len(grid)
    W = len(grid[0])
    flat = [c for row in grid for c in row]
    bg = Counter(flat).most_common(1)[0][0]
    
    out = [row[:] for row in grid]
    
    # Detect orientation
    top_bottom = any(grid[0][c] != bg for c in range(W)) or any(grid[H-1][c] != bg for c in range(W))
    left_right = any(grid[r][0] != bg for r in range(H)) or any(grid[r][W-1] != bg for r in range(H))
    
    if top_bottom:
        center_row = H // 2
        marked_cols = set()
        for c in range(W):
            top_val = grid[0][c]
            bot_val = grid[H-1][c]
            if top_val != bg:
                marked_cols.add(c)
                for r in range(0, center_row):
                    out[r][c] = top_val
            if bot_val != bg:
                marked_cols.add(c)
                for r in range(center_row + 1, H):
                    out[r][c] = bot_val
        # Center row stays as separator (which may equal bg)
        for c in range(W):
            out[center_row][c] = SEP
        
        # Gap midpoint on center row
        sorted_cols = sorted(marked_cols)
        if len(sorted_cols) >= 2:
            max_gap = 0
            gaps = []
            for i in range(len(sorted_cols) - 1):
                gap = sorted_cols[i+1] - sorted_cols[i]
                if gap > max_gap:
                    max_gap = gap
                    gaps = [(sorted_cols[i], sorted_cols[i+1])]
                elif gap == max_gap:
                    gaps.append((sorted_cols[i], sorted_cols[i+1]))
            for s, e in gaps:
                mid = (s + e) // 2
                out[center_row][mid] = SEP  # may be redundant if SEP=bg
    
    elif left_right:
        center_col = W // 2
        marked_rows = set()
        for r in range(H):
            left_val = grid[r][0]
            right_val = grid[r][W-1]
            if left_val != bg:
                marked_rows.add(r)
                for c in range(0, center_col):
                    out[r][c] = left_val
            if right_val != bg:
                marked_rows.add(r)
                for c in range(center_col + 1, W):
                    out[r][c] = right_val
        
        # Set separator at center column for marked rows
        for r in marked_rows:
            out[r][center_col] = SEP
        
        # Gap midpoint
        sorted_rows = sorted(marked_rows)
        if len(sorted_rows) >= 2:
            max_gap = 0
            gaps = []
            for i in range(len(sorted_rows) - 1):
                gap = sorted_rows[i+1] - sorted_rows[i]
                if gap > max_gap:
                    max_gap = gap
                    gaps = [(sorted_rows[i], sorted_rows[i+1])]
                elif gap == max_gap:
                    gaps.append((sorted_rows[i], sorted_rows[i+1]))
            for s, e in gaps:
                mid = (s + e) // 2
                out[mid][center_col] = SEP
    
    return out
