import math
from collections import Counter

def transform(input_grid: list[list[int]]) -> list[list[int]]:
    rows, cols = len(input_grid), len(input_grid[0])
    
    # Count colors; background = most frequent
    color_counts: Counter = Counter()
    for r in range(rows):
        for c in range(cols):
            color_counts[input_grid[r][c]] += 1
    bg = color_counts.most_common(1)[0][0]
    
    # Each non-bg color -> most-square rectangle, sorted by size ascending
    def get_rect_dims(n: int) -> tuple[int, int]:
        s = int(math.isqrt(n))
        for w in range(s, 0, -1):
            if n % w == 0:
                return (n // w, w)  # (height, width) with h >= w
        return (n, 1)
    
    rects = []
    for color, count in color_counts.items():
        if color != bg:
            h, w = get_rect_dims(count)
            rects.append((count, color, h, w))
    rects.sort(key=lambda x: x[0])
    
    # Build output: rectangles left-to-right, bottom-aligned, 1-col gap
    max_h = max(r[2] for r in rects)
    total_w = sum(r[3] for r in rects) + len(rects) - 1
    output = [[bg] * total_w for _ in range(max_h)]
    
    x = 0
    for _, color, h, w in rects:
        sr = max_h - h
        for r in range(h):
            for c in range(w):
                output[sr + r][x + c] = color
        x += w + 1
    
    return output

# --- Test ---
examples = [
    ([[4,4,4,4,4,7,7],[4,7,7,7,4,7,5],[4,7,1,7,4,7,5],[4,7,7,7,4,7,7],[4,4,4,4,4,7,5],[7,7,7,7,7,7,5],[5,5,5,5,5,7,7]],
     [[7,7,7,7,7,7,4,4,4,4],[7,7,5,5,5,7,4,4,4,4],[7,7,5,5,5,7,4,4,4,4],[1,7,5,5,5,7,4,4,4,4]]),
    ([[9,9,9,9],[7,7,9,7],[9,9,9,9],[7,7,7,7],[7,6,7,6],[7,6,7,6],[7,7,7,7]],
     [[7,7,7,9,9,9],[6,6,7,9,9,9],[6,6,7,9,9,9]]),
]

all_ok = True
for i, (inp, exp) in enumerate(examples):
    out = transform(inp)
    if out == exp:
        print(f"Example {i}: PASS")
    else:
        print(f"Example {i}: FAIL")
        print(f"  Expected: {exp}")
        print(f"  Got:      {out}")
        all_ok = False

# Test input
test_in = [[7,7,7,0,0,0,7,7,7,4],[0,0,7,0,7,0,7,9,7,4],[0,0,7,0,0,0,7,7,7,4],[7,7,7,7,7,7,7,0,7,4],[7,7,7,7,7,0,0,0,7,7]]
test_exp = [[7,7,7,7,7,0,0,0,0],[7,7,7,7,7,0,0,0,0],[7,7,4,4,7,0,0,0,0],[9,7,4,4,7,0,0,0,0]]
test_out = transform(test_in)
if test_out == test_exp:
    print("Test: PASS")
else:
    print("Test: FAIL")
    print(f"  Expected: {test_exp}")
    print(f"  Got:      {test_out}")
    all_ok = False

print("SOLVED" if all_ok else "FAILED")
