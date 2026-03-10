import json, sys
from collections import Counter


def solve(grid):
    rows, cols = len(grid), len(grid[0])
    bg = 8

    # Find the 9x9 decorated frame (bounding box of non-background cells)
    min_r, max_r, min_c, max_c = rows, -1, cols, -1
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg:
                min_r = min(min_r, r)
                max_r = max(max_r, r)
                min_c = min(min_c, c)
                max_c = max(max_c, c)

    R, C = min_r, min_c

    # Extract the 9x9 frame
    frame = [[grid[R + r][C + c] for c in range(9)] for r in range(9)]

    # Find dominant non-background color for the center pixel
    color_count: Counter = Counter()
    for r in range(9):
        for c in range(9):
            if frame[r][c] != bg:
                color_count[frame[r][c]] += 1
    dominant = color_count.most_common(1)[0][0]

    # Fill the 3x3 center hole by sub-sampling every-other pixel from ring 2
    # cell(3+i, 3+j) = frame(2+2i, 2+2j); center pixel uses dominant color
    for i in range(3):
        for j in range(3):
            val = frame[2 + 2 * i][2 + 2 * j]
            if val == bg:
                val = dominant
            frame[3 + i][3 + j] = val

    # Build output (copy of input)
    output = [row[:] for row in grid]

    # Write the completed frame back
    for r in range(9):
        for c in range(9):
            output[R + r][C + c] = frame[r][c]

    # Extend outward: place 8 additional 3x3 blocks at odd block-coordinates
    # Block (br, bc) maps to frame block (br % 3, bc % 3)
    for br in [-1, 1, 3]:
        for bc in [-1, 1, 3]:
            if 0 <= br <= 2 and 0 <= bc <= 2:
                continue  # skip the frame itself

            src_br = br % 3
            src_bc = bc % 3
            tr = R + 3 * br
            tc = C + 3 * bc

            for i in range(3):
                for j in range(3):
                    gr, gc = tr + i, tc + j
                    if 0 <= gr < rows and 0 <= gc < cols:
                        output[gr][gc] = frame[src_br * 3 + i][src_bc * 3 + j]

    return output


if __name__ == "__main__":
    with open("/Users/evanpieser/arc-puzzle-catalog/dataset/tasks/db0c5428.json") as f:
        data = json.load(f)

    all_pass = True
    for i, ex in enumerate(data["train"]):
        result = solve(ex["input"])
        expected = ex["output"]
        if result == expected:
            print(f"Train {i}: PASS ✓")
        else:
            mismatches = sum(
                1
                for r in range(len(expected))
                for c in range(len(expected[0]))
                if result[r][c] != expected[r][c]
            )
            print(f"Train {i}: FAIL - {mismatches} mismatches")
            count = 0
            for r in range(len(expected)):
                for c in range(len(expected[0])):
                    if result[r][c] != expected[r][c]:
                        print(f"  ({r},{c}): got {result[r][c]}, expected {expected[r][c]}")
                        count += 1
                        if count >= 15:
                            break
                if count >= 15:
                    break
            all_pass = False

    for i, ex in enumerate(data["test"]):
        result = solve(ex["input"])
        expected = ex["output"]
        if result == expected:
            print(f"Test {i}: PASS ✓")
        else:
            mismatches = sum(
                1
                for r in range(len(expected))
                for c in range(len(expected[0]))
                if result[r][c] != expected[r][c]
            )
            print(f"Test {i}: FAIL - {mismatches} mismatches")
            all_pass = False

    if all_pass:
        print("\nAll examples pass! ✓")
