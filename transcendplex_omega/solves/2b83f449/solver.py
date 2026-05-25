import json, sys

def solve(grid):
    rows, cols = len(grid), len(grid[0])
    out = [row[:] for row in grid]

    # Find centers of 777 groups on odd rows
    centers = {}
    for r in range(rows):
        if r % 2 == 1:
            cs = []
            c = 0
            while c < cols - 2:
                if grid[r][c] == 7 and grid[r][c+1] == 7 and grid[r][c+2] == 7:
                    cs.append(c + 1)
                    c += 3
                else:
                    c += 1
            centers[r] = cs

    # Replace 777 with 868 on odd rows
    for r in centers:
        for c in centers[r]:
            out[r][c-1] = 8
            out[r][c] = 6
            out[r][c+1] = 8

    # Process even rows
    for r in range(0, rows, 2):
        above = set(centers.get(r-1, []))
        below = set(centers.get(r+1, []))
        all_6_cols = above | below

        for c in all_6_cols:
            out[r][c] = 6

        zeros = sorted(c for c in range(cols) if grid[r][c] == 0)

        # Build segments between 0s and grid edges
        boundaries = [-1] + zeros + [cols]
        segments = []
        for i in range(len(boundaries) - 1):
            s = boundaries[i] + 1
            e = boundaries[i+1] - 1
            if s <= e:
                segments.append((s, e))

        is_last = (r + 2 >= rows)

        for seg_s, seg_e in segments:
            is_full_width = (seg_s == 0 and seg_e == cols - 1)

            sixes = []
            has_below = False
            for c in range(seg_s, seg_e + 1):
                if c in above:
                    sixes.append((c, 'A'))
                elif c in below:
                    sixes.append((c, 'B'))
                    has_below = True

            if not sixes:
                continue

            # Remove all input 3s in this segment
            for c in range(seg_s, seg_e + 1):
                if grid[r][c] == 3 and c not in all_6_cols:
                    out[r][c] = 8

            leftmost_c, leftmost_s = sixes[0]
            rightmost_c, rightmost_s = sixes[-1]

            # Pair at grid edge if: not last row AND (full-width OR all-from-above)
            can_pair = not is_last and (is_full_width or not has_below)

            if leftmost_s == 'A':
                out[r][seg_s] = 3
                if can_pair and seg_s == 0:
                    out[r][1] = 3

            if rightmost_s == 'A':
                out[r][seg_e] = 3
                if can_pair and seg_e == cols - 1:
                    out[r][seg_e - 1] = 3

        # 0-boundary suppression: when both sides have from-above 6
        for z in zeros:
            left_seg = right_seg = None
            for seg_s, seg_e in segments:
                if seg_e == z - 1:
                    left_seg = (seg_s, seg_e)
                if seg_s == z + 1:
                    right_seg = (seg_s, seg_e)

            if left_seg and right_seg:
                left_As = [c for c in range(left_seg[0], left_seg[1]+1) if c in above]
                right_As = [c for c in range(right_seg[0], right_seg[1]+1) if c in above]

                if left_As and right_As:
                    left_has_edge = (left_seg[0] == 0 or left_seg[1] == cols - 1)
                    right_has_edge = (right_seg[0] == 0 or right_seg[1] == cols - 1)

                    if left_has_edge and right_has_edge:
                        pass  # no suppression
                    elif not left_has_edge:
                        out[r][left_seg[1]] = 8
                    elif not right_has_edge:
                        out[r][right_seg[0]] = 8
                    else:
                        # fallback: suppress nearer
                        left_nearest = max(left_As)
                        right_nearest = min(right_As)
                        if (z - left_nearest) <= (right_nearest - z):
                            out[r][left_seg[1]] = 8
                        else:
                            out[r][right_seg[0]] = 8

    return out

if __name__ == "__main__":
    with open("/Users/evanpieser/arc-puzzle-catalog/dataset/tasks/2b83f449.json") as f:
        data = json.load(f)
    all_pass = True
    for i, ex in enumerate(data["train"]):
        result = solve(ex["input"])
        expected = ex["output"]
        if result == expected:
            print(f"Train {i}: PASS")
        else:
            mismatches = sum(1 for r in range(len(expected)) for c in range(len(expected[0])) if result[r][c] != expected[r][c])
            print(f"Train {i}: FAIL - {mismatches} mismatches")
            for r in range(len(expected)):
                for c in range(len(expected[0])):
                    if result[r][c] != expected[r][c]:
                        print(f"  ({r},{c}): got {result[r][c]}, exp {expected[r][c]}, inp {ex['input'][r][c]}")
            all_pass = False
    for i, ex in enumerate(data["test"]):
        result = solve(ex["input"])
        expected = ex["output"]
        if result == expected:
            print(f"Test {i}: PASS")
        else:
            mismatches = sum(1 for r in range(len(expected)) for c in range(len(expected[0])) if result[r][c] != expected[r][c])
            print(f"Test {i}: FAIL - {mismatches} mismatches")
            for r in range(len(expected)):
                for c in range(len(expected[0])):
                    if result[r][c] != expected[r][c]:
                        print(f"  ({r},{c}): got {result[r][c]}, exp {expected[r][c]}, inp {ex['input'][r][c]}")
            all_pass = False
    if all_pass:
        print("\nAll examples pass!")
