"""ARC puzzle 332f06d7 solver.

Rule: The Black marker (0) "snakes" through the Blue (1) shape like a worm,
turning at walls, following the shape's winding path. Old Black fills with Blue,
and the marker's final resting position becomes the new Black.
Red (2) stays unless the marker slides over it.
"""
import copy


def find_cells(grid: list[list[int]], color: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(len(grid)) for c in range(len(grid[0])) if grid[r][c] == color}


def transform(input_grid: list[list[int]]) -> list[list[int]]:
    R, C = len(input_grid), len(input_grid[0])

    red_cells = find_cells(input_grid, 2)
    black_cells = find_cells(input_grid, 0)
    blue_cells = find_cells(input_grid, 1)

    # Marker bounding box
    br1 = min(r for r, c in black_cells)
    bc1 = min(c for r, c in black_cells)
    br2 = max(r for r, c in black_cells)
    bc2 = max(c for r, c in black_cells)
    mh = br2 - br1 + 1
    mw = bc2 - bc1 + 1

    # Valid cells for sliding: all non-background
    valid = blue_cells | black_cells | red_cells

    def can_place(r0: int, c0: int) -> bool:
        for dr in range(mh):
            for dc in range(mw):
                if (r0 + dr, c0 + dc) not in valid:
                    return False
        return True

    def slide_length(pos: tuple[int, int], d: tuple[int, int]) -> int:
        count = 0
        r0, c0 = pos
        while True:
            r0 += d[0]
            c0 += d[1]
            if can_place(r0, c0):
                count += 1
            else:
                break
        return count

    # Determine initial slide direction (from Black toward adjacent Blue)
    face_scores: dict[tuple[int, int], int] = {}
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        count = 0
        if dr != 0:
            check_r = br1 - 1 if dr < 0 else br2 + 1
            for ddc in range(mw):
                if (check_r, bc1 + ddc) in blue_cells:
                    count += 1
        else:
            check_c = bc1 - 1 if dc < 0 else bc2 + 1
            for ddr in range(mh):
                if (br1 + ddr, check_c) in blue_cells:
                    count += 1
        if count > 0:
            face_scores[(dr, dc)] = count

    direction = max(face_scores, key=face_scores.get)
    pos = (br1, bc1)

    def turn_left(d: tuple[int, int]) -> tuple[int, int]:
        return (-d[1], d[0])

    def turn_right(d: tuple[int, int]) -> tuple[int, int]:
        return (d[1], -d[0])

    # Snake the marker through the shape
    for _ in range(1000):
        # Slide as far as possible in current direction
        while True:
            nr, nc = pos[0] + direction[0], pos[1] + direction[1]
            if can_place(nr, nc):
                pos = (nr, nc)
            else:
                break

        # Turn: pick the direction with the longer available slide
        ld = turn_left(direction)
        rd = turn_right(direction)
        l_len = slide_length(pos, ld)
        r_len = slide_length(pos, rd)

        if l_len == 0 and r_len == 0:
            break
        elif l_len >= r_len:
            direction = ld
        else:
            direction = rd

    # Build output: fill old Black with Blue, place new Black at final position
    out = copy.deepcopy(input_grid)
    for r, c in black_cells:
        out[r][c] = 1
    for dr in range(mh):
        for dc in range(mw):
            out[pos[0] + dr][pos[1] + dc] = 0

    return out


if __name__ == "__main__":
    # Training examples
    examples = [
        {"input": [[3,2,2,3,3,3,3,3,3,3,3,3],[3,2,2,3,3,3,3,3,3,3,3,3],[3,1,1,3,3,3,3,3,3,3,3,3],[3,1,1,3,1,1,1,1,1,1,3,3],[3,1,1,3,1,1,1,1,1,1,3,3],[3,1,1,3,1,1,3,1,1,1,3,3],[3,1,1,3,1,1,3,1,1,1,1,3],[3,1,1,1,1,1,3,1,1,1,1,3],[3,1,1,1,1,1,3,3,1,1,3,3],[3,1,1,1,3,3,3,3,0,0,3,3],[3,3,3,3,3,3,3,3,0,0,3,3],[3,3,3,3,3,3,3,3,3,3,3,3]],
         "output": [[3,0,0,3,3,3,3,3,3,3,3,3],[3,0,0,3,3,3,3,3,3,3,3,3],[3,1,1,3,3,3,3,3,3,3,3,3],[3,1,1,3,1,1,1,1,1,1,3,3],[3,1,1,3,1,1,1,1,1,1,3,3],[3,1,1,3,1,1,3,1,1,1,3,3],[3,1,1,3,1,1,3,1,1,1,1,3],[3,1,1,1,1,1,3,1,1,1,1,3],[3,1,1,1,1,1,3,3,1,1,3,3],[3,1,1,1,3,3,3,3,1,1,3,3],[3,3,3,3,3,3,3,3,1,1,3,3],[3,3,3,3,3,3,3,3,3,3,3,3]]},
        {"input": [[3,3,3,3,3,3,3,3,3,3,3,3,3,3],[3,3,3,3,3,3,3,3,3,3,3,3,3,3],[3,3,3,3,3,1,1,1,1,1,0,0,3,3],[3,3,3,3,3,1,1,1,1,1,0,0,3,3],[3,3,3,3,3,1,1,3,3,3,3,3,3,3],[3,3,3,3,3,1,1,3,3,3,3,3,3,3],[3,3,3,3,3,1,1,3,3,3,3,3,3,3],[3,3,3,3,3,1,1,1,1,1,3,3,3,3],[3,3,3,3,3,1,1,1,1,1,3,3,3,3],[3,3,3,3,3,3,3,3,3,1,3,3,3,3],[3,1,1,1,1,1,1,1,1,1,3,3,3,3],[3,1,1,1,1,1,1,1,1,1,3,3,3,3],[3,2,2,3,3,3,3,3,3,3,3,3,3,3],[3,2,2,3,3,3,3,3,3,3,3,3,3,3]],
         "output": [[3,3,3,3,3,3,3,3,3,3,3,3,3,3],[3,3,3,3,3,3,3,3,3,3,3,3,3,3],[3,3,3,3,3,1,1,1,1,1,1,1,3,3],[3,3,3,3,3,1,1,1,1,1,1,1,3,3],[3,3,3,3,3,1,1,3,3,3,3,3,3,3],[3,3,3,3,3,1,1,3,3,3,3,3,3,3],[3,3,3,3,3,1,1,3,3,3,3,3,3,3],[3,3,3,3,3,1,1,1,0,0,3,3,3,3],[3,3,3,3,3,1,1,1,0,0,3,3,3,3],[3,3,3,3,3,3,3,3,3,1,3,3,3,3],[3,1,1,1,1,1,1,1,1,1,3,3,3,3],[3,1,1,1,1,1,1,1,1,1,3,3,3,3],[3,2,2,3,3,3,3,3,3,3,3,3,3,3],[3,2,2,3,3,3,3,3,3,3,3,3,3,3]]},
        {"input": [[3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],[2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,3],[2,2,2,1,1,1,1,3,3,1,1,1,1,1,1,3],[2,2,2,1,1,3,3,3,3,3,1,1,1,1,1,3],[3,3,3,3,3,3,3,3,3,3,3,3,3,1,1,3],[3,3,3,1,1,1,1,1,1,1,3,3,3,1,1,3],[3,3,3,1,1,1,1,1,1,1,3,3,3,1,1,3],[3,3,3,1,1,1,1,1,1,1,3,3,3,1,1,3],[3,3,1,1,1,1,3,1,1,1,3,3,3,1,1,3],[3,3,1,1,1,1,3,1,1,3,3,3,1,1,1,3],[3,3,1,1,1,1,3,1,1,3,3,3,1,1,1,3],[3,3,1,1,1,3,3,1,1,3,3,3,1,1,1,3],[3,0,0,0,1,3,3,1,1,1,1,1,1,1,1,3],[3,0,0,0,1,3,3,1,1,1,1,1,1,1,1,3],[3,0,0,0,1,3,3,1,1,1,1,1,1,1,1,3],[3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3]],
         "output": [[3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],[2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,3],[2,2,2,1,1,1,1,3,3,1,1,1,1,1,1,3],[2,2,2,1,1,3,3,3,3,3,1,1,1,1,1,3],[3,3,3,3,3,3,3,3,3,3,3,3,3,1,1,3],[3,3,3,1,1,1,1,1,1,1,3,3,3,1,1,3],[3,3,3,1,1,1,1,0,0,0,3,3,3,1,1,3],[3,3,3,1,1,1,1,0,0,0,3,3,3,1,1,3],[3,3,1,1,1,1,3,0,0,0,3,3,3,1,1,3],[3,3,1,1,1,1,3,1,1,3,3,3,1,1,1,3],[3,3,1,1,1,1,3,1,1,3,3,3,1,1,1,3],[3,3,1,1,1,3,3,1,1,3,3,3,1,1,1,3],[3,1,1,1,1,3,3,1,1,1,1,1,1,1,1,3],[3,1,1,1,1,3,3,1,1,1,1,1,1,1,1,3],[3,1,1,1,1,3,3,1,1,1,1,1,1,1,1,3],[3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3]]},
        {"input": [[3,3,3,3,3,3,3,3,3,3],[3,0,1,1,3,3,3,3,3,3],[3,3,3,1,3,1,1,1,3,3],[3,3,3,1,3,1,3,1,3,3],[3,3,3,1,1,1,3,1,3,3],[3,3,3,3,3,3,3,1,3,3],[3,3,3,3,1,1,1,1,3,3],[3,3,3,3,1,3,3,3,3,3],[3,3,3,3,1,3,3,3,3,3],[3,3,3,3,2,3,3,3,3,3]],
         "output": [[3,3,3,3,3,3,3,3,3,3],[3,1,1,1,3,3,3,3,3,3],[3,3,3,1,3,1,1,1,3,3],[3,3,3,1,3,1,3,1,3,3],[3,3,3,1,1,1,3,1,3,3],[3,3,3,3,3,3,3,1,3,3],[3,3,3,3,1,1,1,1,3,3],[3,3,3,3,1,3,3,3,3,3],[3,3,3,3,1,3,3,3,3,3],[3,3,3,3,0,3,3,3,3,3]]}
    ]

    # Verify training examples
    all_pass = True
    for i, ex in enumerate(examples):
        result = transform(ex["input"])
        match = result == ex["output"]
        all_pass = all_pass and match
        print(f"Example {i}: {'PASS' if match else 'FAIL'}")

    if all_pass:
        # Apply to test input
        test_input = [[3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],[3,1,1,1,1,1,1,1,3,3,3,3,3,3,1,1,1,1,1,3],[3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3],[3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3],[3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3],[3,1,1,1,1,3,3,3,3,3,3,3,3,3,1,1,1,1,1,3],[3,1,1,1,1,1,1,1,1,1,1,1,1,3,3,1,1,1,1,3],[3,1,1,1,1,1,1,1,1,1,1,1,1,3,3,3,1,1,1,3],[3,1,1,1,1,1,1,1,1,1,1,1,1,3,3,3,3,1,1,3],[3,1,1,1,1,1,1,1,1,1,1,1,1,3,3,3,3,1,1,3],[3,3,3,3,3,3,3,3,3,1,1,1,1,3,3,3,1,1,1,3],[3,0,0,0,0,1,1,1,3,1,1,1,1,3,3,1,1,1,1,3],[3,0,0,0,0,1,1,1,3,1,1,1,1,3,3,1,1,1,1,3],[3,0,0,0,0,1,1,1,3,1,1,1,1,1,3,1,1,1,1,3],[3,0,0,0,0,1,1,1,3,1,1,1,1,1,3,1,2,2,2,2],[3,3,3,3,1,1,1,1,1,1,1,1,1,1,3,1,2,2,2,2],[3,3,3,3,1,1,1,1,1,1,1,1,1,3,3,1,2,2,2,2],[3,3,3,1,1,1,1,1,1,1,1,1,1,3,3,1,2,2,2,2],[3,3,3,1,1,1,1,1,1,1,1,1,1,1,3,3,3,3,3,3],[3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3]]

        expected_output = [[3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],[3,1,1,1,0,0,0,0,3,3,3,3,3,3,1,1,1,1,1,3],[3,1,1,1,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,3],[3,1,1,1,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,3],[3,1,1,1,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,3],[3,1,1,1,1,3,3,3,3,3,3,3,3,3,1,1,1,1,1,3],[3,1,1,1,1,1,1,1,1,1,1,1,1,3,3,1,1,1,1,3],[3,1,1,1,1,1,1,1,1,1,1,1,1,3,3,3,1,1,1,3],[3,1,1,1,1,1,1,1,1,1,1,1,1,3,3,3,3,1,1,3],[3,1,1,1,1,1,1,1,1,1,1,1,1,3,3,3,3,1,1,3],[3,3,3,3,3,3,3,3,3,1,1,1,1,3,3,3,1,1,1,3],[3,1,1,1,1,1,1,1,3,1,1,1,1,3,3,1,1,1,1,3],[3,1,1,1,1,1,1,1,3,1,1,1,1,3,3,1,1,1,1,3],[3,1,1,1,1,1,1,1,3,1,1,1,1,1,3,1,1,1,1,3],[3,1,1,1,1,1,1,1,3,1,1,1,1,1,3,1,2,2,2,2],[3,3,3,3,1,1,1,1,1,1,1,1,1,1,3,1,2,2,2,2],[3,3,3,3,1,1,1,1,1,1,1,1,1,3,3,1,2,2,2,2],[3,3,3,1,1,1,1,1,1,1,1,1,1,3,3,1,2,2,2,2],[3,3,3,1,1,1,1,1,1,1,1,1,1,1,3,3,3,3,3,3],[3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3]]

        test_result = transform(test_input)
        test_match = test_result == expected_output
        print(f"\nTest: {'PASS' if test_match else 'FAIL'}")

        if test_match:
            print("\nSOLVED")
        else:
            print("\nFAILED")
            for r in range(len(expected_output)):
                for c in range(len(expected_output[0])):
                    if test_result[r][c] != expected_output[r][c]:
                        print(f"  ({r},{c}): got {test_result[r][c]}, exp {expected_output[r][c]}")
    else:
        print("\nFAILED - training examples don't pass")
