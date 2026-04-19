import numpy as np

def transform(input_grid: list[list[int]]) -> list[list[int]]:
    """Fill black (0) holes to restore transpose-symmetric matrix using rank constraint."""
    inp = np.array(input_grid)
    H, W = inp.shape
    work = inp.copy()

    # Step 1: Fill black cells from transpose position
    for r in range(H):
        for c in range(W):
            if work[r][c] == 0 and work[c][r] != 0:
                work[r][c] = work[c][r]

    # Step 2: For remaining zeros, use low-rank matrix completion
    zeros = [(r, c) for r in range(H) for c in range(W) if work[r][c] == 0]
    if zeros:
        known_rows = [r for r in range(H) if all(work[r][c] != 0 for c in range(W))]
        A = work[known_rows, :].astype(float)

        unknown_rows = sorted(set(r for r, c in zeros))
        for ur in unknown_rows:
            known_cols = [c for c in range(W) if work[ur][c] != 0]
            unknown_cols = [c for c in range(W) if work[ur][c] == 0]

            b = work[ur, known_cols].astype(float)
            M = A[:, known_cols].T
            coeffs, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
            predicted = A[:, unknown_cols].T @ coeffs
            predicted_int = np.round(predicted).astype(int)

            for i, uc in enumerate(unknown_cols):
                work[ur][uc] = predicted_int[i]

    return work.tolist()


if __name__ == "__main__":
    import json

    with open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/981571dc.json") as f:
        puzzle = json.load(f)

    # Verify on all training examples
    all_pass = True
    for ti, ex in enumerate(puzzle["train"]):
        result = transform(ex["input"])
        expected = ex["output"]
        if result == expected:
            print(f"Example {ti}: PASS")
        else:
            print(f"Example {ti}: FAIL")
            diffs = sum(1 for r in range(len(result)) for c in range(len(result[0]))
                        if result[r][c] != expected[r][c])
            print(f"  {diffs} differences")
            all_pass = False

    # Run on test
    test_result = transform(puzzle["test"][0]["input"])
    print(f"\nTest output shape: {len(test_result)}x{len(test_result[0])}")
    print(f"Test output zeros: {sum(1 for r in test_result for c in r if c == 0)}")

    if all_pass:
        print("\nSOLVED")
    else:
        print("\nFAILED")
