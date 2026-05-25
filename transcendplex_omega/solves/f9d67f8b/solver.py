"""
Solver for ARC-AGI puzzle f9d67f8b

The output grid has three verified symmetries:
1. Row reflection: output[i][j] = output[K-i][j]  (K = N+1, valid when K-i ∈ [0,N-1])
2. Col reflection: output[i][j] = output[i][K-j]  (K = N+1, valid when K-j ∈ [0,N-1])
3. Block transpose: output[i][j] = output[j][i]   (valid when one of {i,j} ∈ [0,N/2-8)
                                                     and the other ∈ [N/2-7, N/2])

Cells marked with 9 are "damaged" and reconstructed from symmetric counterparts.
"""

import json
from typing import List


def solve(grid: List[List[int]]) -> List[List[int]]:
    N = len(grid)
    K = N + 1  # symmetry offset (31 for 30x30)
    H = N // 2  # 15 for 30x30

    result = [row[:] for row in grid]

    # Generate all equivalent positions for (i, j) under the symmetry group
    def candidates(i: int, j: int) -> list:
        positions = []
        # Map to canonical range [0, H] using K-symmetry
        variants_i = [i]
        ri = K - i
        if 0 <= ri < N:
            variants_i.append(ri)
        variants_j = [j]
        rj = K - j
        if 0 <= rj < N:
            variants_j.append(rj)

        # All mirror combinations
        for vi in variants_i:
            for vj in variants_j:
                positions.append((vi, vj))
                # Transpose: valid when one canonical index ∈ [0,7] and other ∈ [8,15]
                ci = min(vi, K - vi) if 0 <= K - vi < N else vi
                cj = min(vj, K - vj) if 0 <= K - vj < N else vj
                if (ci < H - 7 and 7 < cj <= H) or (cj < H - 7 and 7 < ci <= H):
                    positions.append((vj, vi))

        return positions

    # Iterative resolution
    changed = True
    while changed:
        changed = False
        for i in range(N):
            for j in range(N):
                if result[i][j] != 9:
                    continue
                for ci, cj in candidates(i, j):
                    if 0 <= ci < N and 0 <= cj < N and result[ci][cj] != 9:
                        result[i][j] = result[ci][cj]
                        changed = True
                        break

    return result


if __name__ == "__main__":
    with open("/tmp/arc_task_f9d67f8b.json") as f:
        task = json.load(f)

    all_pass = True
    for idx, ex in enumerate(task["train"]):
        predicted = solve(ex["input"])
        if predicted == ex["output"]:
            print(f"Train {idx}: PASS")
        else:
            all_pass = False
            diffs = sum(
                1
                for i in range(len(predicted))
                for j in range(len(predicted[0]))
                if predicted[i][j] != ex["output"][i][j]
            )
            print(f"Train {idx}: FAIL ({diffs} cells differ)")

    if all_pass:
        print("\nAll training examples pass!")

    # Check for remaining 9s in test
    for idx, ex in enumerate(task["test"]):
        result = solve(ex["input"])
        nines = sum(1 for r in result for c in r if c == 9)
        print(f"\nTest {idx}: {nines} remaining 9s")
        if nines == 0:
            print("  All cells resolved!")
