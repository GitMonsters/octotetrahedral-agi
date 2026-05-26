#!/usr/bin/env python3
"""
ARC-AGI Task 412555fe Solver

Key observations from analysis:
1. There's a "filler" color that appears in input but mostly disappears in output
2. Non-filler cells remain exactly the same from input to output  
3. The transformation appears to involve point symmetry (180° rotation)
4. Filler cells are replaced based on their mirror position

Strategy:
- Detect filler by finding color with most symmetry breaks
- Keep non-filler cells unchanged
- For filler cells: copy from mirror if mirror is non-filler
- For both-filler cases: use nearest non-filler neighbor
"""

import json

def solve_task(inp):
    H, W = len(inp), len(inp[0])
    
    # Detect filler: color that breaks point symmetry most
    breaks = {}
    for r in range(H):
        for c in range(W):
            mr, mc = H - 1 - r, W - 1 - c
            if inp[r][c] != inp[mr][mc]:
                breaks[inp[r][c]] = breaks.get(inp[r][c], 0) + 1
    
    filler = max(breaks, key=breaks.get) if breaks else None
    if filler is None:
        return [row[:] for row in inp]
    
    out = [row[:] for row in inp]
    
    # Fill filler cells
    for r in range(H):
        for c in range(W):
            if inp[r][c] == filler:
                mr, mc = H - 1 - r, W - 1 - c
                
                # If mirror is not filler, copy from it
                if inp[mr][mc] != filler:
                    out[r][c] = inp[mr][mc]
                else:
                    # Both are filler - find nearest non-filler
                    found = False
                    for dist in range(1, max(H, W)):
                        if found:
                            break
                        for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                            nr, nc = r + dr*dist, c + dc*dist
                            if 0 <= nr < H and 0 <= nc < W and out[nr][nc] != filler:
                                out[r][c] = out[nr][nc]
                                found = True
                                break
    
    return out

if __name__ == "__main__":
    # Load data
    with open('/tmp/rearc45/412555fe.json') as f:
        data = json.load(f)
    
    # Test on training
    print("Training results:")
    for i, pair in enumerate(data['train']):
        pred = solve_task(pair['input'])
        exp = pair['output']
        matches = sum(1 for r in range(len(exp)) for c in range(len(exp[0]))
                      if pred[r][c] == exp[r][c])
        total = len(exp) * len(exp[0])
        print(f"  Pair {i}: {matches}/{total} ({100*matches/total:.1f}%)")
    
    # Solve test cases
    print("\nGenerating test solutions...")
    solutions = []
    for i, test in enumerate(data['test']):
        sol = solve_task(test['input'])
        solutions.append(sol)
        print(f"  Test {i}: {len(sol)}x{len(sol[0])}")
    
    # Save
    with open('412555fe_solution.json', 'w') as f:
        json.dump(solutions, f)
    
    print("\n✓ Solution saved to 412555fe_solution.json")
