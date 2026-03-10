import json, sys
from collections import Counter

def solve(grid):
    H = len(grid)
    W = len(grid[0])
    bg = Counter(grid[r][c] for r in range(H) for c in range(W)).most_common(1)[0][0]
    
    frames = []
    used = set()
    for r in range(H-4):
        for c in range(W-4):
            if (r,c) in used: continue
            bc = grid[r][c]
            if bc == bg: continue
            ok = True
            for i in range(5):
                for j in range(5):
                    if i == 0 or i == 4 or j == 0 or j == 4:
                        if grid[r+i][c+j] != bc:
                            ok = False; break
                if not ok: break
            if not ok: continue
            fc = grid[r+1][c+1]
            if fc == bc or fc == bg: continue
            if not all(grid[r+1+i][c+1+j] == fc for i in range(3) for j in range(3)):
                continue
            frames.append((r, c, bc, fc))
            for i in range(5):
                for j in range(5):
                    used.add((r+i, c+j))
    
    border_colors = list(set(f[2] for f in frames))
    if len(border_colors) != 2: return grid
    
    color_left = max(border_colors)
    color_right = min(border_colors)
    
    left_frames = sorted([f for f in frames if f[2] == color_left], key=lambda f: f[0])
    right_frames = sorted([f for f in frames if f[2] == color_right], key=lambda f: f[0])
    
    def find_chains(flist):
        n = len(flist)
        if n == 0: return [], []
        connected = [False] * (n-1)
        for i in range(n-1):
            if flist[i+1][0] < flist[i][0] + 5:
                connected[i] = True
        chains, standalones = [], []
        i = 0
        while i < n:
            chain = [i]
            while i < n-1 and connected[i]:
                i += 1; chain.append(i)
            if len(chain) > 1: chains.append(chain)
            else: standalones.append(chain[0])
            i += 1
        return chains, standalones
    
    left_chains, left_solo = find_chains(left_frames)
    right_chains, right_solo = find_chains(right_frames)
    
    out = [[bg]*W for _ in range(H)]
    
    def draw(r, c, bc, fc):
        for i in range(5):
            for j in range(5):
                if 0 <= r+i < H and 0 <= c+j < W:
                    out[r+i][c+j] = bc if (i==0 or i==4 or j==0 or j==4) else fc
    
    for chain in left_chains:
        f0, f1 = left_frames[chain[0]], left_frames[chain[1]]
        col = 0 if f0[1] < f1[1] else 5
        for idx in chain:
            r, _, bc, fc = left_frames[idx]
            draw(r, col, bc, fc)
            col = 5 if col == 0 else 0
    
    for idx in left_solo:
        r, _, bc, fc = left_frames[idx]
        draw(r, 0, bc, fc)
    
    for chain in right_chains:
        f0, f1 = right_frames[chain[0]], right_frames[chain[1]]
        col = W - 10 if f0[1] < f1[1] else W - 5
        for idx in chain:
            r, _, bc, fc = right_frames[idx]
            draw(r, col, bc, fc)
            col = W-5 if col == W-10 else W-10
    
    for idx in right_solo:
        r, _, bc, fc = right_frames[idx]
        draw(r, W-5, bc, fc)
    
    return out

if __name__ == '__main__':
    with open(sys.argv[1]) as f:
        task = json.load(f)
    for i, ex in enumerate(task['train']):
        result = solve(ex['input'])
        match = result == ex['output']
        print(f"Train {i}: {'PASS' if match else 'FAIL'}")
        if not match:
            import numpy as np
            r = np.array(result)
            e = np.array(ex['output'])
            diffs = np.argwhere(r != e)
            print(f"  {len(diffs)} diffs")
            for d in diffs[:10]:
                print(f"  ({d[0]},{d[1]}): got {r[d[0],d[1]]}, exp {e[d[0],d[1]]}")
