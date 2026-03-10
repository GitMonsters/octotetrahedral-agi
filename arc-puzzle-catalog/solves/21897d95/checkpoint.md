# ARC Puzzle 21897d95 — Checkpoint

## Status: Unsolved

**Task file**: `dataset/tasks/21897d95.json`
**Training examples**: 4
**Test examples**: 2
**Test outputs available**: Yes

## Examples

- **Train 0**: 16×12 → 12×16 (DIMS CHANGE)
- **Train 1**: 10×10 → 10×10 (same dims)
- **Train 2**: 14×9 → 9×14 (DIMS CHANGE)
- **Train 3**: 10×10 → 10×10 (same dims)
- **Test 0**: 30×30 → 30×30 (same dims)
- **Test 1**: 24×24 → 24×24 (same dims)

**Input colors**: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
**Output colors**: [0, 2, 3, 4, 5, 6, 7, 8, 9]
**Colors in input only**: [1]

## Grids (Train 0)

### Input
```
666667774444
666667774444
666167774444
666317774999
666167774449
666667774449
666667174444
666667114444
666667174444
666667774444
771777777777
710177777111
777777777717
888888888888
888888888888
888888888888
```

### Output
```
7777777777333777
7777777777333777
7777777777333777
7777777777333777
3333333333333777
3333333333333777
3333333333333777
0000000000333777
0000000000333777
0000000000333777
0000000000333777
0000000000333777
```

## Analysis (In Progress)

### Transformation Rule

1. **Block Grid**: Input is divided into rectangular blocks of uniform color forming a regular grid.
   Block boundaries found where row/column color signatures change after cleaning markers.

2. **Arrows/Markers**: 4-cell T-shaped patterns (3 collinear + 1 perpendicular) indicate color flow.
   - Colors: mostly `1` cells, but some have a "payload" color cell
   - L-shapes (5+ cells) are NOT valid arrows

3. **Arrow Direction**: The perpendicular arm points in the flow direction.
   Missing arm → opposite direction: `not left → RIGHT`, `not right → LEFT`, etc.

4. **Color Flow**: Each arrow transfers color from source block to target block's
   connected component (same-color region in block grid).
   - Flow color = payload color if present, else source block color

5. **Non-square grids rotate 90° CCW**: `new(i,j) = old(j, m-1-i)`

### Current Solver Status

- **Train 0**: ✅ PASS
- **Train 1**: ❌ 2 mismatches at (3,4) and (4,4) — block grid too granular (4×5 instead of 3×4)
- **Train 2**: ❌ 106 mismatches — color 1 wrongly classified as block color (threshold too low for 14×9)
- **Train 3**: ✅ PASS
- **Test 0/1**: ❌ FAIL

### Known Issues

- Block color threshold `min(rows,cols)//2` is too aggressive for small grids (Train 2: threshold=4, includes color 1)
- Train 1 block grid is 4×5 instead of expected 3×4 (spurious boundary at rows 3-4)
- Connected component propagation verified correct for Train 3
- Solver at `/tmp/solve_21897d95.py`

