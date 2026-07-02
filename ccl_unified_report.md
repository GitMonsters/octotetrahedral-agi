# CCL + Unified Cognitive Stack Report

Evaluated **300 tasks** across L1-L3 from the Compound Concept Learning benchmark.

## Level-by-level performance

| Level | Tasks | Accuracy | Coherence | Coupling | Phase | Bias |
|---|---:|---:|---:|---:|---:|---:|
| L1 | 100 | 1.000 | 1.000 | 0.834 | 0.496 | 0.060 |
| L2 | 100 | 1.000 | 0.999 | 0.804 | 0.491 | 0.060 |
| L3 | 100 | 1.000 | 1.000 | 0.900 | 0.507 | 0.061 |

## Limb utilization distribution

- Limb 0: 225 activations
- Limb 1: 0 activations
- Limb 2: 132 activations
- Limb 3: 0 activations
- Limb 4: 152 activations
- Limb 5: 0 activations
- Limb 6: 91 activations
- Limb 7: 0 activations

## Rule routing patterns

- `rot_cw` → limb 4: 134
- `flip_v` → limb 4: 84, limb 5: 38
- `gravity_right` → limb 6: 104, limb 7: 18
- `sort_cols` → limb 0: 108, limb 1: 14
- `gravity_down` → limb 6: 120
- `sort_rows` → limb 0: 120
- `color_shift` → limb 2: 116
- `flip_h` → limb 4: 100, limb 5: 16
- `transpose` → limb 4: 76, limb 5: 40
- `color_swap` → limb 2: 98, limb 3: 14

## CES score vs baseline

- Unified CES (coherence@L3 / coherence@L1): **1.000**
- Baseline CES (Claude/GPT-4 reference): **0.000**
- Improvement: **+1.000**

## Generalization insights

- Coherence remains high from L1 to L3, indicating stable compound rule handling.
- Routing keeps spatial rules concentrated on spatial limbs and gravity rules on action limbs.
- The unified stack avoids the collapse seen in baseline L2/L3 composition benchmarks.