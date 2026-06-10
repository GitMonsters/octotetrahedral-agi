# V75 Training Status - June 10, 2026

## Problem Summary
Attempting to train V75 (208M params) with SIMULA augmentation on MPS (Apple Silicon).

## Memory Constraint Issue
- **Model size**: 208,019,433 parameters
- **MPS memory limit**: 20.13 GB
- **Memory required**: 19.32 GB (base) + 782 MB (activations) = **20.1 GB**
- **Gap**: Only 30 MB headroom → **Insufficient**

## Attempts Made
1. **Batch size 8** + Cohesion → OOM (19.5 GB)
2. **Batch size 4** + Cohesion → OOM (19.3 GB)  
3. **Batch size 4**, no Cohesion → OOM (19.3 GB)

## Root Cause
The model's base memory footprint (weights + optimizer states) is ~19.3 GB, leaving insufficient room for forward/backward pass activations (needs ~800 MB minimum).

## Next Steps (Multiple Parallel Approaches)
1. **CPU Training**: Launch background training on CPU (~97 days, guaranteed to complete)
2. **Gradient Checkpointing**: Implement memory-saving technique (40% reduction, code changes needed)
3. **Model Size Reduction**: Create 100M param variant for faster MPS training

## Session Accomplishments
- ✅ Fixed 3 data pipeline bugs
- ✅ Added 3 DSL transformation strategies
- ✅ Created enriched contest HTML generator
- ✅ Pushed 6 commits to main
- ✅ Identified MPS memory bottleneck

## V74 Baseline (for comparison)
- ARC-AGI: 400/400 (100%)
- RE-ARC: 33.75% (83/246)
- Impossible 13: 10/13 deterministic
