# Migration Guide: Modular Stack to Unified Cognitive Stack

## What changed

The new unified stack consolidates forward execution into `UnifiedForwardModel` while keeping a legacy adapter for existing integrations.

## New primary API

```python
from unified.forward_model import UnifiedForwardModel

model = UnifiedForwardModel()
result = model.forward([0.1] * 8, task_signal="reasoning")
```

Returned dictionary keys:
- `limb_states`
- `shared_component`
- `residuals`
- `coherence`
- `coupling_strength`
- `phase`
- `bias`

## Legacy compatibility path

```python
from unified.forward_model import LegacyForwardAdapter

adapter = LegacyForwardAdapter()
limb_states = adapter.run([0.1] * 8, task_type="reasoning")
```

## Recommended rollout

1. Deploy `LegacyForwardAdapter` first to keep existing call signatures stable.
2. Move downstream consumers to `UnifiedForwardModel.forward` result fields.
3. Enable benchmark checks using `python benchmarks/unified_perf.py`.
4. Remove modular wrappers after migration is complete.
