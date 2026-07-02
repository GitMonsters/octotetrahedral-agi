# Unified Cognitive Stack Architecture

This repository now includes a unified quantum-biological stack built around three core modules:

- `unified/forward_model.py`
- `unified/feedback_loop.py`
- `unified/state_transitions.py`

## Highlights

- **Quantum Coupling**: all 8 limbs are synchronized by shared coherence operations.
- **RNA Adaptation**: task-conditioned regulatory controls tune coupling and gate parameters.
- **Bidirectional Integration**: forward and feedback streams are fused in one coherent pass.
- **Scalable Design**: tensor decomposition exposes a shared component and per-limb residuals.
- **Action Channel Selection**: the dominant output limb is identified via `select_action_channel`.

## Layering

1. RNA adaptation produces coupling + gate parameters.
2. Quantum operators perform phase/bias transforms and coherence synchronization.
3. Feedback loop re-couples all limbs.
4. State transitions apply quantum evolution then biological adaptation.
5. Cognitive integration returns the unified limb state.

## Backward Compatibility

`LegacyForwardAdapter` in `unified/forward_model.py` preserves a legacy `run(...)` API for callers that still expect modular execution.
