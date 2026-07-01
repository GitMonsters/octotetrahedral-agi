"""RNA adaptation entrypoints used by the unified forward model."""

from __future__ import annotations

from rna.regulatory import RNARegulatoryNetwork
from rna.splicing import splice_gate_parameters


def adapt_for_task(regulatory_network: RNARegulatoryNetwork, task_signal: str | None) -> dict[str, float]:
    """Produce task-conditioned coupling and gate parameters."""
    normalized = (task_signal or "default").strip().lower()
    regulatory_network.update_for_task(normalized)
    coupling = regulatory_network.coupling_strength(normalized)

    gain = 1.0 + coupling * 0.5
    phase, bias = splice_gate_parameters(phase=0.35, bias=0.05, adaptation_gain=gain)
    return {
        "coupling_strength": coupling,
        "phase": phase,
        "bias": bias,
    }
