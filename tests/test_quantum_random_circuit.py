"""Random circuit sampling tests for the quantum operator stack.

These tests use deterministic random sampling (`random.Random(seed=0)`) to
exercise many gate/coupling combinations while keeping runs reproducible and
fast.
"""

from __future__ import annotations

import math
import random

import pytest

from quantum.coherence import coherence_score, synchronize
from quantum.gates import apply_bias_gate, apply_phase_gate
from quantum.operators import apply_unified_quantum_operator, tensor_decompose


def test_random_circuit_coherence_bounds() -> None:
    rng = random.Random(0)

    for _ in range(200):
        n_limbs = rng.randint(2, 16)
        limb_states = [rng.uniform(-1.0, 1.0) for _ in range(n_limbs)]
        phase = rng.uniform(0.0, 2.0 * math.pi)
        bias = rng.uniform(-1.0, 1.0)
        coupling_strength = rng.uniform(0.0, 1.0)

        output, coherence = apply_unified_quantum_operator(
            limb_states,
            phase=phase,
            bias=bias,
            coupling_strength=coupling_strength,
        )

        assert 0.0 <= coherence <= 1.0
        assert len(output) == len(limb_states)


def test_random_circuit_full_coupling_maximizes_coherence() -> None:
    rng = random.Random(0)

    for _ in range(50):
        n_limbs = rng.randint(2, 16)
        limb_states = [rng.uniform(-1.0, 1.0) for _ in range(n_limbs)]
        phase = rng.uniform(0.0, 2.0 * math.pi)
        bias = rng.uniform(-1.0, 1.0)

        output, coherence = apply_unified_quantum_operator(
            limb_states,
            phase=phase,
            bias=bias,
            coupling_strength=1.0,
        )

        gated = [apply_bias_gate(apply_phase_gate(value, phase), bias) for value in limb_states]
        expected_mean = sum(gated) / len(gated)
        variance = sum((value - expected_mean) ** 2 for value in output) / len(output)

        assert output == pytest.approx([expected_mean] * len(output))
        assert variance == pytest.approx(0.0)
        assert coherence == pytest.approx(1.0)


def test_random_circuit_zero_coupling_preserves_shape() -> None:
    rng = random.Random(0)

    for _ in range(50):
        n_limbs = rng.randint(2, 16)
        limb_states = [rng.uniform(-1.0, 1.0) for _ in range(n_limbs)]
        phase = rng.uniform(0.0, 2.0 * math.pi)
        bias = rng.uniform(-1.0, 1.0)

        output, _ = apply_unified_quantum_operator(
            limb_states,
            phase=phase,
            bias=bias,
            coupling_strength=0.0,
        )

        gated = [apply_bias_gate(apply_phase_gate(value, phase), bias) for value in limb_states]

        assert len(output) == len(limb_states)
        assert output == pytest.approx(gated)
        assert synchronize(gated, 0.0) == pytest.approx(gated)


def test_tensor_decompose_roundtrip() -> None:
    rng = random.Random(0)

    for _ in range(100):
        n_limbs = rng.randint(1, 32)
        limb_states = [rng.uniform(-10.0, 10.0) for _ in range(n_limbs)]

        shared, residuals = tensor_decompose(limb_states)

        assert len(residuals) == len(limb_states)
        for index, value in enumerate(limb_states):
            assert shared + residuals[index] == pytest.approx(value)


def test_phase_gate_zero_phase_is_identity() -> None:
    rng = random.Random(0)

    for _ in range(100):
        value = rng.uniform(-100.0, 100.0)
        assert apply_phase_gate(value, 0.0) == pytest.approx(value)


def test_bias_gate_zero_bias_is_identity() -> None:
    rng = random.Random(0)

    for _ in range(100):
        value = rng.uniform(-100.0, 100.0)
        assert apply_bias_gate(value, 0.0) == pytest.approx(value)


def test_random_circuit_monotone_coupling() -> None:
    rng = random.Random(0)

    coupling_steps = [step / 10.0 for step in range(11)]

    for _ in range(20):
        n_limbs = rng.randint(2, 16)
        limb_states = [rng.uniform(-1.0, 1.0) for _ in range(n_limbs)]
        phase = rng.uniform(0.0, 2.0 * math.pi)
        bias = rng.uniform(-1.0, 1.0)

        coherences = [
            apply_unified_quantum_operator(
                limb_states,
                phase=phase,
                bias=bias,
                coupling_strength=coupling_strength,
            )[1]
            for coupling_strength in coupling_steps
        ]

        for previous, current in zip(coherences, coherences[1:]):
            assert current + 1e-12 >= previous


def test_coherence_score_uniform_input_is_one() -> None:
    rng = random.Random(0)

    for _ in range(100):
        value = rng.uniform(-100.0, 100.0)
        n_limbs = rng.randint(1, 32)
        assert coherence_score([value] * n_limbs) == pytest.approx(1.0)
