//! Canonical orchestrator — the unified cognitive stack.
//!
//! `OctoModel` wires together all 8 limbs, the hub-synchronisation step,
//! and the RNA-inspired adaptation layer into a single forward pass that
//! returns structured `ForwardResult` outputs.

use serde::{Deserialize, Serialize};

use super::limb::{Limb, LimbKind};

/// Configuration for the OctoModel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OctoModelConfig {
    /// Number of cognitive limbs (default 8).
    pub limb_count: usize,
    /// Base coupling strength between limbs.
    pub coupling_strength: f32,
    /// Initial phase offset for quantum-inspired operations.
    pub phase: f32,
    /// Bias term applied during adaptation.
    pub bias: f32,
}

impl Default for OctoModelConfig {
    fn default() -> Self {
        Self {
            limb_count: 8,
            coupling_strength: 0.5,
            phase: 0.0,
            bias: 0.0,
        }
    }
}

/// Structured output of a single forward pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForwardResult {
    /// Integrated limb activation states after the forward pass.
    pub limb_states: Vec<f32>,
    /// Shared component extracted by tensor decomposition.
    pub shared_component: f32,
    /// Per-limb residuals after shared-component removal.
    pub residuals: Vec<f32>,
    /// Coherence measure ∈ [0, 1].
    pub coherence: f32,
    /// Effective coupling strength after adaptation.
    pub coupling_strength: f32,
    /// Effective phase after adaptation.
    pub phase: f32,
    /// Effective bias after adaptation.
    pub bias: f32,
    /// Index of the selected action channel (argmax of limb_states).
    pub action_channel: usize,
}

/// Main model type.  Holds configuration and per-limb state.
#[derive(Debug)]
pub struct OctoModel {
    pub config: OctoModelConfig,
    limbs: Vec<Limb>,
}

impl OctoModel {
    /// Construct a new model with default zero-state limbs.
    pub fn new(config: OctoModelConfig) -> Self {
        let limbs = LimbKind::ALL
            .iter()
            .take(config.limb_count)
            .map(|&k| Limb::new(k, 0.0))
            .collect();
        Self { config, limbs }
    }

    /// Run a single forward pass.
    ///
    /// # Arguments
    /// * `limb_states` — activation levels for each limb (length must equal
    ///   `config.limb_count`).
    /// * `task_signal` — optional task hint used by the adaptation layer.
    pub fn forward(
        &self,
        limb_states: &[f32],
        task_signal: Option<&str>,
    ) -> Result<ForwardResult, String> {
        if limb_states.len() != self.config.limb_count {
            return Err(format!(
                "expected {} limb states, got {}",
                self.config.limb_count,
                limb_states.len()
            ));
        }

        // ── 1. RNA-inspired adaptation ────────────────────────────────────
        let adaptation = adapt_for_task(task_signal, self.config.coupling_strength, self.config.phase, self.config.bias);

        // ── 2. Tensor decomposition ───────────────────────────────────────
        let (shared_component, residuals) = tensor_decompose(limb_states);

        // ── 3. Quantum-inspired operator ──────────────────────────────────
        let (quantum_states, coherence) = apply_quantum_operator(
            limb_states,
            adaptation.phase,
            adaptation.bias,
            adaptation.coupling_strength,
        );

        // ── 4. Limb-level modulation ──────────────────────────────────────
        let modulated: Vec<f32> = self
            .limbs
            .iter()
            .zip(&quantum_states)
            .map(|(limb, &qs)| {
                let task = task_signal.unwrap_or("");
                let m = Limb::new(limb.kind, qs).modulate(task, adaptation.coupling_strength);
                m
            })
            .collect();

        // ── 5. Hub synchronisation ────────────────────────────────────────
        let synced = hub_sync(&modulated, adaptation.coupling_strength);

        // ── 6. Bidirectional integration ──────────────────────────────────
        let integrated = bidirectional_integrate(&synced);

        // ── 7. Action channel selection ───────────────────────────────────
        let action_channel = integrated
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        Ok(ForwardResult {
            limb_states: integrated,
            shared_component,
            residuals,
            coherence,
            coupling_strength: adaptation.coupling_strength,
            phase: adaptation.phase,
            bias: adaptation.bias,
            action_channel,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers (mirror of Python unified/ and rna/ sub-packages)
// ─────────────────────────────────────────────────────────────────────────────

struct Adaptation {
    coupling_strength: f32,
    phase: f32,
    bias: f32,
}

/// RNA-inspired task adaptation: modulate coupling/phase/bias based on the
/// task signal keyword.
fn adapt_for_task(
    task_signal: Option<&str>,
    base_coupling: f32,
    base_phase: f32,
    base_bias: f32,
) -> Adaptation {
    let (coupling_delta, phase_delta, bias_delta) = match task_signal {
        Some("reasoning") => (0.1, 0.05, 0.01),
        Some("language") => (0.05, 0.1, 0.02),
        Some("spatial") => (0.08, 0.0, -0.01),
        Some("action") => (0.15, -0.05, 0.0),
        _ => (0.0, 0.0, 0.0),
    };
    Adaptation {
        coupling_strength: (base_coupling + coupling_delta).clamp(0.0, 1.0),
        phase: base_phase + phase_delta,
        bias: base_bias + bias_delta,
    }
}

/// Tensor decomposition: extract shared mean component and residuals.
fn tensor_decompose(states: &[f32]) -> (f32, Vec<f32>) {
    let shared = states.iter().sum::<f32>() / states.len() as f32;
    let residuals = states.iter().map(|&s| s - shared).collect();
    (shared, residuals)
}

/// Quantum-inspired operator: applies phase rotation and coupling.
/// Returns (evolved_states, coherence).
fn apply_quantum_operator(
    states: &[f32],
    phase: f32,
    bias: f32,
    coupling: f32,
) -> (Vec<f32>, f32) {
    let n = states.len() as f32;
    let mean = states.iter().sum::<f32>() / n;

    let evolved: Vec<f32> = states
        .iter()
        .map(|&s| {
            let coupled = s + coupling * (mean - s);
            let rotated = coupled * phase.cos() + bias * phase.sin();
            rotated.clamp(0.0, 1.0)
        })
        .collect();

    // Coherence: 1 − normalised variance
    let evolved_mean = evolved.iter().sum::<f32>() / n;
    let variance = evolved.iter().map(|&s| (s - evolved_mean).powi(2)).sum::<f32>() / n;
    let coherence = (1.0 - variance).clamp(0.0, 1.0);

    (evolved, coherence)
}

/// Hub synchronisation: pull each limb toward the global mean scaled by
/// `coupling`.
fn hub_sync(states: &[f32], coupling: f32) -> Vec<f32> {
    let mean = states.iter().sum::<f32>() / states.len() as f32;
    states.iter().map(|&s| (s + coupling * mean).clamp(0.0, 1.0)).collect()
}

/// Bidirectional integration: forward pass followed by reverse pass, averaged.
fn bidirectional_integrate(states: &[f32]) -> Vec<f32> {
    let n = states.len();
    let fwd: Vec<f32> = states
        .iter()
        .enumerate()
        .map(|(i, &s)| {
            let prev = if i > 0 { states[i - 1] } else { s };
            (s + prev) / 2.0
        })
        .collect();
    let bwd: Vec<f32> = states
        .iter()
        .enumerate()
        .map(|(i, &s)| {
            let next = if i + 1 < n { states[i + 1] } else { s };
            (s + next) / 2.0
        })
        .collect();
    fwd.iter().zip(&bwd).map(|(&f, &b)| (f + b) / 2.0).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_model() -> OctoModel {
        OctoModel::new(OctoModelConfig::default())
    }

    #[test]
    fn forward_pass_correct_length() {
        let model = default_model();
        let states = vec![0.5f32; 8];
        let result = model.forward(&states, None).unwrap();
        assert_eq!(result.limb_states.len(), 8);
        assert_eq!(result.residuals.len(), 8);
    }

    #[test]
    fn forward_pass_wrong_length_errors() {
        let model = default_model();
        let err = model.forward(&[0.5f32; 5], None).unwrap_err();
        assert!(err.contains("expected 8"));
    }

    #[test]
    fn coherence_in_range() {
        let model = default_model();
        let result = model.forward(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], Some("reasoning")).unwrap();
        assert!(result.coherence >= 0.0 && result.coherence <= 1.0);
    }

    #[test]
    fn action_channel_valid_index() {
        let model = default_model();
        let result = model.forward(&[0.5f32; 8], Some("action")).unwrap();
        assert!(result.action_channel < 8);
    }

    #[test]
    fn tensor_decompose_residuals_sum_to_zero() {
        let states = vec![0.1, 0.3, 0.5, 0.7, 0.2, 0.4, 0.6, 0.8];
        let (_shared, residuals) = tensor_decompose(&states);
        let sum: f32 = residuals.iter().sum();
        assert!(sum.abs() < 1e-5, "residuals should sum to ~0, got {sum}");
    }
}
