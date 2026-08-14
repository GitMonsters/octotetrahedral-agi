//! Health checks and diagnostics for the cognitive stack.

use serde::{Deserialize, Serialize};

use crate::inference::types::InferenceRequest;
use crate::inference::InferenceService;

/// Result of a health-check run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    pub healthy: bool,
    pub model_loaded: bool,
    pub coherence_baseline: f32,
    pub limb_symmetry_ok: bool,
    pub self_test_passed: bool,
    pub self_test_details: Vec<SelfTestCase>,
    pub diagnostics: Diagnostics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfTestCase {
    pub task_signal: Option<String>,
    pub passed: bool,
    pub coherence: f32,
    pub latency_ms: f64,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Diagnostics {
    pub total_requests: u64,
    pub error_count: u64,
    pub avg_latency_ms: f64,
}

static SELF_TEST_INPUTS: &[(&[f32], &str)] = &[
    (&[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], "reasoning"),
    (&[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], "language"),
    (&[0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1], "spatial"),
    (&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], "action"),
    (&[0.3, 0.5, 0.7, 0.2, 0.8, 0.4, 0.6, 0.1], "compound"),
];

/// Run a set of self-tests against `service` and return a [`HealthStatus`].
pub fn run_health_check(service: &InferenceService, num_tests: usize) -> HealthStatus {
    let num_tests = num_tests.clamp(1, SELF_TEST_INPUTS.len());
    let mut details = Vec::with_capacity(num_tests);
    let mut all_passed = true;
    let mut coherence_sum = 0.0f32;

    for &(states, task) in SELF_TEST_INPUTS.iter().take(num_tests) {
        let req = InferenceRequest::new(states.to_vec(), Some(task.to_string()));
        let resp = service.infer(req);
        let passed = resp.error.is_none();
        if !passed {
            all_passed = false;
        }
        coherence_sum += resp.coherence;
        details.push(SelfTestCase {
            task_signal: Some(task.to_string()),
            passed,
            coherence: resp.coherence,
            latency_ms: resp.latency_ms,
            error: resp.error,
        });
    }

    let coherence_baseline = coherence_sum / num_tests as f32;

    // Limb symmetry check: run uniform input and check variance is small
    let sym_req = InferenceRequest::new(vec![0.5f32; 8], None);
    let sym_resp = service.infer(sym_req);
    let mean = sym_resp.limb_states.iter().sum::<f32>() / 8.0;
    let variance = sym_resp.limb_states.iter().map(|&s| (s - mean).powi(2)).sum::<f32>() / 8.0;
    let limb_symmetry_ok = variance < 0.05;

    let snap = service.monitor().snapshot();
    let diagnostics = Diagnostics {
        total_requests: snap.total_requests,
        error_count: snap.error_count,
        avg_latency_ms: snap.avg_latency_ms,
    };

    HealthStatus {
        healthy: all_passed && limb_symmetry_ok,
        model_loaded: true,
        coherence_baseline,
        limb_symmetry_ok,
        self_test_passed: all_passed,
        self_test_details: details,
        diagnostics,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::InferenceService;
    use crate::model::OctoModelConfig;

    #[test]
    fn health_check_passes() {
        let svc = InferenceService::new(OctoModelConfig::default(), 1);
        let status = run_health_check(&svc, 3);
        assert!(status.model_loaded);
        assert!(status.self_test_passed);
        assert!(status.healthy);
    }
}
