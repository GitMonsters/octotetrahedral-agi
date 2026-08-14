//! Scoring utilities for the evaluation harness.

use serde::{Deserialize, Serialize};

use crate::inference::InferenceResponse;

use super::generator::EvalTask;

/// Per-task evaluation result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalResult {
    pub task_id: String,
    pub passed: bool,
    pub coherence: f32,
    pub latency_ms: f64,
    pub action_channel: usize,
    pub error: Option<String>,
}

/// Aggregate evaluation summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalSummary {
    pub total: usize,
    pub passed: usize,
    pub failed: usize,
    pub pass_rate: f64,
    pub avg_coherence: f64,
    pub avg_latency_ms: f64,
}

impl EvalSummary {
    pub fn from_results(results: &[EvalResult]) -> Self {
        let total = results.len();
        let passed = results.iter().filter(|r| r.passed).count();
        let avg_coherence =
            results.iter().map(|r| r.coherence as f64).sum::<f64>() / total.max(1) as f64;
        let avg_latency_ms =
            results.iter().map(|r| r.latency_ms).sum::<f64>() / total.max(1) as f64;
        Self {
            total,
            passed,
            failed: total - passed,
            pass_rate: passed as f64 / total.max(1) as f64,
            avg_coherence,
            avg_latency_ms,
        }
    }
}

/// Score a single task result.
pub fn score_result(task: &EvalTask, resp: &InferenceResponse) -> EvalResult {
    EvalResult {
        task_id: task.id.clone(),
        passed: resp.error.is_none(),
        coherence: resp.coherence,
        latency_ms: resp.latency_ms,
        action_channel: resp.action_channel,
        error: resp.error.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::generator::generate_tasks;
    use crate::inference::{InferenceService, InferenceRequest};
    use crate::model::OctoModelConfig;

    #[test]
    fn eval_summary_pass_rate() {
        let service = InferenceService::new(OctoModelConfig::default(), 1);
        let tasks = generate_tasks(10, 0);
        let results: Vec<EvalResult> = tasks
            .iter()
            .map(|t| {
                let req = InferenceRequest::new(t.limb_states.clone(), t.task_signal.clone());
                score_result(t, &service.infer(req))
            })
            .collect();
        let summary = EvalSummary::from_results(&results);
        assert_eq!(summary.total, 10);
        assert!(summary.pass_rate >= 0.0 && summary.pass_rate <= 1.0);
        assert_eq!(summary.passed + summary.failed, summary.total);
    }

    #[test]
    fn eval_summary_all_tasks_pass() {
        let service = InferenceService::new(OctoModelConfig::default(), 1);
        let tasks = generate_tasks(5, 42);
        let results: Vec<EvalResult> = tasks
            .iter()
            .map(|t| {
                let req = InferenceRequest::new(t.limb_states.clone(), t.task_signal.clone());
                score_result(t, &service.infer(req))
            })
            .collect();
        let summary = EvalSummary::from_results(&results);
        assert_eq!(summary.pass_rate, 1.0, "all tasks should pass");
    }
}
