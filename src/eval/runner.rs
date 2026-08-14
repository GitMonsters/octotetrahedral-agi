//! Evaluation task runner.

use crate::inference::{InferenceRequest, InferenceResponse, InferenceService};

use super::generator::EvalTask;
use super::scorer::{score_result, EvalResult};

/// Run `tasks` through `service` and return per-task results.
pub fn run_eval(service: &InferenceService, tasks: &[EvalTask]) -> Vec<EvalResult> {
    tasks
        .iter()
        .map(|task| {
            let req = InferenceRequest::new(
                task.limb_states.clone(),
                task.task_signal.clone(),
            );
            let resp: InferenceResponse = service.infer(req);
            score_result(task, &resp)
        })
        .collect()
}
