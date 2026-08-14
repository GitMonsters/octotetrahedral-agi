//! Production inference service with a fixed-size model pool.
//!
//! Mirrors the Python `inference_service.py` module.

use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;

use anyhow::Result;

use crate::model::{OctoModel, OctoModelConfig};
use crate::monitoring::InferenceMonitor;

use super::types::{BatchRequest, BatchResponse, InferenceRequest, InferenceResponse};

// ─────────────────────────────────────────────────────────────────────────────
// Model pool
// ─────────────────────────────────────────────────────────────────────────────

/// A fixed-size pool of [`OctoModel`] instances.
///
/// Models are checked out individually: the mutex is released before inference
/// runs, so multiple threads can hold different models concurrently.
struct ModelPool {
    /// Parked models waiting to be checked out.
    available: Mutex<Vec<OctoModel>>,
    /// Notified whenever a model is returned to the pool.
    returned: Condvar,
}

impl ModelPool {
    fn new(size: usize, config: OctoModelConfig) -> Self {
        let models = (0..size).map(|_| OctoModel::new(config.clone())).collect();
        Self {
            available: Mutex::new(models),
            returned: Condvar::new(),
        }
    }

    /// Check out a model, blocking until one is available.
    fn checkout(&self) -> OctoModel {
        let mut guard = self.available.lock().expect("model pool mutex poisoned");
        loop {
            if let Some(model) = guard.pop() {
                return model;
            }
            guard = self.returned.wait(guard).expect("condvar wait failed");
        }
    }

    /// Return a model to the pool and wake a waiting thread.
    fn checkin(&self, model: OctoModel) {
        let mut guard = self.available.lock().expect("model pool mutex poisoned");
        guard.push(model);
        self.returned.notify_one();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// InferenceService
// ─────────────────────────────────────────────────────────────────────────────

/// Thread-safe inference service backed by a fixed-size model pool.
#[derive(Clone)]
pub struct InferenceService {
    pool: Arc<ModelPool>,
    monitor: Arc<InferenceMonitor>,
    limb_count: usize,
}

impl InferenceService {
    /// Create a new service with a pool of `pool_size` model instances.
    pub fn new(config: OctoModelConfig, pool_size: usize) -> Self {
        let limb_count = config.limb_count;
        Self {
            pool: Arc::new(ModelPool::new(pool_size.max(1), config)),
            monitor: Arc::new(InferenceMonitor::new()),
            limb_count,
        }
    }

    /// Run a single inference request.
    ///
    /// A model is checked out from the pool (blocking if none are free),
    /// inference is run with the pool lock released, then the model is
    /// returned before this function returns.
    pub fn infer(&self, request: InferenceRequest) -> InferenceResponse {
        let start = Instant::now();

        // Check out a model — mutex released for the duration of inference.
        let model = self.pool.checkout();
        let result = model.forward(&request.limb_states, request.task_signal.as_deref());
        self.pool.checkin(model);

        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

        match result {
            Ok(fwd) => {
                self.monitor.record(latency_ms, false);
                InferenceResponse {
                    request_id: request.request_id,
                    limb_states: fwd.limb_states,
                    shared_component: fwd.shared_component,
                    residuals: fwd.residuals,
                    coherence: fwd.coherence,
                    coupling_strength: fwd.coupling_strength,
                    phase: fwd.phase,
                    bias: fwd.bias,
                    action_channel: fwd.action_channel,
                    latency_ms,
                    error: None,
                }
            }
            Err(e) => {
                self.monitor.record(latency_ms, true);
                InferenceResponse::error(request.request_id, e, latency_ms, self.limb_count)
            }
        }
    }

    /// Run a batch of inference requests sequentially.
    pub fn infer_batch(&self, batch: BatchRequest) -> Result<BatchResponse> {
        let start = Instant::now();
        let responses: Vec<InferenceResponse> =
            batch.requests.into_iter().map(|r| self.infer(r)).collect();
        let total_latency_ms = start.elapsed().as_secs_f64() * 1000.0;
        Ok(BatchResponse {
            batch_id: batch.batch_id,
            responses,
            total_latency_ms,
        })
    }

    /// Expose monitoring metrics.
    pub fn monitor(&self) -> &InferenceMonitor {
        &self.monitor
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::types::InferenceRequest;

    fn make_service() -> InferenceService {
        InferenceService::new(OctoModelConfig::default(), 1)
    }

    #[test]
    fn infer_returns_valid_response() {
        let svc = make_service();
        let req = InferenceRequest::new(vec![0.5f32; 8], None);
        let resp = svc.infer(req);
        assert!(resp.error.is_none());
        assert_eq!(resp.limb_states.len(), 8);
    }

    #[test]
    fn infer_bad_input_returns_error() {
        let svc = make_service();
        let req = InferenceRequest::new(vec![0.5f32; 5], None);
        let resp = svc.infer(req);
        assert!(resp.error.is_some());
    }

    #[test]
    fn batch_infer_works() {
        let svc = make_service();
        let requests: Vec<InferenceRequest> = (0..3)
            .map(|_| InferenceRequest::new(vec![0.5f32; 8], Some("reasoning".into())))
            .collect();
        let batch = BatchRequest::new(requests);
        let resp = svc.infer_batch(batch).unwrap();
        assert_eq!(resp.responses.len(), 3);
    }

    #[test]
    fn pool_checkin_allows_reuse() {
        // pool_size = 1 means the second request blocks until the first
        // model is returned; with sequential calls this should always succeed.
        let svc = InferenceService::new(OctoModelConfig::default(), 1);
        for _ in 0..5 {
            let req = InferenceRequest::new(vec![0.5f32; 8], None);
            let resp = svc.infer(req);
            assert!(resp.error.is_none());
        }
    }
}
