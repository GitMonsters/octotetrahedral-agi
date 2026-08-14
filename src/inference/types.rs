//! Request and response types for the inference API.
//!
//! These types are the Rust equivalents of the Python `api_types` module.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ─────────────────────────────────────────────────────────────────────────────
// Request
// ─────────────────────────────────────────────────────────────────────────────

/// A single inference request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceRequest {
    /// Unique request identifier.  Auto-generated if not supplied.
    pub request_id: String,
    /// Per-limb activation levels (length must equal the model's `limb_count`).
    pub limb_states: Vec<f32>,
    /// Optional task hint used by the adaptation layer.
    pub task_signal: Option<String>,
}

impl InferenceRequest {
    /// Build a request with an auto-generated UUID.
    pub fn new(limb_states: Vec<f32>, task_signal: Option<String>) -> Self {
        Self {
            request_id: Uuid::new_v4().to_string(),
            limb_states,
            task_signal,
        }
    }
}

/// A batch of inference requests.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchRequest {
    pub batch_id: String,
    pub requests: Vec<InferenceRequest>,
}

impl BatchRequest {
    pub fn new(requests: Vec<InferenceRequest>) -> Self {
        Self {
            batch_id: Uuid::new_v4().to_string(),
            requests,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Response
// ─────────────────────────────────────────────────────────────────────────────

/// Response to a single inference request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResponse {
    pub request_id: String,
    pub limb_states: Vec<f32>,
    pub shared_component: f32,
    pub residuals: Vec<f32>,
    pub coherence: f32,
    pub coupling_strength: f32,
    pub phase: f32,
    pub bias: f32,
    pub action_channel: usize,
    /// Wall-clock inference latency in milliseconds.
    pub latency_ms: f64,
    /// Non-null only on error.
    pub error: Option<String>,
}

impl InferenceResponse {
    /// Build an error response.
    pub fn error(request_id: String, error: String, latency_ms: f64, limb_count: usize) -> Self {
        Self {
            request_id,
            limb_states: vec![0.0; limb_count],
            shared_component: 0.0,
            residuals: vec![0.0; limb_count],
            coherence: 0.0,
            coupling_strength: 0.0,
            phase: 0.0,
            bias: 0.0,
            action_channel: 0,
            latency_ms,
            error: Some(error),
        }
    }
}

/// Response to a batch of inference requests.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchResponse {
    pub batch_id: String,
    pub responses: Vec<InferenceResponse>,
    pub total_latency_ms: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_roundtrip_json() {
        let req = InferenceRequest::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], Some("reasoning".into()));
        let json = serde_json::to_string(&req).unwrap();
        let decoded: InferenceRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.limb_states.len(), 8);
        assert_eq!(decoded.task_signal.as_deref(), Some("reasoning"));
    }

    #[test]
    fn response_roundtrip_json() {
        let resp = InferenceResponse {
            request_id: "test-id".into(),
            limb_states: vec![0.5; 8],
            shared_component: 0.5,
            residuals: vec![0.0; 8],
            coherence: 0.9,
            coupling_strength: 0.5,
            phase: 0.0,
            bias: 0.0,
            action_channel: 3,
            latency_ms: 1.23,
            error: None,
        };
        let json = serde_json::to_string(&resp).unwrap();
        let decoded: InferenceResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.action_channel, 3);
        assert!(decoded.error.is_none());
    }

    #[test]
    fn error_response_has_error_field() {
        let resp = InferenceResponse::error("id".into(), "something failed".into(), 0.5, 8);
        assert!(resp.error.is_some());
        assert_eq!(resp.limb_states.len(), 8);
    }
}
