//! Inference module: request/response types and `InferenceService`.

pub mod service;
pub mod types;

pub use service::InferenceService;
pub use types::{BatchRequest, BatchResponse, InferenceRequest, InferenceResponse};
