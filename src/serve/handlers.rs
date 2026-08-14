//! Axum route handlers.

use axum::{
    extract::State,
    http::StatusCode,
    Json,
};
use serde_json::{json, Value};

use crate::inference::types::{BatchRequest, InferenceRequest};
use crate::monitoring::run_health_check;

use super::router::AppState;

// ─────────────────────────────────────────────────────────────────────────────
// GET /health
// ─────────────────────────────────────────────────────────────────────────────

pub async fn health(State(state): State<AppState>) -> (StatusCode, Json<Value>) {
    let status = run_health_check(&state.service, 3);
    let code = if status.healthy { StatusCode::OK } else { StatusCode::SERVICE_UNAVAILABLE };
    (code, Json(json!(status)))
}

// ─────────────────────────────────────────────────────────────────────────────
// POST /infer
// ─────────────────────────────────────────────────────────────────────────────

pub async fn infer(
    State(state): State<AppState>,
    Json(req): Json<InferenceRequest>,
) -> (StatusCode, Json<Value>) {
    let resp = state.service.infer(req);
    let code = if resp.error.is_none() { StatusCode::OK } else { StatusCode::UNPROCESSABLE_ENTITY };
    (code, Json(json!(resp)))
}

// ─────────────────────────────────────────────────────────────────────────────
// POST /batch
// ─────────────────────────────────────────────────────────────────────────────

pub async fn infer_batch(
    State(state): State<AppState>,
    Json(batch): Json<BatchRequest>,
) -> (StatusCode, Json<Value>) {
    match state.service.infer_batch(batch) {
        Ok(resp) => (StatusCode::OK, Json(json!(resp))),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        ),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GET /metrics
// ─────────────────────────────────────────────────────────────────────────────

pub async fn metrics(State(state): State<AppState>) -> Json<Value> {
    let snap = state.service.monitor().snapshot();
    Json(json!(snap))
}
