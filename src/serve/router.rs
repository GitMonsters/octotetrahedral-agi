//! Axum router and shared application state.

use std::sync::Arc;

use axum::{
    http::{HeaderValue, Method},
    routing::{get, post},
    Router,
};
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

use crate::inference::InferenceService;

use super::handlers;

/// Shared state injected into all route handlers.
#[derive(Clone)]
pub struct AppState {
    pub service: Arc<InferenceService>,
}

/// Build the Axum router with all routes attached.
///
/// CORS is intentionally restrictive: only `localhost` origins are allowed by
/// default so that arbitrary remote sites cannot call the inference API on
/// behalf of browser users.  Override `OCTOAGI_CORS_ORIGIN` at runtime to add
/// additional trusted origins (e.g. `http://localhost:3000` for a local UI).
pub fn build_router(state: AppState) -> Router {
    // Restrict CORS to localhost callers and explicitly named origins.
    let cors = build_cors();

    Router::new()
        .route("/health", get(handlers::health))
        .route("/infer", post(handlers::infer))
        .route("/batch", post(handlers::infer_batch))
        .route("/metrics", get(handlers::metrics))
        .layer(cors)
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

/// Build a CORS layer that allows only same-host callers.
///
/// The set of allowed origins is:
/// - `http://localhost:*` and `http://127.0.0.1:*` (local Python orchestrator)
/// - the value of the `OCTOAGI_CORS_ORIGIN` environment variable, if set
fn build_cors() -> CorsLayer {
    let mut layer = CorsLayer::new()
        .allow_methods([Method::GET, Method::POST])
        .allow_headers(tower_http::cors::Any);

    // Always allow localhost callers (the Python orchestration layer).
    let localhost_origins: Vec<HeaderValue> = [
        "http://localhost:8000",
        "http://localhost:8765",
        "http://localhost:8766",
        "http://127.0.0.1:8000",
        "http://127.0.0.1:8765",
        "http://127.0.0.1:8766",
    ]
    .iter()
    .filter_map(|s| s.parse().ok())
    .collect();

    // Optionally allow an extra origin configured at runtime.
    let extra: Vec<HeaderValue> = std::env::var("OCTOAGI_CORS_ORIGIN")
        .ok()
        .and_then(|s| s.parse().ok())
        .into_iter()
        .collect();

    let all_origins: Vec<HeaderValue> =
        localhost_origins.into_iter().chain(extra).collect();

    layer = layer.allow_origin(all_origins);
    layer
}
