//! Axum HTTP server and API routes.

pub mod handlers;
pub mod router;

pub use router::build_router;
pub use router::AppState;
