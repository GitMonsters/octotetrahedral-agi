//! Metrics collection, health checks, and diagnostics.
//!
//! Mirrors the Python `monitoring.py` module.
//!
//! Note: [`InferenceMonitor`] uses a `Mutex`-protected ring buffer for
//! simplicity and correctness under concurrent access.  For very
//! high-throughput use cases the inner lock can be replaced with an
//! atomic/lock-free structure without changing the public API.

pub mod health;
pub mod metrics;

pub use health::{run_health_check, HealthStatus};
pub use metrics::InferenceMonitor;
