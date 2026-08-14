//! OctoTetrahedral AGI — Rust-native core library for performance-sensitive
//! subsystems.
//!
//! # Design philosophy — incremental Rustification
//!
//! **Python remains the canonical layer** for solver experimentation, ARC
//! reasoning, training loops, and rapid prototyping.  This crate owns only
//! the hot-path primitives where Rust delivers clear benefit:
//!
//! - deterministic forward pass & hub synchronisation (`model`)
//! - high-throughput inference service with a lock-safe model pool (`inference`)
//! - Axum HTTP server so Python calls Rust over a stable JSON API (`serve`)
//! - `Mutex`-protected latency metrics with a rolling ring buffer (`monitoring`)
//! - seeded, reproducible eval harness (`eval`)
//! - checkpoint / config serialisation (`adaptation`)
//!
//! # Integration with Python
//!
//! **Option A — HTTP** (zero-dependency, current default):
//!
//! ```python
//! import requests
//! resp = requests.post("http://localhost:8000/infer", json={
//!     "request_id": "abc",
//!     "limb_states": [0.5] * 8,
//!     "task_signal": "reasoning",
//! })
//! ```
//!
//! **Option B — PyO3 FFI** (future, lower latency): add the
//! `python-bindings` feature and expose a `#[pymodule]` wrapping
//! [`inference::InferenceService`].
//!
//! See [`RUST_MIGRATION.md`](../RUST_MIGRATION.md) for the full incremental
//! migration strategy, layer responsibilities, and decision rules.
//!
//! # Crate layout
//!
//! | Module | Responsibility |
//! |--------|---------------|
//! | [`model`] | `OctoModel`, forward pass, hub-sync, geometry/attention primitives |
//! | [`inference`] | Inference service, request/response types |
//! | [`serve`] | Axum HTTP server and API routes |
//! | [`monitoring`] | Metrics collection, health checks, diagnostics |
//! | [`eval`] | Deterministic evaluation harness |
//! | [`adaptation`] | Configuration, model adaptation, checkpoint serialisation |

pub mod adaptation;
pub mod eval;
pub mod inference;
pub mod model;
pub mod monitoring;
pub mod serve;
