//! Deterministic evaluation harness.

pub mod generator;
pub mod runner;
pub mod scorer;

pub use generator::{generate_tasks, EvalTask, TaskFamily};
pub use runner::run_eval;
pub use scorer::{score_result, EvalSummary};
