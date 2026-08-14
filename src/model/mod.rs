//! Core model types, forward pass, limb orchestration and geometry primitives.

pub mod attention;
pub mod geometry;
pub mod limb;
pub mod orchestrator;

pub use orchestrator::{OctoModel, OctoModelConfig};

// ─────────────────────────────────────────────────────────────────────────────
// Re-exports used across the crate
// ─────────────────────────────────────────────────────────────────────────────

pub use limb::{Limb, LimbKind};
pub use orchestrator::ForwardResult;
