//! Cognitive limb abstraction.
//!
//! Each limb is a semi-autonomous processing unit responsible for a
//! distinct cognitive domain.  The 8 primary limbs mirror the Python
//! implementation:
//!
//! | Index | Kind          |
//! |-------|---------------|
//! | 0     | Perception    |
//! | 1     | Memory        |
//! | 2     | Planning      |
//! | 3     | Language      |
//! | 4     | Spatial       |
//! | 5     | Reasoning     |
//! | 6     | Metacognition |
//! | 7     | Action        |

use serde::{Deserialize, Serialize};

/// Named cognitive domains mirroring the Python limbs package.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LimbKind {
    Perception,
    Memory,
    Planning,
    Language,
    Spatial,
    Reasoning,
    Metacognition,
    Action,
}

impl LimbKind {
    /// Canonical ordered set of 8 primary limbs.
    pub const ALL: [LimbKind; 8] = [
        LimbKind::Perception,
        LimbKind::Memory,
        LimbKind::Planning,
        LimbKind::Language,
        LimbKind::Spatial,
        LimbKind::Reasoning,
        LimbKind::Metacognition,
        LimbKind::Action,
    ];

    /// Human-readable name.
    pub fn name(self) -> &'static str {
        match self {
            LimbKind::Perception => "perception",
            LimbKind::Memory => "memory",
            LimbKind::Planning => "planning",
            LimbKind::Language => "language",
            LimbKind::Spatial => "spatial",
            LimbKind::Reasoning => "reasoning",
            LimbKind::Metacognition => "metacognition",
            LimbKind::Action => "action",
        }
    }
}

/// A single cognitive limb with its current activation state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Limb {
    pub kind: LimbKind,
    /// Current activation level in [0, 1].
    pub state: f32,
}

impl Limb {
    pub fn new(kind: LimbKind, state: f32) -> Self {
        Self { kind, state: state.clamp(0.0, 1.0) }
    }

    /// Apply a task-signal modulation: reasoning and language tasks boost
    /// their respective limbs; all others are clamped.
    pub fn modulate(&self, task_signal: &str, coupling: f32) -> f32 {
        let boost = match (self.kind, task_signal) {
            (LimbKind::Reasoning, "reasoning") => 0.2,
            (LimbKind::Language, "language") => 0.2,
            (LimbKind::Spatial, "spatial") => 0.2,
            (LimbKind::Action, "action") => 0.2,
            _ => 0.0,
        };
        (self.state + boost * coupling).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn limb_kind_count() {
        assert_eq!(LimbKind::ALL.len(), 8);
    }

    #[test]
    fn limb_modulate_clamped() {
        let limb = Limb::new(LimbKind::Reasoning, 0.95);
        let v = limb.modulate("reasoning", 1.0);
        assert!(v <= 1.0);
    }
}
