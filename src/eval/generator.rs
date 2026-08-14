//! Deterministic task generator for the evaluation harness.
//!
//! All randomness is derived from the supplied seed so parallel calls with
//! different seeds are always independent.

use serde::{Deserialize, Serialize};

/// Cognitive task families exercised by the harness.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskFamily {
    /// Symmetric uniform activation — tests hub synchronisation.
    Uniform,
    /// Alternating high/low activation — tests spatial processing.
    Alternating,
    /// Single-limb spike — tests action-channel selection.
    Spike,
    /// Linearly increasing activation — tests reasoning gradient.
    Linear,
    /// Random (seeded) activation — tests general robustness.
    Random,
}

impl TaskFamily {
    pub const ALL: [TaskFamily; 5] = [
        TaskFamily::Uniform,
        TaskFamily::Alternating,
        TaskFamily::Spike,
        TaskFamily::Linear,
        TaskFamily::Random,
    ];

    pub fn task_signal(self) -> Option<&'static str> {
        match self {
            TaskFamily::Uniform => None,
            TaskFamily::Alternating => Some("spatial"),
            TaskFamily::Spike => Some("action"),
            TaskFamily::Linear => Some("reasoning"),
            TaskFamily::Random => Some("compound"),
        }
    }
}

/// A single evaluation task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalTask {
    pub id: String,
    pub family: TaskFamily,
    pub limb_states: Vec<f32>,
    pub task_signal: Option<String>,
}

/// Generate `n` evaluation tasks with the given seed.
pub fn generate_tasks(n: usize, seed: u64) -> Vec<EvalTask> {
    let families = TaskFamily::ALL;
    let mut tasks = Vec::with_capacity(n);

    for i in 0..n {
        let family = families[i % families.len()];
        let limb_states = generate_states(family, i, seed);
        tasks.push(EvalTask {
            id: format!("task_{i:04}"),
            family,
            limb_states,
            task_signal: family.task_signal().map(str::to_string),
        });
    }

    tasks
}

fn generate_states(family: TaskFamily, index: usize, seed: u64) -> Vec<f32> {
    const N: usize = 8;
    match family {
        TaskFamily::Uniform => vec![0.5; N],
        TaskFamily::Alternating => (0..N).map(|i| if i % 2 == 0 { 0.9 } else { 0.1 }).collect(),
        TaskFamily::Spike => {
            let mut v = vec![0.1f32; N];
            v[index % N] = 1.0;
            v
        }
        TaskFamily::Linear => (0..N).map(|i| i as f32 / (N - 1) as f32).collect(),
        TaskFamily::Random => {
            // LCG pseudo-random
            let mut state = seed.wrapping_add(index as u64).wrapping_mul(6364136223846793005);
            (0..N)
                .map(|_| {
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    (state >> 33) as f32 / u32::MAX as f32
                })
                .collect()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_tasks_correct_count() {
        let tasks = generate_tasks(20, 42);
        assert_eq!(tasks.len(), 20);
    }

    #[test]
    fn generate_tasks_deterministic() {
        let a = generate_tasks(10, 42);
        let b = generate_tasks(10, 42);
        for (ta, tb) in a.iter().zip(b.iter()) {
            assert_eq!(ta.id, tb.id);
            for (sa, sb) in ta.limb_states.iter().zip(tb.limb_states.iter()) {
                assert!((sa - sb).abs() < 1e-7);
            }
        }
    }
}
