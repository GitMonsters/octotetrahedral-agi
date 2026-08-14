//! Checkpoint serialisation for the OctoModel.

use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::model::OctoModelConfig;

/// Serialisable model checkpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    /// Crate version at save time.
    pub version: String,
    /// Model configuration.
    pub config: OctoModelConfig,
    /// Per-limb initial state values stored at checkpoint time.
    pub limb_states: Vec<f32>,
    /// Optional freeform metadata (e.g. training step, loss).
    pub metadata: serde_json::Value,
}

impl Checkpoint {
    pub fn new(config: OctoModelConfig, limb_states: Vec<f32>) -> Self {
        Self {
            version: env!("CARGO_PKG_VERSION").to_string(),
            config,
            limb_states,
            metadata: serde_json::Value::Null,
        }
    }

    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = metadata;
        self
    }
}

/// Persist a checkpoint to `path` as pretty-printed JSON.
pub fn save_checkpoint(checkpoint: &Checkpoint, path: &Path) -> Result<()> {
    let json = serde_json::to_string_pretty(checkpoint)
        .context("failed to serialise checkpoint")?;
    fs::write(path, json).with_context(|| format!("failed to write checkpoint to {}", path.display()))?;
    Ok(())
}

/// Load a checkpoint from `path`.
pub fn load_checkpoint(path: &Path) -> Result<Checkpoint> {
    let json = fs::read_to_string(path)
        .with_context(|| format!("failed to read checkpoint from {}", path.display()))?;
    let ck: Checkpoint =
        serde_json::from_str(&json).context("failed to deserialise checkpoint")?;
    Ok(ck)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::OctoModelConfig;
    use std::path::PathBuf;

    #[test]
    fn checkpoint_roundtrip() {
        let dir = tempfile::tempdir().unwrap_or_else(|_| {
            // Fallback if tempfile is not available: use /tmp
            panic!("tempdir failed");
        });
        let path: PathBuf = dir.path().join("ck.json");
        let ck = Checkpoint::new(OctoModelConfig::default(), vec![0.5f32; 8]);
        save_checkpoint(&ck, &path).unwrap();
        let loaded = load_checkpoint(&path).unwrap();
        assert_eq!(loaded.config.limb_count, 8);
        assert_eq!(loaded.limb_states.len(), 8);
    }
}
