//! Configuration, model adaptation and checkpoint serialisation.

pub mod checkpoint;
pub mod config;

pub use checkpoint::{load_checkpoint, save_checkpoint, Checkpoint};
pub use config::{AppConfig, ModelAdaptation};
