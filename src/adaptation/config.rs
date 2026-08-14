//! Application configuration with environment-variable overrides.
//!
//! Mirrors `production_config.py`.

use serde::{Deserialize, Serialize};
use std::env;

/// Runtime environment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Environment {
    #[default]
    Dev,
    Staging,
    Prod,
}

impl Environment {
    fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "staging" => Self::Staging,
            "prod" | "production" => Self::Prod,
            _ => Self::Dev,
        }
    }
}

/// Full application configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    pub env: Environment,
    pub model_version: String,
    pub model_path: String,
    pub limb_count: usize,
    pub batch_size_min: usize,
    pub batch_size_max: usize,
    pub inference_timeout_ms: f64,
    pub max_retries: usize,
    pub pool_size: usize,
    pub min_coherence_threshold: f32,
    pub max_latency_ms: f64,
    pub server_host: String,
    pub server_port: u16,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            env: Environment::Dev,
            model_version: "1.0.0".into(),
            model_path: String::new(),
            limb_count: 8,
            batch_size_min: 1,
            batch_size_max: 100,
            inference_timeout_ms: 20.0,
            max_retries: 3,
            pool_size: 4,
            min_coherence_threshold: 0.5,
            max_latency_ms: 50.0,
            server_host: "0.0.0.0".into(),
            server_port: 8000,
        }
    }
}

impl AppConfig {
    /// Build config with environment-variable overrides.
    pub fn from_env() -> Self {
        let mut cfg = Self::default();
        if let Ok(v) = env::var("OCTOAGI_ENV") {
            cfg.env = Environment::from_str(&v);
        }
        if let Ok(v) = env::var("OCTOAGI_MODEL_VERSION") {
            cfg.model_version = v;
        }
        if let Ok(v) = env::var("OCTOAGI_MODEL_PATH") {
            cfg.model_path = v;
        }
        if let Ok(v) = env::var("OCTOAGI_LIMB_COUNT") {
            if let Ok(n) = v.parse() {
                cfg.limb_count = n;
            }
        }
        if let Ok(v) = env::var("OCTOAGI_POOL_SIZE") {
            if let Ok(n) = v.parse() {
                cfg.pool_size = n;
            }
        }
        if let Ok(v) = env::var("OCTOAGI_PORT") {
            if let Ok(n) = v.parse() {
                cfg.server_port = n;
            }
        }
        cfg
    }
}

/// Per-task adaptation parameters derived from a task signal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelAdaptation {
    pub coupling_strength: f32,
    pub phase: f32,
    pub bias: f32,
}

impl ModelAdaptation {
    pub fn for_task(signal: &str, _base: &AppConfig) -> Self {
        let (c, p, b): (f32, f32, f32) = match signal {
            "reasoning" => (0.6, 0.05, 0.01),
            "language" => (0.55, 0.1, 0.02),
            "spatial" => (0.58, 0.0, -0.01),
            "action" => (0.65, -0.05, 0.0),
            _ => (0.5, 0.0, 0.0),
        };
        Self {
            coupling_strength: c.min(1.0),
            phase: p,
            bias: b,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_has_8_limbs() {
        assert_eq!(AppConfig::default().limb_count, 8);
    }

    #[test]
    fn adaptation_for_known_task() {
        let cfg = AppConfig::default();
        let adapt = ModelAdaptation::for_task("reasoning", &cfg);
        assert!(adapt.coupling_strength > 0.5);
    }
}
