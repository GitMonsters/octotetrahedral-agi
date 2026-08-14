//! `octo-infer` — run a single forward pass and print the result.

use clap::Parser;
use octotetrahedral_agi::adaptation::AppConfig;
use octotetrahedral_agi::inference::{InferenceRequest, InferenceService};
use octotetrahedral_agi::model::OctoModelConfig;

#[derive(Parser, Debug)]
#[command(about = "OctoTetrahedral AGI single inference")]
struct Args {
    /// Comma-separated limb activation values (default: 0.5 × 8)
    #[arg(long, default_value = "0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5")]
    limb_states: String,

    /// Optional task signal (reasoning | language | spatial | action | compound)
    #[arg(long)]
    task_signal: Option<String>,
}

fn main() {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    let limb_states: Vec<f32> = args
        .limb_states
        .split(',')
        .map(|s| s.trim().parse::<f32>().expect("limb state must be a float"))
        .collect();

    let cfg = AppConfig::from_env();
    let model_cfg = OctoModelConfig {
        limb_count: cfg.limb_count,
        coupling_strength: 0.5,
        phase: 0.0,
        bias: 0.0,
    };
    let service = InferenceService::new(model_cfg, 1);
    let req = InferenceRequest::new(limb_states, args.task_signal);
    let resp = service.infer(req);
    println!("{}", serde_json::to_string_pretty(&resp).unwrap());
    if resp.error.is_some() {
        std::process::exit(1);
    }
}
