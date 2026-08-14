//! `octo-health` — run health checks and print diagnostics.

use clap::Parser;
use octotetrahedral_agi::adaptation::AppConfig;
use octotetrahedral_agi::inference::InferenceService;
use octotetrahedral_agi::model::OctoModelConfig;
use octotetrahedral_agi::monitoring::run_health_check;

#[derive(Parser, Debug)]
#[command(about = "OctoTetrahedral AGI health check")]
struct Args {
    /// Number of self-test cases to run (1–5)
    #[arg(long, default_value_t = 5)]
    num_tests: usize,
}

fn main() {
    tracing_subscriber::fmt::init();
    let args = Args::parse();
    let cfg = AppConfig::from_env();
    let model_cfg = OctoModelConfig {
        limb_count: cfg.limb_count,
        coupling_strength: 0.5,
        phase: 0.0,
        bias: 0.0,
    };
    let service = InferenceService::new(model_cfg, cfg.pool_size);
    let status = run_health_check(&service, args.num_tests);
    println!("{}", serde_json::to_string_pretty(&status).unwrap());
    if !status.healthy {
        std::process::exit(1);
    }
}
