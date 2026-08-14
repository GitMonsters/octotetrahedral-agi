//! `octo-eval` — run the deterministic evaluation harness and print a summary.

use clap::Parser;
use octotetrahedral_agi::adaptation::AppConfig;
use octotetrahedral_agi::eval::{generate_tasks, run_eval, EvalSummary};
use octotetrahedral_agi::inference::InferenceService;
use octotetrahedral_agi::model::OctoModelConfig;

#[derive(Parser, Debug)]
#[command(about = "OctoTetrahedral AGI evaluation harness")]
struct Args {
    /// Number of evaluation tasks to generate
    #[arg(long, default_value_t = 20)]
    num_tasks: usize,

    /// Random seed for deterministic task generation
    #[arg(long, default_value_t = 42)]
    seed: u64,
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
    let tasks = generate_tasks(args.num_tasks, args.seed);
    let results = run_eval(&service, &tasks);
    let summary = EvalSummary::from_results(&results);

    println!("{}", serde_json::to_string_pretty(&summary).unwrap());

    if summary.pass_rate < 1.0 {
        std::process::exit(1);
    }
}
