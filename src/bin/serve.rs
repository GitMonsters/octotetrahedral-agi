//! `octo-serve` — start the HTTP inference server.

use std::sync::Arc;

use clap::Parser;
use tokio::net::TcpListener;

use octotetrahedral_agi::adaptation::AppConfig;
use octotetrahedral_agi::inference::InferenceService;
use octotetrahedral_agi::model::OctoModelConfig;
use octotetrahedral_agi::serve::{build_router, AppState};

#[derive(Parser, Debug)]
#[command(about = "OctoTetrahedral AGI HTTP inference server")]
struct Args {
    /// Host to listen on
    #[arg(long, default_value = "0.0.0.0")]
    host: String,

    /// Port to listen on
    #[arg(long, default_value_t = 8000)]
    port: u16,
}

#[tokio::main]
async fn main() {
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
    let state = AppState { service: Arc::new(service) };
    let app = build_router(state);

    let addr = format!("{}:{}", args.host, args.port);
    tracing::info!("listening on {addr}");

    let listener = TcpListener::bind(&addr)
        .await
        .unwrap_or_else(|e| panic!("failed to bind {addr}: {e}"));

    axum::serve(listener, app).await.expect("server error");
}
