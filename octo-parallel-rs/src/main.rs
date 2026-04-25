//! OctoTetrahedral Parallel Cognitive Limb Orchestrator
//!
//! Architecture
//! ============
//!
//!  ┌─────────────────────────────────────────────────────────┐
//!  │  Rust axum HTTP gateway  (this binary)                  │
//!  │                                                         │
//!  │  POST /infer                                            │
//!  │    1. call Python /encode       → memory_enhanced       │
//!  │    2. fan-out 13 /limb/* calls  → tokio join_all()      │
//!  │    3. Block AttnRes in Rust     → compound output       │
//!  │    4. return JSON stream                                 │
//!  └──────────────────────┬──────────────────────────────────┘
//!                         │ HTTP (localhost:8765)
//!  ┌──────────────────────▼──────────────────────────────────┐
//!  │  Python FastAPI limb server (octo_limb_server.py)       │
//!  │  /encode  /limb/memory  /limb/spatial  …(13 limbs)      │
//!  └─────────────────────────────────────────────────────────┘
//!
//! The Block AttnRes step is implemented natively in Rust (block_attn_res.rs)
//! using the exported pseudo-query weights (kimi_weights.json), so the
//! compound-braid computation never needs to round-trip back to Python.

mod block_attn_res;
mod limb_client;

use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use base64::Engine as _;
use clap::Parser;
use serde::{Deserialize, Serialize};
use tokio::task::JoinSet;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use tracing::info;

use block_attn_res::{block_attn_res_all_streams, KimiWeights};
use limb_client::{encode, run_limb, TensorWire};

// ─────────────────────────────────────────────────────────────────────────────
// CLI
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(about = "OctoTetrahedral parallel limb orchestrator")]
struct Args {
    /// Port to listen on
    #[arg(long, default_value = "8766")]
    port: u16,

    /// Python limb server URL
    #[arg(long, default_value = "http://localhost:8765")]
    python_url: String,

    /// Path to kimi_weights.json (exported via model.export_kimi_weights())
    #[arg(long, default_value = "kimi_weights.json")]
    weights: String,
}

// ─────────────────────────────────────────────────────────────────────────────
// App state
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone)]
struct AppState {
    python_url: Arc<String>,
    kimi: Arc<Option<KimiWeights>>,
    client: Arc<reqwest::Client>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Request / response types
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct InferRequest {
    /// Token IDs for a single sequence
    input_ids: Vec<u32>,
}

#[derive(Serialize)]
struct LimbResult {
    limb: String,
    confidence: f32,
    shape: Vec<usize>,
    elapsed_ms: f32,
}

#[derive(Serialize)]
struct BraidedStream {
    limb: String,
    /// Base64 little-endian float32 braided tensor
    data: String,
    shape: Vec<usize>,
}

#[derive(Serialize)]
struct InferResponse {
    /// Per-limb metadata (confidence, shape, timing)
    limbs: Vec<LimbResult>,
    /// Braided output tensors from Rust Block AttnRes (one per braid-eligible stream)
    braided: Vec<BraidedStream>,
    /// Whether native Rust Block AttnRes was applied
    rust_braid_applied: bool,
    /// Total wall-clock time in ms
    total_ms: f32,
    /// Number of streams processed by the braid
    n_braid_blocks: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// Handlers
// ─────────────────────────────────────────────────────────────────────────────

async fn healthz() -> impl IntoResponse {
    Json(serde_json::json!({"status": "ok", "component": "rust-orchestrator"}))
}

async fn infer(
    State(state): State<AppState>,
    Json(req): Json<InferRequest>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    let t_total = Instant::now();

    // ── 1. Encode: get shared memory_enhanced from Python ─────────────────
    let encoded = encode(&state.client, &state.python_url, &req.input_ids)
        .await
        .map_err(|e| (StatusCode::BAD_GATEWAY, format!("encode failed: {e}")))?;

    let mem_enh = Arc::new(encoded);

    // ── 2. Fan-out: call all 12 post-encode limbs in parallel (tokio JoinSet) ─
    // Note: "perception" is the *encode* step — it takes token IDs, not
    // memory_enhanced.  All 12 names here are post-encode cognitive limbs.
    let limb_names = vec![
        "memory", "planning", "language", "spatial",
        "reasoning", "metacognition", "action",
        "visualization", "imagination", "empathy", "emotion", "ethics",
    ];

    let mut join_set = JoinSet::new();
    for limb_name in &limb_names {
        let client   = state.client.clone();
        let url      = state.python_url.clone();
        let wire     = mem_enh.clone();
        let name     = limb_name.to_string();
        join_set.spawn(async move {
            let t = Instant::now();
            let result = run_limb(&client, &url, &name, &wire).await;
            (name, result, t.elapsed().as_secs_f32() * 1000.0)
        });
    }

    // Collect results as they arrive
    let mut limb_outputs: Vec<(String, TensorWire)> = Vec::with_capacity(limb_names.len());
    let mut limb_results_meta: Vec<LimbResult> = Vec::with_capacity(limb_names.len());

    while let Some(res) = join_set.join_next().await {
        match res {
            Ok((name, Ok(wire), elapsed)) => {
                let lr = LimbResult {
                    confidence: wire.confidence.unwrap_or(0.5),
                    shape: wire.shape.clone(),
                    limb: name.clone(),
                    elapsed_ms: elapsed,
                };
                limb_results_meta.push(lr);
                limb_outputs.push((name, wire));
            }
            Ok((name, Err(e), _)) => {
                tracing::warn!("Limb '{}' failed: {}", name, e);
            }
            Err(e) => {
                tracing::error!("JoinSet error: {}", e);
            }
        }
    }

    // Keep ordering consistent with limb_names
    limb_outputs.sort_by_key(|(name, _)| {
        limb_names.iter().position(|n| n == name).unwrap_or(999)
    });
    limb_results_meta.sort_by_key(|lr| {
        limb_names.iter().position(|n| *n == lr.limb).unwrap_or(999)
    });

    // ── 3. Block AttnRes in Rust (if weights loaded) ─────────────────────
    let (rust_braid_applied, n_braid_blocks, braided) =
        if let Some(kimi) = state.kimi.as_ref().as_ref() {
            let streams: Vec<&TensorWire> = limb_outputs.iter().map(|(_, w)| w).collect();
            match block_attn_res_all_streams(kimi, &streams) {
                Ok((flat_vecs, shape)) => {
                    let braid_count = flat_vecs.len();
                    // Encode braided tensors as base64 LE float32
                    let braided_streams: Vec<BraidedStream> = flat_vecs
                        .into_iter()
                        .enumerate()
                        .map(|(i, v)| {
                            let bytes: Vec<u8> = v
                                .iter()
                                .flat_map(|f| f.to_le_bytes())
                                .collect();
                            BraidedStream {
                                limb: limb_outputs
                                    .get(i)
                                    .map(|(n, _)| n.clone())
                                    .unwrap_or_else(|| format!("stream_{i}")),
                                data: base64::engine::general_purpose::STANDARD.encode(&bytes),
                                shape: shape.clone(),
                            }
                        })
                        .collect();
                    (true, braid_count, braided_streams)
                }
                Err(e) => {
                    tracing::warn!("Rust Block AttnRes failed (falling back): {e}");
                    (false, 0, vec![])
                }
            }
        } else {
            info!("No kimi_weights.json loaded — Block AttnRes skipped (pure fan-out mode)");
            (false, 0, vec![])
        };

    let total_ms = t_total.elapsed().as_secs_f32() * 1000.0;
    info!(
        "Infer complete  limbs={}  braid={}  total={:.1}ms",
        limb_outputs.len(),
        rust_braid_applied,
        total_ms
    );

    Ok(Json(InferResponse {
        limbs: limb_results_meta,
        braided,
        rust_braid_applied,
        total_ms,
        n_braid_blocks,
    }))
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "octo_parallel=info,tower_http=debug".into()),
        )
        .init();

    let args = Args::parse();

    // Load kimi weights if available
    let kimi = if std::path::Path::new(&args.weights).exists() {
        match KimiWeights::load(&args.weights) {
            Ok(w) => {
                info!(
                    "Loaded kimi weights: {} streams × {}D",
                    w.n_streams, w.hidden_dim
                );
                Some(w)
            }
            Err(e) => {
                tracing::warn!("Failed to load kimi weights from '{}': {e}", args.weights);
                None
            }
        }
    } else {
        info!(
            "kimi_weights.json not found — run `POST /export_kimi_weights` on Python server first"
        );
        None
    };

    let state = AppState {
        python_url: Arc::new(args.python_url.clone()),
        kimi: Arc::new(kimi),
        client: Arc::new(
            reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(60))
                .pool_max_idle_per_host(20)
                .build()?,
        ),
    };

    let app = Router::new()
        .route("/healthz", get(healthz))
        .route("/infer",   post(infer))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let addr = format!("0.0.0.0:{}", args.port);
    info!("Rust orchestrator listening on {addr}  →  Python @ {}", args.python_url);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
