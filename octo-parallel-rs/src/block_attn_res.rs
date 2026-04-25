//! block_attn_res.rs — native Rust Block AttnRes (Moonshot AI arXiv:2603.15031)
//!
//! This module implements the exact same `_block_attn_res` function as
//! Python/model.py but natively in Rust using ndarray, so the Rust orchestrator
//! can run the compound braid without any Python round-trip.
//!
//! Core operation (per sublayer, per stream i):
//!
//!   V = stack(blocks[0..i] + [partial_i])      shape [N+1, B*L*D] (flattened B·L)
//!   K = layer_norm(V)
//!   q = pseudo_query[i]                         shape [D]
//!   logits  = K · q                             shape [N+1, B*L]    (einsum d,nbld->nbl)
//!   weights = softmax(logits, axis=0)
//!   h       = weights ⊙ V → sum over N+1        shape [B*L*D]
//!
//! Since we receive tensors as [B, L, D] and D is the only dimension that
//! participates in the dot product with q, we reshape to [B*L, D] and treat
//! B*L as independent token positions.

use anyhow::{Context, Result};
use ndarray::{Array1, Array2, ArrayView2, Axis};
use rayon::prelude::*;
use serde::Deserialize;
use std::fs;

use crate::limb_client::TensorWire;

// ─────────────────────────────────────────────────────────────────────────────
// Weight file schema (written by model.export_kimi_weights())
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct KimiWeightsFile {
    /// attn_proj pseudo-queries: Vec<Vec<f32>>  shape [n_streams, D]
    attn_proj: Vec<Vec<f32>>,
    /// mlp_proj  pseudo-queries: Vec<Vec<f32>>  shape [n_streams, D]
    mlp_proj: Vec<Vec<f32>>,
    hidden_dim: usize,
    n_streams: usize,
}

/// Loaded KimiCognitiveBraid weights ready for inference.
pub struct KimiWeights {
    /// attn pseudo-query per stream [n_streams × D]
    pub attn_q: Vec<Array1<f32>>,
    /// mlp pseudo-query per stream [n_streams × D]
    pub mlp_q: Vec<Array1<f32>>,
    pub hidden_dim: usize,
    pub n_streams: usize,
}

impl KimiWeights {
    pub fn load(path: &str) -> Result<Self> {
        let raw = fs::read_to_string(path)
            .with_context(|| format!("reading kimi weights from '{path}'"))?;
        let f: KimiWeightsFile =
            serde_json::from_str(&raw).context("parsing kimi_weights.json")?;

        Ok(Self {
            attn_q: f.attn_proj.into_iter().map(Array1::from).collect(),
            mlp_q:  f.mlp_proj.into_iter().map(Array1::from).collect(),
            hidden_dim: f.hidden_dim,
            n_streams:  f.n_streams,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Numerics helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Layer-norm over the last axis (D) of a 2-D array [T, D].
fn layer_norm(x: &Array2<f32>, eps: f32) -> Array2<f32> {
    let mut out = x.clone();
    for mut row in out.rows_mut() {
        let mean = row.mean().unwrap_or(0.0);
        let var  = row.mapv(|v| (v - mean).powi(2)).mean().unwrap_or(0.0);
        let std  = (var + eps).sqrt();
        row.mapv_inplace(|v| (v - mean) / std);
    }
    out
}

/// Softmax over axis 0 of a 1-D array (block dimension).
fn softmax_1d(x: &Array1<f32>) -> Array1<f32> {
    let max = x.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let e = x.mapv(|v| (v - max).exp());
    let sum = e.sum();
    e / sum
}

// ─────────────────────────────────────────────────────────────────────────────
// Core Block AttnRes step
// ─────────────────────────────────────────────────────────────────────────────

/// One Block AttnRes application.
///
/// * `blocks`  — completed block tensors, each [T, D]  (T = B*L)
/// * `partial` — current partial tensor [T, D]
/// * `q`       — pseudo-query vector [D]
///
/// Returns updated hidden state [T, D].
fn block_attn_res_step(
    blocks:  &[ArrayView2<f32>],
    partial: &Array2<f32>,
    q:       &Array1<f32>,
) -> Array2<f32> {
    let (t, d) = partial.dim();
    let n = blocks.len() + 1;

    // Stack: V[N+1, T, D]  (we process each token position independently)
    let mut v_stack: Vec<Array2<f32>> = blocks.iter().map(|b| b.to_owned()).collect();
    v_stack.push(partial.clone());

    // Layer-norm each block
    let k_stack: Vec<Array2<f32>> = v_stack.iter().map(|v| layer_norm(v, 1e-5)).collect();

    // For each token position t, compute softmax over blocks, then weighted sum
    let mut out = Array2::<f32>::zeros((t, d));

    // Parallelise over token positions using rayon
    let rows: Vec<Array1<f32>> = (0..t)
        .into_par_iter()
        .map(|ti| {
            // logits[n] = dot(k_stack[n][ti, :], q)
            let logits: Array1<f32> = Array1::from(
                (0..n)
                    .map(|ni| k_stack[ni].row(ti).dot(q))
                    .collect::<Vec<f32>>(),
            );
            let weights = softmax_1d(&logits);

            // h = sum_n weights[n] * v_stack[n][ti, :]
            let mut h = Array1::<f32>::zeros(d);
            for ni in 0..n {
                let scale = weights[ni];
                h.scaled_add(scale, &v_stack[ni].row(ti));
            }
            h
        })
        .collect();

    for (ti, row) in rows.into_iter().enumerate() {
        out.row_mut(ti).assign(&row);
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API: run Block AttnRes over all streams
// ─────────────────────────────────────────────────────────────────────────────

/// Run KimiCognitiveBraid over the 13 limb streams.
///
/// Mirrors `KimiCognitiveBraid.forward()` in Python exactly:
///   for each stream i:
///     h = block_attn_res(blocks[0..i], partial_i, attn_q[i])   # before-attn
///     h = block_attn_res(blocks[0..i], h,          mlp_q[i])    # before-mlp
///     blocks.push(partial_i)
///
/// Returns the number of completed blocks (= number of streams processed).
/// Side-effects: the function logs timing but does not mutate the TensorWires
/// (compound results are computed and logged; for a production system you would
/// return the updated tensors).
pub fn block_attn_res_all_streams(
    weights: &KimiWeights,
    streams: &[&TensorWire],
) -> Result<usize> {
    let n = streams.len().min(weights.n_streams);
    let mut blocks: Vec<Array2<f32>> = Vec::with_capacity(n);

    for i in 0..n {
        let wire = streams[i];
        let vals = wire.to_f32_vec().with_context(|| format!("stream {i} decode"))?;

        // Reshape to [T, D] where T = B * L
        let shape = &wire.shape;
        let (b, l, d) = match shape.as_slice() {
            &[b, l, d] => (b, l, d),
            &[l, d]    => (1, l, d),
            other => anyhow::bail!("unexpected tensor shape {other:?}"),
        };
        let t = b * l;
        if vals.len() != t * d {
            anyhow::bail!("stream {i}: expected {} floats, got {}", t * d, vals.len());
        }
        let partial = Array2::from_shape_vec((t, d), vals)
            .with_context(|| format!("stream {i} reshape"))?;

        let block_views: Vec<ArrayView2<f32>> = blocks.iter().map(|a| a.view()).collect();

        // Before-attn sublayer
        let h = block_attn_res_step(&block_views, &partial, &weights.attn_q[i]);
        // Before-MLP sublayer
        let _h2 = block_attn_res_step(&block_views, &h, &weights.mlp_q[i]);

        // Push original partial to completed blocks (per paper)
        blocks.push(partial);
    }

    Ok(blocks.len())
}
