//! limb_client.rs — async reqwest client for the Python limb server

use anyhow::{Context, Result};
use base64::{engine::general_purpose::STANDARD as B64, Engine};
use serde::{Deserialize, Serialize};

// ─────────────────────────────────────────────────────────────────────────────
// Tensor wire types (matches Python server JSON schema)
// ─────────────────────────────────────────────────────────────────────────────

/// A flat float32 tensor transmitted as base64-encoded little-endian bytes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorWire {
    /// base64-encoded raw float32 bytes (little-endian)
    pub data: String,
    /// tensor shape, e.g. [1, 16, 512]
    pub shape: Vec<usize>,
    /// optional confidence scalar
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
}

impl TensorWire {
    /// Decode base64 → Vec<f32>.  Returns Err if bytes are not a multiple of 4.
    pub fn to_f32_vec(&self) -> Result<Vec<f32>> {
        let bytes = B64.decode(&self.data).context("base64 decode")?;
        if bytes.len() % 4 != 0 {
            anyhow::bail!("tensor bytes not aligned to f32 ({})", bytes.len());
        }
        Ok(bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect())
    }

    /// Total number of elements (product of shape dims).
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// /encode
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct EncodeRequest<'a> {
    input_ids: &'a [u32],
}

pub async fn encode(
    client: &reqwest::Client,
    base_url: &str,
    input_ids: &[u32],
) -> Result<TensorWire> {
    let url = format!("{base_url}/encode");
    let resp = client
        .post(&url)
        .json(&EncodeRequest { input_ids })
        .send()
        .await
        .context("POST /encode")?
        .error_for_status()
        .context("/encode non-2xx")?
        .json::<serde_json::Value>()
        .await
        .context("/encode deserialise")?;

    Ok(TensorWire {
        data:       resp["data"].as_str().unwrap_or("").to_string(),
        shape:      resp["shape"]
            .as_array()
            .unwrap_or(&vec![])
            .iter()
            .filter_map(|v| v.as_u64().map(|u| u as usize))
            .collect(),
        confidence: None,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// /limb/{name}
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct LimbRequest<'a> {
    data:  &'a str,
    shape: &'a [usize],
}

pub async fn run_limb(
    client: &reqwest::Client,
    base_url: &str,
    limb_name: &str,
    wire: &TensorWire,
) -> Result<TensorWire> {
    let url = format!("{base_url}/limb/{limb_name}");
    let resp = client
        .post(&url)
        .json(&LimbRequest { data: &wire.data, shape: &wire.shape })
        .send()
        .await
        .with_context(|| format!("POST /limb/{limb_name}"))?
        .error_for_status()
        .with_context(|| format!("/limb/{limb_name} non-2xx"))?
        .json::<serde_json::Value>()
        .await
        .with_context(|| format!("/limb/{limb_name} deserialise"))?;

    let confidence = resp["confidence"].as_f64().map(|v| v as f32);

    Ok(TensorWire {
        data:  resp["data"].as_str().unwrap_or("").to_string(),
        shape: resp["shape"]
            .as_array()
            .unwrap_or(&vec![])
            .iter()
            .filter_map(|v| v.as_u64().map(|u| u as usize))
            .collect(),
        confidence,
    })
}
