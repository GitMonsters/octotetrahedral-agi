//! Geometry-aware attention primitives.
//!
//! Implements a scaled dot-product attention weighted by the tetrahedral
//! geometry adjacency, so attention between non-adjacent points is masked.

use super::geometry::TetrahedralGeometry;

/// Compute geometry-masked self-attention over `query`, `key`, `value` vectors.
///
/// * `q`, `k`, `v` — each a flat slice of length `seq_len × d_model`
/// * Returns the attended output flat slice of length `seq_len × d_model`.
pub fn geometry_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    d_model: usize,
    geometry: &TetrahedralGeometry,
) -> Vec<f32> {
    assert_eq!(q.len(), seq_len * d_model);
    assert_eq!(k.len(), seq_len * d_model);
    assert_eq!(v.len(), seq_len * d_model);

    let scale = (d_model as f32).sqrt().recip();
    let n = geometry.points.len();

    let mut out = vec![0.0f32; seq_len * d_model];

    for i in 0..seq_len {
        // compute raw attention scores over all j
        let mut scores = vec![f32::NEG_INFINITY; seq_len];
        for j in 0..seq_len {
            // geometry mask: if both indices map into the geometry, only
            // allow adjacent or same-point pairs
            let gi = i % n;
            let gj = j % n;
            if i != j && !geometry.adjacency[gi][gj] {
                continue;
            }

            let dot: f32 = (0..d_model)
                .map(|d| q[i * d_model + d] * k[j * d_model + d])
                .sum();
            scores[j] = dot * scale;
        }

        // softmax
        let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut exp_sum = 0.0f32;
        let mut exp_scores: Vec<f32> = scores
            .iter()
            .map(|&s| {
                if s.is_finite() {
                    let e = (s - max_s).exp();
                    exp_sum += e;
                    e
                } else {
                    0.0
                }
            })
            .collect();
        if exp_sum > 0.0 {
            for e in &mut exp_scores {
                *e /= exp_sum;
            }
        }

        // weighted sum of values
        for j in 0..seq_len {
            for d in 0..d_model {
                out[i * d_model + d] += exp_scores[j] * v[j * d_model + d];
            }
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::geometry::TetrahedralGeometry;

    #[test]
    fn attention_output_length() {
        let geo = TetrahedralGeometry::new();
        let seq_len = 4;
        let d_model = 8;
        let q = vec![0.5f32; seq_len * d_model];
        let k = q.clone();
        let v = q.clone();
        let out = geometry_attention(&q, &k, &v, seq_len, d_model, &geo);
        assert_eq!(out.len(), seq_len * d_model);
    }
}
