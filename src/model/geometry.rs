//! Tetrahedral geometry primitives.
//!
//! The tetrahedral coordinate system underpins the OctoTetrahedral
//! architecture: 4 primary vertices, 6 edge midpoints, 4 face centres and
//! various sub-divisions give 64 canonical geometry points that drive the
//! attention mask structure.

use serde::{Deserialize, Serialize};

/// Number of canonical geometry points in the tetrahedral structure.
pub const NUM_GEOMETRY_POINTS: usize = 64;

/// Single 3-D point in the tetrahedral coordinate system.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TetrahedralPoint {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl TetrahedralPoint {
    /// Euclidean distance between two tetrahedral points.
    pub fn distance(&self, other: &TetrahedralPoint) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

/// Complete tetrahedral geometry: 64 canonical points and the adjacency
/// matrix that drives the geometry-aware attention mask.
#[derive(Debug, Clone)]
pub struct TetrahedralGeometry {
    pub points: Vec<TetrahedralPoint>,
    /// `adjacency[i][j]` is true when points i and j share an edge in the
    /// tetrahedral mesh.
    pub adjacency: Vec<Vec<bool>>,
}

impl TetrahedralGeometry {
    /// Build the default 64-point tetrahedral geometry.
    pub fn new() -> Self {
        let points = Self::build_points();
        let n = points.len();
        let adjacency = Self::build_adjacency(&points, n);
        Self { points, adjacency }
    }

    fn build_points() -> Vec<TetrahedralPoint> {
        // 4 primary vertices of a regular tetrahedron
        let vertices: &[(f32, f32, f32)] = &[
            (1.0, 1.0, 1.0),
            (1.0, -1.0, -1.0),
            (-1.0, 1.0, -1.0),
            (-1.0, -1.0, 1.0),
        ];

        let mut pts: Vec<TetrahedralPoint> = Vec::with_capacity(NUM_GEOMETRY_POINTS);

        // Primary vertices
        for &(x, y, z) in vertices {
            pts.push(TetrahedralPoint { x, y, z });
        }

        // Edge midpoints (6 edges)
        for i in 0..4 {
            for j in (i + 1)..4 {
                let a = vertices[i];
                let b = vertices[j];
                pts.push(TetrahedralPoint {
                    x: (a.0 + b.0) / 2.0,
                    y: (a.1 + b.1) / 2.0,
                    z: (a.2 + b.2) / 2.0,
                });
            }
        }

        // Face centres (4 faces)
        let faces: &[[usize; 3]] = &[[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]];
        for face in faces {
            let v: Vec<_> = face.iter().map(|&i| vertices[i]).collect();
            pts.push(TetrahedralPoint {
                x: v.iter().map(|p| p.0).sum::<f32>() / 3.0,
                y: v.iter().map(|p| p.1).sum::<f32>() / 3.0,
                z: v.iter().map(|p| p.2).sum::<f32>() / 3.0,
            });
        }

        // Pad to 64 with fractional sub-division points
        while pts.len() < NUM_GEOMETRY_POINTS {
            let idx = pts.len();
            let base = &pts[idx % (vertices.len() + 6 + 4)];
            let angle = (idx as f32) * std::f32::consts::TAU / NUM_GEOMETRY_POINTS as f32;
            pts.push(TetrahedralPoint {
                x: base.x * angle.cos() * 0.5,
                y: base.y * angle.sin() * 0.5,
                z: base.z * (1.0 - angle / std::f32::consts::TAU),
            });
        }

        pts
    }

    fn build_adjacency(points: &[TetrahedralPoint], n: usize) -> Vec<Vec<bool>> {
        // Two points are adjacent if their distance is at most 1.5× the
        // minimum inter-point distance (heuristic for the default geometry).
        let min_dist = {
            let mut m = f32::MAX;
            for i in 0..n {
                for j in (i + 1)..n {
                    let d = points[i].distance(&points[j]);
                    if d < m {
                        m = d;
                    }
                }
            }
            m
        };
        let threshold = min_dist * 1.5;

        let mut adj = vec![vec![false; n]; n];
        for i in 0..n {
            for j in (i + 1)..n {
                if points[i].distance(&points[j]) <= threshold {
                    adj[i][j] = true;
                    adj[j][i] = true;
                }
            }
        }
        adj
    }
}

impl Default for TetrahedralGeometry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometry_has_64_points() {
        let g = TetrahedralGeometry::new();
        assert_eq!(g.points.len(), NUM_GEOMETRY_POINTS);
    }

    #[test]
    fn adjacency_is_symmetric() {
        let g = TetrahedralGeometry::new();
        let n = g.points.len();
        for i in 0..n {
            for j in 0..n {
                assert_eq!(g.adjacency[i][j], g.adjacency[j][i]);
            }
        }
    }
}
