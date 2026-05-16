use pyo3::prelude::*;
use std::f32::consts::PI;

/// Projects flat image-space coordinates onto the surface of a torus manifold
/// using the standard parametric formula, then applies Compounding Braid
/// attention amplification near previously registered failure zones.
///
/// This is a pure math library with no openpilot runtime dependencies.
/// It is designed for offline log analysis via tools/ngvt_analysis.py.
#[pyclass]
pub struct NgvtBraidEngine {
    major_radius: f32,
    minor_radius: f32,
    boost_factor: f32,
    failure_zones_manifold: Vec<[f32; 3]>,
}

#[pymethods]
impl NgvtBraidEngine {
    #[new]
    #[pyo3(signature = (major_radius=10.0, minor_radius=3.0, boost_factor=3.0))]
    fn new(major_radius: f32, minor_radius: f32, boost_factor: f32) -> Self {
        Self {
            major_radius,
            minor_radius,
            boost_factor,
            failure_zones_manifold: Vec::new(),
        }
    }

    /// Projects (x, y) from image space to 3D torus coordinates, then
    /// applies the Braid attention boost if the node lands near a cached
    /// failure zone from the previous analysis frame.
    ///
    /// Returns: (torus_coords [X, Y, Z], adjusted_score clamped to [0, 1])
    #[pyo3(signature = (x, y, raw_prob, x_bounds=(0.0, 1164.0), y_bounds=(0.0, 874.0)))]
    fn process_node(
        &mut self,
        x: f32,
        y: f32,
        raw_prob: f32,
        x_bounds: (f32, f32),
        y_bounds: (f32, f32),
    ) -> (Vec<f32>, f32) {
        let theta = ((x - x_bounds.0) / (x_bounds.1 - x_bounds.0)) * 2.0 * PI;
        let phi   = ((y - y_bounds.0) / (y_bounds.1 - y_bounds.0)) * 2.0 * PI;

        let m_x = (self.major_radius + self.minor_radius * phi.cos()) * theta.cos();
        let m_y = (self.major_radius + self.minor_radius * phi.cos()) * theta.sin();
        let m_z = self.minor_radius * phi.sin();
        let current_coords = [m_x, m_y, m_z];

        let mut final_score = raw_prob;
        for zone in &self.failure_zones_manifold {
            let dist = ((current_coords[0] - zone[0]).powi(2)
                + (current_coords[1] - zone[1]).powi(2)
                + (current_coords[2] - zone[2]).powi(2))
            .sqrt();
            if dist < 1.5 {
                final_score *= self.boost_factor;
            }
        }

        (current_coords.to_vec(), final_score.min(1.0))
    }

    /// Registers unstable nodes from the current frame so the next frame
    /// can apply the Braid boost near those manifold positions.
    fn register_verification_results(&mut self, unstable_nodes: Vec<Vec<f32>>) {
        self.failure_zones_manifold.clear();
        for node in unstable_nodes {
            if node.len() == 3 {
                self.failure_zones_manifold.push([node[0], node[1], node[2]]);
            }
        }
    }

    fn get_active_failure_zones(&self) -> Vec<Vec<f32>> {
        self.failure_zones_manifold.iter().map(|z| z.to_vec()).collect()
    }
}

#[pymodule]
fn ngvt_braid(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<NgvtBraidEngine>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_torus_projection_always_finite() {
        let mut engine = NgvtBraidEngine::new(10.0, 3.0, 3.0);
        for &(x, y) in &[(0.0f32, 0.0f32), (582.0, 437.0), (1164.0, 874.0), (1164.0, 0.0)] {
            let (coords, score) = engine.process_node(x, y, 0.5, (0.0, 1164.0), (0.0, 874.0));
            assert_eq!(coords.len(), 3);
            for &v in &coords {
                assert!(v.is_finite(), "Manifold coord exploded at ({x},{y}): {v}");
            }
            assert!(score >= 0.0 && score <= 1.0);
        }
    }

    #[test]
    fn test_score_clamped_at_one() {
        let mut engine = NgvtBraidEngine::new(10.0, 3.0, 3.0);
        let (coords, _) = engine.process_node(100.0, 100.0, 1.0, (0.0, 1164.0), (0.0, 874.0));
        engine.register_verification_results(vec![coords]);
        let (_, score) = engine.process_node(100.0, 100.0, 1.0, (0.0, 1164.0), (0.0, 874.0));
        assert_abs_diff_eq!(score, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_braid_boost_applied() {
        let mut engine = NgvtBraidEngine::new(10.0, 3.0, 3.0);
        let (coords, _) = engine.process_node(100.0, 100.0, 0.2, (0.0, 1164.0), (0.0, 874.0));
        engine.register_verification_results(vec![coords]);
        let (_, score) = engine.process_node(100.0, 100.0, 0.2, (0.0, 1164.0), (0.0, 874.0));
        assert_abs_diff_eq!(score, 0.6, epsilon = 1e-5);
    }

    #[test]
    fn test_failure_zones_cleared_each_frame() {
        let mut engine = NgvtBraidEngine::new(10.0, 3.0, 3.0);
        engine.register_verification_results(vec![vec![1.0, 2.0, 3.0]]);
        assert_eq!(engine.get_active_failure_zones().len(), 1);
        engine.register_verification_results(vec![]);
        assert_eq!(engine.get_active_failure_zones().len(), 0);
    }
}
