#!/usr/bin/env python3
"""
pytest test suite for the NgvtBraidEngine PyO3 interface.

All tests use synthetic data — no live openpilot processes, no messaging,
no vehicle.  Run from an openpilot checkout after building the Rust crate:

  maturin develop --manifest-path selfdrive/controls/lib/ngvt_braid/Cargo.toml
  pytest selfdrive/test/test_ngvt_braid.py -v
"""

import pytest
import numpy as np

try:
  from ngvt_braid import NgvtBraidEngine
except ImportError:
  pytest.skip("ngvt_braid Rust extension not built — run `maturin develop`", allow_module_level=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def engine() -> NgvtBraidEngine:
  return NgvtBraidEngine(major_radius=10.0, minor_radius=3.0, boost_factor=3.0)


# ---------------------------------------------------------------------------
# Return type contract
# ---------------------------------------------------------------------------

class TestInterfaceTypes:
  def test_returns_list_of_three_floats(self, engine):
    coords, score = engine.process_node(200.0, 150.0, 0.5)
    assert isinstance(coords, list)
    assert len(coords) == 3
    assert all(isinstance(v, float) for v in coords)
    assert isinstance(score, float)

  def test_score_in_unit_interval(self, engine):
    for prob in (0.0, 0.25, 0.5, 0.99, 1.0):
      _, score = engine.process_node(100.0, 100.0, prob)
      assert 0.0 <= score <= 1.0, f"score {score} out of [0,1] for prob={prob}"


# ---------------------------------------------------------------------------
# Torus projection geometry
# ---------------------------------------------------------------------------

class TestTorusProjection:
  """Verify that torus mapping always produces finite, bounded coordinates."""

  CORNERS = [
    (0.0,    0.0,    (0.0, 1164.0), (0.0, 874.0)),
    (1164.0, 874.0,  (0.0, 1164.0), (0.0, 874.0)),
    (1164.0, 0.0,    (0.0, 1164.0), (0.0, 874.0)),
    (0.0,    874.0,  (0.0, 1164.0), (0.0, 874.0)),
    (582.0,  437.0,  (0.0, 1164.0), (0.0, 874.0)),  # centre
  ]

  @pytest.mark.parametrize("x,y,xb,yb", CORNERS)
  def test_all_finite(self, engine, x, y, xb, yb):
    coords, _ = engine.process_node(x, y, 0.5, xb, yb)
    for i, v in enumerate(coords):
      assert np.isfinite(v), f"coord[{i}]={v} is not finite at ({x},{y})"

  def test_centre_maps_to_torus_surface(self, engine):
    """Centre pixel (θ=π, φ=π) should land on torus surface."""
    coords, _ = engine.process_node(582.0, 437.0, 0.5)
    R, r = 10.0, 3.0
    dist_to_ring = np.sqrt(coords[0]**2 + coords[1]**2)
    dist_to_surface = abs(np.sqrt((dist_to_ring - R)**2 + coords[2]**2) - r)
    assert dist_to_surface < 0.05, f"Point not on torus surface: {coords}"


# ---------------------------------------------------------------------------
# Braid amplification logic
# ---------------------------------------------------------------------------

class TestBraidAmplification:
  def test_boost_applied_when_in_failure_zone(self, engine):
    coords, _ = engine.process_node(100.0, 100.0, 0.2)
    engine.register_verification_results([coords])
    _, score = engine.process_node(100.0, 100.0, 0.2)
    assert abs(score - 0.6) < 1e-5, f"Expected 0.6 after 3× boost, got {score}"

  def test_score_capped_at_one(self, engine):
    coords, _ = engine.process_node(100.0, 100.0, 0.9)
    engine.register_verification_results([coords])
    _, score = engine.process_node(100.0, 100.0, 0.9)
    assert score == pytest.approx(1.0, abs=1e-6)

  def test_no_boost_outside_zone_radius(self, engine):
    """A far-away failure zone should not boost a different node."""
    engine.register_verification_results([[999.0, 999.0, 999.0]])
    _, score = engine.process_node(100.0, 100.0, 0.3)
    assert score == pytest.approx(0.3, abs=1e-5)

  def test_multiple_failure_zones_only_nearby_boost(self, engine):
    near_coords, _ = engine.process_node(100.0, 100.0, 0.2)
    engine.register_verification_results([near_coords, [999.0, 999.0, 999.0]])
    _, score = engine.process_node(100.0, 100.0, 0.2)
    assert abs(score - 0.6) < 1e-5


# ---------------------------------------------------------------------------
# Failure-zone cache lifecycle
# ---------------------------------------------------------------------------

class TestFailureZoneCache:
  def test_register_stores_zones(self, engine):
    zone = [5.5, -2.1, 0.3]
    engine.register_verification_results([zone])
    active = engine.get_active_failure_zones()
    assert len(active) == 1
    np.testing.assert_allclose(active[0], zone, atol=1e-6)

  def test_register_clears_previous_frame(self, engine):
    engine.register_verification_results([[1.0, 2.0, 3.0]])
    engine.register_verification_results([[4.0, 5.0, 6.0]])
    active = engine.get_active_failure_zones()
    assert len(active) == 1
    np.testing.assert_allclose(active[0], [4.0, 5.0, 6.0], atol=1e-6)

  def test_empty_registration_clears_cache(self, engine):
    engine.register_verification_results([[1.0, 2.0, 3.0]])
    engine.register_verification_results([])
    assert engine.get_active_failure_zones() == []

  def test_malformed_nodes_ignored(self, engine):
    """Nodes with wrong length should be silently dropped."""
    engine.register_verification_results([[1.0, 2.0], [3.0, 4.0, 5.0]])
    active = engine.get_active_failure_zones()
    assert len(active) == 1
    np.testing.assert_allclose(active[0], [3.0, 4.0, 5.0], atol=1e-6)

  def test_many_zones(self, engine):
    zones = [[float(i), float(i), float(i)] for i in range(50)]
    engine.register_verification_results(zones)
    assert len(engine.get_active_failure_zones()) == 50


# ---------------------------------------------------------------------------
# Simulated frame sequence (mirrors what ngvt_analysis.py does)
# ---------------------------------------------------------------------------

class TestFrameSequence:
  """Simulate multiple consecutive frames as the offline analyzer would."""

  def test_score_evolves_across_frames(self, engine):
    raw_prob = 0.25
    scores = []
    for _ in range(5):
      coords, score = engine.process_node(100.0, 100.0, raw_prob)
      scores.append(score)
      # flag as unstable every frame
      engine.register_verification_results([coords])

    # First frame: no prior zone, so score == raw_prob
    assert scores[0] == pytest.approx(raw_prob, abs=1e-5)
    # Subsequent frames: boosted (capped at 1.0)
    for s in scores[1:]:
      assert s == pytest.approx(min(raw_prob * 3.0, 1.0), abs=1e-5)

  def test_stable_node_never_boosted(self, engine):
    for _ in range(10):
      _, score = engine.process_node(999.0, 999.0, 0.5)
      engine.register_verification_results([])   # never flag anything
      assert score == pytest.approx(0.5, abs=1e-5)
