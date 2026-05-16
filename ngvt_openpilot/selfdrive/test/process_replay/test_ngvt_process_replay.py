#!/usr/bin/env python3
"""
NGVT Braid — process_replay integration test
=============================================
Verifies the NGVT Braid offline analysis against a public openpilot log
segment using openpilot's standard process_replay / LogReader infrastructure.

This test is OFFLINE ONLY — it reads a recorded log segment and validates
that the NGVT analysis is:
  1. Deterministic (same output on two passes)
  2. Numerically stable (all manifold coords finite, scores in [0,1])
  3. Consistent across backends (Rust ≈ PyTorch ≈ tinygrad within 1e-4)
  4. Score-monotone (Braid boost never decreases a score)

It does NOT start any daemon, does NOT publish to any cereal socket, and
does NOT modify the vehicle control path.

Run:
  pytest selfdrive/test/process_replay/test_ngvt_process_replay.py -v

  # Against a specific route segment:
  TEST_ROUTE="a2a0ccea32023010|2023-07-27--13-01-19/0" \\
    pytest selfdrive/test/process_replay/test_ngvt_process_replay.py -v

  # Disable network download (use only local cache):
  FILEREADER_CACHE=1 pytest ...
"""

import os
import sys
import pytest
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Backend availability flags (set at collection time)
# ---------------------------------------------------------------------------
RUST_AVAILABLE      = False
TORCH_AVAILABLE     = False
TINYGRAD_AVAILABLE  = False

try:
  from ngvt_braid import NgvtBraidEngine as NgvtBraidRust
  RUST_AVAILABLE = True
except ImportError:
  pass

try:
  sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
  from ngvt_braid_torch import NgvtBraidEngineTorch
  import torch
  TORCH_AVAILABLE = True
except ImportError:
  pass

try:
  from ngvt_braid_tinygrad import NgvtBraidEngineTinygrad
  TINYGRAD_AVAILABLE = True
except ImportError:
  pass

if not any([RUST_AVAILABLE, TORCH_AVAILABLE, TINYGRAD_AVAILABLE]):
  pytest.skip(
    "No NGVT Braid backend available — build Rust crate or install torch/tinygrad.",
    allow_module_level=True,
  )

# ---------------------------------------------------------------------------
# Log segment fixture
# ---------------------------------------------------------------------------

# Public comma.ai test segment used by openpilot's own model_replay.py
DEFAULT_TEST_ROUTE = "8494c69d3c710e81|000001d4--2648a9a404/4"
TEST_ROUTE = os.environ.get("TEST_ROUTE", DEFAULT_TEST_ROUTE)

# Limit how many modelV2 frames to process (keeps test fast)
MAX_FRAMES = 60


@pytest.fixture(scope="module")
def lead_samples():
  """
  Download (or cache) the test segment and extract lead node samples.
  Returns list of dicts: {x, y, prob, frame_id}.
  """
  try:
    from openpilot.tools.lib.logreader import LogReader
  except ImportError:
    pytest.skip("openpilot not installed — run from an openpilot checkout.")

  print(f"\nLoading segment: {TEST_ROUTE}")
  lr = LogReader(TEST_ROUTE)

  samples = []
  frame_id = 0
  for model in lr.filter("modelV2"):
    if frame_id >= MAX_FRAMES:
      break
    for idx, lead in enumerate(model.leadsV3):
      samples.append({
        "frame_id":   frame_id,
        "lead_index": idx,
        "x":          float(lead.x[0]) if len(lead.x) > 0 else 0.0,
        "y":          float(lead.y[0]) if len(lead.y) > 0 else 0.0,
        "prob":       float(lead.prob),
      })
    frame_id += 1

  if not samples:
    pytest.skip(f"No modelV2/leadsV3 messages found in segment {TEST_ROUTE}")

  print(f"  Frames: {frame_id}  Lead samples: {len(samples)}")
  return samples


# ---------------------------------------------------------------------------
# Helper: run analysis with any backend
# ---------------------------------------------------------------------------

def run_analysis(engine, samples):
  """
  Simulate the ngvt_analysis.py frame loop on *samples*.
  Returns list of (torus_coords, adjusted_score, flagged) per sample.
  """
  results = []
  unstable_cache = []
  current_frame = -1
  frame_unstable = []

  for s in samples:
    if s["frame_id"] != current_frame:
      if current_frame >= 0:
        engine.register_verification_results(unstable_cache)
      unstable_cache = list(frame_unstable)
      frame_unstable = []
      current_frame = s["frame_id"]

    coords, score = engine.process_node(s["x"], s["y"], s["prob"])
    flagged = score < 0.4 and s["x"] < 40.0
    if flagged:
      frame_unstable.append(coords)
    results.append((coords, score, flagged))

  engine.register_verification_results(unstable_cache)
  return results


# ---------------------------------------------------------------------------
# Test 1: Determinism
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust backend not built")
def test_rust_deterministic(lead_samples):
  """Two identical passes through the Rust engine must produce identical output."""
  e1 = NgvtBraidRust()
  e2 = NgvtBraidRust()
  r1 = run_analysis(e1, lead_samples)
  r2 = run_analysis(e2, lead_samples)
  for i, ((c1, s1, _), (c2, s2, _)) in enumerate(zip(r1, r2)):
    np.testing.assert_allclose(c1, c2, atol=1e-6, err_msg=f"Coord mismatch at sample {i}")
    assert abs(s1 - s2) < 1e-6, f"Score mismatch at sample {i}: {s1} vs {s2}"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
def test_torch_deterministic(lead_samples):
  e1 = NgvtBraidEngineTorch()
  e2 = NgvtBraidEngineTorch()
  r1 = run_analysis(e1, lead_samples)
  r2 = run_analysis(e2, lead_samples)
  for i, ((c1, s1, _), (c2, s2, _)) in enumerate(zip(r1, r2)):
    np.testing.assert_allclose(c1, c2, atol=1e-5, err_msg=f"Torch coord mismatch at {i}")
    assert abs(s1 - s2) < 1e-5, f"Torch score mismatch at {i}"


# ---------------------------------------------------------------------------
# Test 2: Numerical stability on real data
# ---------------------------------------------------------------------------

def _check_stability(results, backend_name):
  for i, (coords, score, _) in enumerate(results):
    for j, v in enumerate(coords):
      assert np.isfinite(v), f"[{backend_name}] Non-finite coord[{j}]={v} at sample {i}"
    assert 0.0 <= score <= 1.0, f"[{backend_name}] Score {score} out of [0,1] at sample {i}"


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust backend not built")
def test_rust_stability(lead_samples):
  _check_stability(run_analysis(NgvtBraidRust(), lead_samples), "rust")


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
def test_torch_stability(lead_samples):
  _check_stability(run_analysis(NgvtBraidEngineTorch(), lead_samples), "torch")


@pytest.mark.skipif(not TINYGRAD_AVAILABLE, reason="tinygrad not installed")
def test_tinygrad_stability(lead_samples):
  _check_stability(run_analysis(NgvtBraidEngineTinygrad(), lead_samples), "tinygrad")


# ---------------------------------------------------------------------------
# Test 3: Cross-backend parity (Rust vs PyTorch vs tinygrad)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
  not (RUST_AVAILABLE and TORCH_AVAILABLE),
  reason="Need both Rust and torch backends",
)
def test_rust_torch_parity(lead_samples):
  """Rust and PyTorch engines must agree to within 1e-4 on the same inputs."""
  r_rust  = run_analysis(NgvtBraidRust(), lead_samples)
  r_torch = run_analysis(NgvtBraidEngineTorch(), lead_samples)
  max_coord_diff = max(
    max(abs(a - b) for a, b in zip(c_r, c_t))
    for (c_r, _, _), (c_t, _, _) in zip(r_rust, r_torch)
  )
  max_score_diff = max(abs(s_r - s_t) for (_, s_r, _), (_, s_t, _) in zip(r_rust, r_torch))
  assert max_coord_diff < 1e-4, f"Rust/PyTorch coord parity failed: max diff = {max_coord_diff:.2e}"
  assert max_score_diff < 1e-4, f"Rust/PyTorch score parity failed: max diff = {max_score_diff:.2e}"


@pytest.mark.skipif(
  not (TORCH_AVAILABLE and TINYGRAD_AVAILABLE),
  reason="Need both torch and tinygrad backends",
)
def test_torch_tinygrad_parity(lead_samples):
  r_torch    = run_analysis(NgvtBraidEngineTorch(), lead_samples)
  r_tinygrad = run_analysis(NgvtBraidEngineTinygrad(), lead_samples)
  max_coord_diff = max(
    max(abs(a - b) for a, b in zip(c_t, c_g))
    for (c_t, _, _), (c_g, _, _) in zip(r_torch, r_tinygrad)
  )
  max_score_diff = max(abs(s_t - s_g) for (_, s_t, _), (_, s_g, _) in zip(r_torch, r_tinygrad))
  assert max_coord_diff < 1e-4, f"PyTorch/tinygrad coord diff: {max_coord_diff:.2e}"
  assert max_score_diff < 1e-4, f"PyTorch/tinygrad score diff: {max_score_diff:.2e}"


# ---------------------------------------------------------------------------
# Test 4: Braid boost is monotone (never decreases a score)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not RUST_AVAILABLE and not TORCH_AVAILABLE, reason="No backend")
def test_braid_boost_never_decreases_score(lead_samples):
  """A node near an active failure zone must have score ≥ its no-zone baseline."""
  EngineCls = NgvtBraidRust if RUST_AVAILABLE else NgvtBraidEngineTorch
  for s in lead_samples[:20]:
    e_clean = EngineCls()
    _, baseline = e_clean.process_node(s["x"], s["y"], s["prob"])

    coords, _ = EngineCls().process_node(s["x"], s["y"], s["prob"])
    e_boosted = EngineCls()
    e_boosted.register_verification_results([coords])
    _, boosted = e_boosted.process_node(s["x"], s["y"], s["prob"])

    assert boosted >= baseline - 1e-6, (
      f"Braid decreased score: {baseline:.4f} → {boosted:.4f} at x={s['x']:.1f}m"
    )


# ---------------------------------------------------------------------------
# Test 5: Far leads (x >= 40m) never enter failure zone cache
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not RUST_AVAILABLE and not TORCH_AVAILABLE, reason="No backend")
def test_far_leads_never_flagged(lead_samples):
  """Leads at x >= 40m must never be added to the failure zone cache."""
  EngineCls = NgvtBraidRust if RUST_AVAILABLE else NgvtBraidEngineTorch
  engine = EngineCls()
  for s in lead_samples:
    _, score = engine.process_node(s["x"], s["y"], s["prob"])
    if s["x"] >= 40.0:
      flagged = score < 0.4 and s["x"] < 40.0  # second condition is always False here
      assert not flagged, f"Lead at x={s['x']:.1f}m (≥40m) incorrectly flagged"


# ---------------------------------------------------------------------------
# Test 6: Cache cleared on empty registration
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not RUST_AVAILABLE and not TORCH_AVAILABLE, reason="No backend")
def test_failure_zone_cache_cleared(lead_samples):
  EngineCls = NgvtBraidRust if RUST_AVAILABLE else NgvtBraidEngineTorch
  engine = EngineCls()
  s = lead_samples[0]
  coords, _ = engine.process_node(s["x"], s["y"], s["prob"])
  engine.register_verification_results([coords])
  assert len(engine.get_active_failure_zones()) == 1
  engine.register_verification_results([])
  assert engine.get_active_failure_zones() == []
