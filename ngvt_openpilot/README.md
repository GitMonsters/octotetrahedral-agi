# ngvt_openpilot — Drop-in Package

Offline-safe NGVT Braid analysis tools for openpilot.  
**Nothing here touches the vehicle control path.**

---

## Directory → openpilot target mapping

```
ngvt_openpilot/
├── ngvt_braid/                        → selfdrive/controls/lib/ngvt_braid/
│   ├── Cargo.toml
│   └── src/lib.rs
├── tools/
│   ├── ngvt_analysis.py               → tools/ngvt_analysis.py
│   └── ngvt_visualizer.py             → tools/ngvt_visualizer.py
├── selfdrive/test/
│   └── test_ngvt_braid.py             → selfdrive/test/test_ngvt_braid.py
├── cereal/
│   └── custom.capnp.append            → append contents to cereal/custom.capnp
└── NGVT_SAFETY_ANALYSIS.md            → keep as reference, do not commit to openpilot
```

---

## Three-Tier Architecture

| Tier | File | When to use |
|---|---|---|
| **Rust** | `ngvt_braid/src/lib.rs` | Production speed, unit tests, hard threshold (non-differentiable) |
| **PyTorch** | `ngvt_braid_torch.py` | Training, experimentation, gradient flow, TorchScript export |
| **tinygrad** | `ngvt_braid_tinygrad.py` | openpilot inference — mirrors modeld's runtime, GPU via METAL/CUDA |

All three implement the same API (`process_node`, `register_verification_results`,
`get_active_failure_zones`) so they're interchangeable in `tools/ngvt_analysis.py`.

The **PyTorch version** replaces the hard `dist < 1.5` threshold with a
differentiable **Gaussian RBF kernel** so gradients flow through the Braid boost.
The tinygrad version matches the PyTorch math exactly for numerical parity.

```
Train (PyTorch) → export weights → Load (tinygrad) → openpilot inference
```

---

## Quick start (inside an openpilot checkout)

```bash
# 1. Copy files into your openpilot fork
cp -r ngvt_openpilot/ngvt_braid   selfdrive/controls/lib/ngvt_braid
cp    ngvt_openpilot/tools/*.py    tools/
cp    ngvt_openpilot/selfdrive/test/test_ngvt_braid.py  selfdrive/test/
cp    ngvt_openpilot/ngvt_braid_torch.py     tools/
cp    ngvt_openpilot/ngvt_braid_tinygrad.py  tools/
cat   ngvt_openpilot/cereal/custom.capnp.append >> cereal/custom.capnp

# 2. Build the Rust extension (fastest, for production use)
pip install maturin
maturin develop --manifest-path selfdrive/controls/lib/ngvt_braid/Cargo.toml

# 3. Run unit tests (no vehicle, no live processes)
pytest selfdrive/test/test_ngvt_braid.py -v

# 4. Analyze a recorded log (offline) — auto-selects available backend
python tools/ngvt_analysis.py "a2a0ccea32023010|2023-07-27--13-01-19/0" --out results.json

# 5. Visualize instability zones
python tools/ngvt_visualizer.py results.json
python tools/ngvt_visualizer.py results.json --show-all-leads --save leads.png

# 6. Run tinygrad version (openpilot-native backend)
python ngvt_braid_tinygrad.py          # self-tests + parity check vs PyTorch
METAL=1 python ngvt_braid_tinygrad.py  # Apple GPU

# 7. Run PyTorch version (training / TorchScript export)
python ngvt_braid_torch.py             # self-tests + exports /tmp/ngvt_braid.pt
```

---

## Architecture

```
LogReader(.rlog / route_id)
        │
        │  modelV2.leadsV3[i].{x, y, prob}
        ▼
NgvtBraidEngine (Rust/PyO3)
  ├─ process_node(x, y, prob)  →  (torus_coords [X,Y,Z], adjusted_score)
  ├─ register_verification_results(unstable_nodes)
  └─ get_active_failure_zones()
        │
        ▼
  FrameResult list  →  JSON  →  ngvt_visualizer.py  →  3D manifold plots
```

The **Braid boost** (3× amplification) fires when a new node lands within `r=1.5`
of a failure zone cached from the previous frame.  Score is always clamped to `[0, 1]`.

---

## Building the Rust crate (standalone)

```bash
cd selfdrive/controls/lib/ngvt_braid
cargo test          # run Rust unit tests
cargo build --release
```

---

## Safety boundary

See `NGVT_SAFETY_ANALYSIS.md` for the full analysis of what would be required
before any live-vehicle integration PR.
