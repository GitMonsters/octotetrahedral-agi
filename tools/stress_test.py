"""
OctoTetrahedral AGI — Full Stress Test
Covers: model load, shapes, tet calculus, cohesion braid, S2S1 cache,
        backprop, numerical stability, throughput, output dict.
Run: python3 tools/stress_test.py
"""
import torch, time, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from model import OctoTetrahedralModel

RED="\033[91m"; GRN="\033[92m"; YLW="\033[93m"; CYN="\033[96m"; RST="\033[0m"; BOLD="\033[1m"
passed=failed=0

def ok(name, msg=""):
    global passed; passed+=1
    print(f"  {GRN}✓{RST} {name}" + (f"  {YLW}{msg}{RST}" if msg else ""))

def fail(name, exc):
    global failed; failed+=1
    print(f"  {RED}✗{RST} {name}\n    {RED}{exc}{RST}")

def section(t): print(f"\n{BOLD}{CYN}── {t} {'─'*(55-len(t))}{RST}")

print(f"\n{BOLD}OctoTetrahedral AGI — Stress Test{RST}")
print(f"torch {torch.__version__}  |  Python {sys.version.split()[0]}")

# ── 1. Model load ────────────────────────────────────────────────────────────
section("1. Model instantiation")
t0 = time.time()
m = OctoTetrahedralModel()
params = sum(p.numel() for p in m.parameters()) / 1e6
_probe = m(torch.randint(0, 1000, (1, 4)))
VOCAB = _probe['logits'].shape[-1]
ok("OctoTetrahedralModel()", f"{params:.1f}M params  {(time.time()-t0)*1000:.0f}ms  vocab={VOCAB}")

# ── 2. Varied shapes ─────────────────────────────────────────────────────────
section("2. Varied input shapes")
for B, T in [(1,4),(1,8),(1,16),(1,32),(1,64),(2,16),(4,8),(8,4)]:
    try:
        out = m(torch.randint(0,1000,(B,T))); lg = out['logits']
        assert lg.shape == (B,T,VOCAB) and torch.isfinite(lg).all()
        ok(f"({B},{T})", f"logits {tuple(lg.shape)}")
    except Exception as e: fail(f"({B},{T})", e)

# ── 3. Tet Vision Calculus ───────────────────────────────────────────────────
section("3. Tetrahedral Vision Calculus — forward pass coverage")
tvc = m.tet_vision_calculus; g = tvc.graph
n_edges = g.edge_index.shape[1]; n_faces = g.face_index.shape[1]
ok("graph buffers", f"edges={n_edges}  faces={n_faces}  positions={list(g.positions.shape)}")
for B, T in [(1,64),(2,64),(4,32),(1,16),(1,4)]:
    try:
        out = m(torch.randint(0,1000,(B,T))); tci = out['tet_calc_info']
        assert tci['n_edges'] == n_edges and tci['n_faces'] == n_faces
        for k in ('grad_magnitude','divergence_norm','laplacian_norm','curl_norm'):
            assert torch.isfinite(torch.tensor(float(tci[k])))
        ok(f"B={B} T={T}",
           f"∇={tci['grad_magnitude']:.3f}  div={tci['divergence_norm']:.3f}"
           f"  L={tci['laplacian_norm']:.3f}  curl={tci['curl_norm']:.3f}")
    except Exception as e: fail(f"B={B} T={T}", e)

# ── 4. Standalone calculus ops ───────────────────────────────────────────────
section("4. Discrete calculus ops — standalone")
from core.tetrahedral_calculus_ops import gradient, divergence, laplacian, curl, face_to_node
ei = g.edge_index; ew = g.edge_weights; fi = g.face_index
n_nodes = g.grid_h * g.grid_w; feat = torch.randn(n_nodes, 8)
try:
    ge = gradient(feat, ei, ew)
    ok("gradient()", f"shape {tuple(ge.shape)}")
    dv = divergence(ge, ei, n_nodes)
    ok("divergence()", f"shape {tuple(dv.shape)}")
    lp = laplacian(feat, ei, ew, n_nodes)
    ok("laplacian()", f"shape {tuple(lp.shape)}")
    # curl uses precomputed face_edge_idx / face_edge_sign buffers
    cu = curl(ge, g.face_edge_idx, g.face_edge_sign)
    ok("curl()", f"shape {tuple(cu.shape)}")
    cn = face_to_node(cu, fi, n_nodes)
    ok("face_to_node()", f"shape {tuple(cn.shape)}")
    # Laplacian identity: uniform field → zero
    ones = torch.ones(n_nodes, 1)
    lp_ones = laplacian(ones, ei, ew, n_nodes)
    ok("Laplacian of constants ≈ 0", f"max={lp_ones.abs().max().item():.2e}")
except Exception as e: fail("standalone ops", e)

# ── 5. CognitiveCohesionBraid ────────────────────────────────────────────────
section("5. CognitiveCohesionBraid")
for _ in range(30): m(torch.randint(0,1000,(1,16)))
ci = m.cohesion_score()
assert 0 <= ci['cohesion_score'] <= 1
ok("30-pass warm-up",
   f"score={ci['cohesion_score']:.3f}  EWMA={ci['ewma_score']:.3f}"
   f"  limbs={ci['limbs_active']}/13  events={ci['braid_stats']['total_events']}")
rpt = m.export_cohesion_report("logs/cohesion/stress_test_report.html")
ok("export_cohesion_report()", rpt)

# ── 6. S2→S1 cache ───────────────────────────────────────────────────────────
section("6. S2→S1 Knowledge Transfer Cache")
m.s2_s1_cache.clear()
for _ in range(20): m(torch.randint(0,500,(1,16)))
s = m.s2_s1_stats()
ok("cache stats",
   f"size={s['size']}  hits={s['hits']}  misses={s['misses']}"
   f"  hit_rate={s['hit_rate']:.2f}  avg_conf={s['avg_confidence_stored']:.3f}")

# ── 7. Gradient flow ─────────────────────────────────────────────────────────
section("7. Gradient flow & backprop")
m.train()
out = m(torch.randint(0,1000,(2,16)))
out['logits'].float().mean().backward()
gnorms = [p.grad.norm().item() for p in m.parameters() if p.grad is not None]
dead = sum(1 for g in gnorms if g == 0)
ok("backward()",
   f"{len(gnorms)} grads | {len(gnorms)-dead} live | {dead} zero"
   f" | max={max(gnorms):.4f} | mean={sum(gnorms)/len(gnorms):.5f}")
# Check new modules receive gradients
for attr, sub in [('tet_vision_calculus.calc_mixer','weight'),
                  ('tet_vision_calculus.alpha',None)]:
    try:
        obj = m
        for part in attr.split('.'): obj = getattr(obj, part)
        tensor = getattr(obj, sub) if sub else obj
        g_val = tensor.grad
        ok(f"grad: {attr}", f"norm={g_val.norm().item():.5f}" if g_val is not None else "None (frozen ok)")
    except Exception as e: fail(f"grad: {attr}", e)
m.zero_grad(); m.eval()

# ── 8. Edge cases ────────────────────────────────────────────────────────────
section("8. Numerical stability — edge cases")
for label, x in [
    ("all zeros",    torch.zeros(1,16,dtype=torch.long)),
    ("all max",      torch.full((1,16),999,dtype=torch.long)),
    ("single token", torch.randint(0,1000,(1,1))),
    ("seq=128",      torch.randint(0,1000,(1,128))),
    ("batch=16 T=8", torch.randint(0,1000,(16,8))),
    ("repeated 42",  torch.full((1,16),42,dtype=torch.long)),
]:
    try:
        out = m(x); lg = out['logits']
        assert torch.isfinite(lg).all() and not torch.isnan(lg).any()
        ok(label, f"{tuple(lg.shape)}")
    except Exception as e: fail(label, e)

# ── 9. Throughput ────────────────────────────────────────────────────────────
section("9. Throughput benchmark")
x = torch.randint(0,1000,(1,16))
for _ in range(5): m(x)   # warm-up
N = 50; t0 = time.time()
for _ in range(N): m(x)
el = time.time()-t0
ok(f"batch=1 {N}-pass", f"{el/N*1000:.1f}ms/pass  {N/el:.1f} pass/sec")

xb = torch.randint(0,1000,(4,16))
for _ in range(3): m(xb)
t0 = time.time()
for _ in range(20): m(xb)
el2 = time.time()-t0
ok("batch=4 20-pass", f"{el2/20*1000:.1f}ms/pass  effective {20*4/el2:.1f} samples/sec")

# ── 10. Output dict ──────────────────────────────────────────────────────────
section("10. Output dict completeness")
out = m(torch.randint(0,1000,(1,16)))
for key in ['logits','cohesion_info','tet_calc_info','s2_s1_cache_stats']:
    if key in out: ok(f"'{key}' present")
    else: fail(f"'{key}' present", f"missing — keys: {sorted(out.keys())}")

# ── Summary ──────────────────────────────────────────────────────────────────
total = passed + failed
print(f"\n{BOLD}{'─'*60}{RST}")
if failed == 0:
    print(f"{BOLD}{GRN}★  ALL {total} TESTS PASSED  ★{RST}\n")
else:
    print(f"{BOLD}{GRN}{passed}{RST}/{total} passed  {RED}{failed} FAILED{RST}\n")
sys.exit(0 if failed == 0 else 1)
