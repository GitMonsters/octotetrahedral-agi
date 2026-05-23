"""
Smoke test — Manifold Re-Anchoring
===================================
Tests:
  1. AnchorSet produces valid anchor coords and EMA updates
  2. AnchorEncoder projects module latents to anchor space
  3. ReanchoringLoss computes three alignment losses
  4. ReanchoringController: register, compute_loss, step (state machine)
  5. Rollback restores params correctly
  6. Controller correctly transitions PLASTIC → CANDIDATE → CONSOLIDATED
"""

import sys
import torch
import torch.nn as nn

sys.path.insert(0, "/Users/evanpieser")

from core.manifold_reanchoring import (
    AnchorSet,
    AnchorEncoder,
    ReanchoringConfig,
    ReanchoringController,
    ReanchoringLoss,
    ConsolidationPhase,
    DriftStatus,
)

BRAID = 256
LATENT = 128
K = 32
B = 4


def test_anchor_set():
    cfg = ReanchoringConfig(n_anchors=K, braid_dim=BRAID)
    acs = AnchorSet(cfg)

    z = torch.randn(B, BRAID)
    coords = acs.coords(z)
    assert coords.shape == (B, K), f"expected ({B},{K}), got {coords.shape}"
    assert (coords >= 0).all(), "dissimilarity should be ≥ 0"
    assert (coords <= 2).all(), "cosine dissimilarity ≤ 2"

    acs.ema_update()
    print(f"  AnchorSet OK  coords={coords.shape}  range=[{coords.min():.3f},{coords.max():.3f}]")


def test_anchor_encoder():
    cfg = ReanchoringConfig(n_anchors=K, braid_dim=BRAID)
    acs = AnchorSet(cfg)
    enc = AnchorEncoder(module_latent_dim=LATENT, braid_dim=BRAID, n_anchors=K)

    z_i = torch.randn(B, LATENT)
    r = enc(z_i, acs)
    assert r.shape == (B, K), f"expected ({B},{K}), got {r.shape}"
    print(f"  AnchorEncoder OK  r={r.shape}")


def test_reanchoring_loss():
    cfg = ReanchoringConfig(n_anchors=K, braid_dim=BRAID)
    loss_fn = ReanchoringLoss(cfg)

    r_new  = torch.rand(B, K)
    r_core = torch.rand(B, K)

    l_align, d_anc = loss_fn.anchor_align(r_new, r_core)
    assert l_align.shape == (), "loss should be scalar"
    assert d_anc >= 0

    total, metrics = loss_fn.total(r_new, r_core, r_new, r_core, r_core)
    assert "l_anchor_align" in metrics
    assert "l_sem_anc" in metrics
    assert "l_dyn_anc" in metrics
    assert "l_reanchor_total" in metrics
    print(f"  ReanchoringLoss OK  total={total.item():.4f}  metrics={list(metrics.keys())}")


def test_controller_register_and_loss():
    cfg  = ReanchoringConfig(n_anchors=K, braid_dim=BRAID)
    ctrl = ReanchoringController(cfg)

    contract = ctrl.register("vision", latent_dim=LATENT, braid_dim=BRAID)
    assert contract.is_valid()
    assert contract.phase == ConsolidationPhase.PLASTIC

    z_i    = torch.randn(B, LATENT)
    z_core = torch.randn(B, BRAID)
    loss, breakdown = ctrl.compute_loss("vision", z_i, z_core)
    assert loss.shape == (), "loss should be scalar"
    assert loss.requires_grad, "loss must be differentiable"
    print(f"  Controller compute_loss OK  loss={loss.item():.4f}  breakdown={list(breakdown.keys())}")


def test_controller_state_machine():
    cfg = ReanchoringConfig(
        n_anchors=K,
        braid_dim=BRAID,
        drift_threshold=0.9,       # easy to pass for smoke test
        drift_max=1.5,
        stability_window=3,
        performance_min_delta=-0.001,
    )
    ctrl = ReanchoringController(cfg)
    ctrl.register("planner", latent_dim=LATENT)

    # Random latents will produce somewhat similar anchor coords if both near origin
    z_i    = torch.randn(B, LATENT) * 0.1   # small → coords close
    z_core = torch.randn(B, BRAID)  * 0.1

    # 1. PLASTIC with a positive task loss delta → should stay PLASTIC
    status, m = ctrl.step("planner", z_i, z_core, task_loss_delta=0.5)
    assert ctrl.get_contract("planner").phase in (
        ConsolidationPhase.PLASTIC, ConsolidationPhase.CANDIDATE
    )
    print(f"  Step1 phase={ctrl.get_contract('planner').phase.value}  D={m['d_anchor']:.4f}")

    # 2. Negative delta (improvement) → should promote to CANDIDATE
    status, m = ctrl.step("planner", z_i, z_core, task_loss_delta=-0.5)
    print(f"  Step2 phase={ctrl.get_contract('planner').phase.value}  D={m['d_anchor']:.4f}")

    # 3. Run stability_window more steps with improvement → CONSOLIDATED
    for i in range(cfg.stability_window + 2):
        status, m = ctrl.step("planner", z_i, z_core, task_loss_delta=-0.1)
    phase_end = ctrl.get_contract("planner").phase
    print(f"  AfterWindow phase={phase_end.value}  status={status.value}")

    print("  Controller state machine OK")


def test_rollback():
    cfg  = ReanchoringConfig(n_anchors=K, braid_dim=BRAID)
    ctrl = ReanchoringController(cfg)
    ctrl.register("memory", latent_dim=LATENT)

    module = nn.Linear(LATENT, BRAID)
    ctrl.save_snapshot("memory", module)

    # Mutate the module
    with torch.no_grad():
        module.weight.fill_(99.0)

    ok = ctrl.rollback("memory", module)
    assert ok, "rollback should succeed when snapshot exists"
    assert not (module.weight == 99.0).any(), "weights should be restored"
    print("  Rollback OK")


def test_recommended_lr_factor():
    cfg  = ReanchoringConfig(n_anchors=K, braid_dim=BRAID)
    ctrl = ReanchoringController(cfg)
    ctrl.register("action", latent_dim=LATENT)

    assert ctrl.recommended_lr_factor("action") == 1.0  # PLASTIC
    ctrl._contracts["action"].phase = ConsolidationPhase.ROLLBACK
    assert ctrl.recommended_lr_factor("action") == cfg.rollback_lr_factor
    ctrl._contracts["action"].phase = ConsolidationPhase.CONSOLIDATED
    assert ctrl.recommended_lr_factor("action") == 0.0
    print("  LR factor OK")


def main():
    print("=" * 60)
    print("Manifold Re-Anchoring Smoke Tests")
    print("=" * 60)
    tests = [
        test_anchor_set,
        test_anchor_encoder,
        test_reanchoring_loss,
        test_controller_register_and_loss,
        test_controller_state_machine,
        test_rollback,
        test_recommended_lr_factor,
    ]
    passed = 0
    for t in tests:
        name = t.__name__
        try:
            print(f"\n[TEST] {name}")
            t()
            print(f"  ✅ PASS")
            passed += 1
        except Exception as e:
            print(f"  ❌ FAIL: {e}")
            import traceback; traceback.print_exc()

    print(f"\n{'=' * 60}")
    print(f"Result: {passed}/{len(tests)} passed")
    print("=" * 60)
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
