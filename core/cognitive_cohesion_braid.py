"""
Cognitive Cohesion Braid — Master Compound Integration Layer
==============================================================

Recompiles SIMULA + EUPHAN + HERMES + CodeGen + 14 limbs + 20 skills +
compound subsystems into a single braided cognitive substrate where every
aspect feeds every other.

Design principles
-----------------
1. **Braided, not parallel** — bridges cross-pollinate. SIMULA failures route to
   HERMES; EUPHAN low-confidence events trigger SIMULA augmentation; HERMES
   solver outcomes reweight EUPHAN attention.
2. **Cohesion = measurable** — every braid emits a `cohesion_score` derived
   from skill activation alignment, limb event entropy, and feedback latency.
3. **Limb-skill bound** — each of the 20 skills binds to one of the 14 limbs;
   activating a skill "lights up" its limb and propagates through the braid.
4. **Closed-loop** — active braid loops plus CodeGen refinement feedback:
       SIMULA → EUPHAN        (track each synthesized example)
       EUPHAN → HERMES        (queue solver agents on weak limbs)
       HERMES → SIMULA        (trigger targeted augmentation on failures)
       CodeGen → MetaCognition (quality-check generated code/tool output)
       MetaCognition → CodeGen (route critique/refinement back to CodeGen)
5. **Zero overhead when disabled** — all braids gated by feature flags.

This module is *pure orchestration* — no torch dependency. It composes the
existing CompoundIntegrationManager, hermes_compound_bridge, euphan_compound_bridge,
and simula_compound_bridge into a coherent whole.

Usage
-----
    from core.cognitive_cohesion_braid import CognitiveCohesionBraid

    braid = CognitiveCohesionBraid(enable_all=True)
    braid.bind_simula(simula_bridge)
    braid.bind_euphan(euphan_bridge)
    braid.bind_hermes(hermes_bridge)

    # During training/inference, just call the braid:
    braid.on_simula_data(event)        # auto-routes to EUPHAN + cohesion
    braid.on_euphan_event(event)       # auto-routes to HERMES + cohesion
    braid.on_hermes_result(result)     # auto-routes to SIMULA + cohesion

    score = braid.cohesion_score()     # 0..1, higher = more aligned
    report = braid.generate_html_report("logs/cohesion/report.html")
"""

from __future__ import annotations

import json
import math
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple


# ───────────────────────────────────────────────────────────────────────────────
# Skill → Limb registry  (mirrors the SQL skills table — single source of truth)
# ───────────────────────────────────────────────────────────────────────────────

SKILL_LIMB_MAP: Dict[str, str] = {
    # SIMULA
    "taxonomy-design":          "reasoning",
    "metaprompt-gen":           "language",
    "complexity-tuning":        "action",
    "dual-critic":              "reasoning",
    "mode-collapse-prevent":    "perception",
    # EUPHAN
    "log-parsing":              "metacognition",
    "workflow-viz":             "perception",
    "role-filtering":           "metacognition",
    "session-replay":           "memory",
    # HERMES
    "agent-persistence":        "memory",
    "skill-assignment":         "action",
    "trigger-execution":        "planning",
    "multi-agent-coord":        "planning",
    # VortexDisCode
    "nvidia-code-gen":          "codegen",
    "torus-code-map":           "codegen",
    "code-refactor":            "codegen",
    "code-debug":               "codegen",
    # OctoTetrahedral
    "agent-observability":      "metacognition",
    # Compound-transform reasoning (V30+ pipeline)
    "scale-ratio-routing":      "spatial",
    "compound-transform-order": "reasoning",
}

SKILL_SOURCE: Dict[str, str] = {
    "taxonomy-design": "SIMULA", "metaprompt-gen": "SIMULA",
    "complexity-tuning": "SIMULA", "dual-critic": "SIMULA",
    "mode-collapse-prevent": "SIMULA",
    "log-parsing": "EUPHAN", "workflow-viz": "EUPHAN",
    "role-filtering": "EUPHAN", "session-replay": "EUPHAN",
    "agent-persistence": "HERMES", "skill-assignment": "HERMES",
    "trigger-execution": "HERMES", "multi-agent-coord": "HERMES",
    "nvidia-code-gen": "VortexDisCode", "torus-code-map": "VortexDisCode",
    "code-refactor": "VortexDisCode", "code-debug": "VortexDisCode",
    "agent-observability": "OctoTetrahedral",
    "scale-ratio-routing": "OctoTetrahedral",
    "compound-transform-order": "OctoTetrahedral",
}

ALL_LIMBS: Tuple[str, ...] = (
    "memory", "planning", "language", "spatial", "reasoning",
    "metacognition", "perception", "action", "codegen",
    "emotion", "empathy", "ethics", "imagination", "visualization",
)


# ───────────────────────────────────────────────────────────────────────────────
# Configuration
# ───────────────────────────────────────────────────────────────────────────────

@dataclass
class CohesionConfig:
    enabled: bool = True

    # Cross-bridge braiding loops
    braid_simula_to_euphan: bool = True
    braid_euphan_to_hermes: bool = True
    braid_hermes_to_simula: bool = True

    # Triggers
    weak_confidence_threshold: float = 0.5     # EUPHAN→HERMES queue trigger
    failure_augmentation_count: int = 3        # HERMES→SIMULA fan-out

    # Cohesion scoring
    history_window: int = 256                   # rolling event window
    coherence_decay: float = 0.95               # EWMA factor

    # Reporting
    output_dir: str = "logs/cohesion"
    report_interval_events: int = 100


@dataclass
class BraidEvent:
    """Unified event flowing through the braid."""
    source: str                     # 'simula' | 'euphan' | 'hermes'
    kind: str                       # event subtype
    limb: Optional[str] = None
    skill: Optional[str] = None
    confidence: float = 1.0
    success: bool = True
    timestamp: float = field(default_factory=time.time)
    payload: Dict[str, Any] = field(default_factory=dict)
    routed_to: List[str] = field(default_factory=list)


# ───────────────────────────────────────────────────────────────────────────────
# Cohesion scorer — measures alignment across the system
# ───────────────────────────────────────────────────────────────────────────────

class CohesionScorer:
    """Tracks limb activation, skill firing, and cross-bridge feedback latency.

    cohesion_score = 0.4 * limb_balance        (entropy of limb activations)
                   + 0.3 * skill_coverage      (fraction of skills exercised)
                   + 0.3 * feedback_latency    (1 - normalized latency)
    """

    def __init__(self, window: int = 256, decay: float = 0.95):
        self.window = window
        self.decay = decay
        self.limb_counts: Dict[str, int] = defaultdict(int)
        self.skill_counts: Dict[str, int] = defaultdict(int)
        self.events: Deque[BraidEvent] = deque(maxlen=window)
        self.feedback_latencies: Deque[float] = deque(maxlen=window)
        self._ewma_score: float = 0.0

    def record(self, ev: BraidEvent) -> None:
        if self.events.maxlen is not None and len(self.events) == self.events.maxlen:
            old = self.events[0]
            if old.limb:
                self.limb_counts[old.limb] -= 1
                if self.limb_counts[old.limb] <= 0:
                    del self.limb_counts[old.limb]
            if old.skill:
                self.skill_counts[old.skill] -= 1
                if self.skill_counts[old.skill] <= 0:
                    del self.skill_counts[old.skill]
        self.events.append(ev)
        if ev.limb:
            self.limb_counts[ev.limb] += 1
        if ev.skill:
            self.skill_counts[ev.skill] += 1

    def record_feedback_latency(self, seconds: float) -> None:
        self.feedback_latencies.append(max(0.0, seconds))

    def _entropy(self, counts: Dict[str, int], universe_size: int) -> float:
        total = sum(counts.values())
        if total == 0:
            return 0.0
        h = 0.0
        for c in counts.values():
            p = c / total
            if p > 0:
                h -= p * math.log(p)
        return h / math.log(max(universe_size, 2))  # normalize to [0, 1]

    def compute(self) -> Dict[str, float]:
        limb_balance = self._entropy(self.limb_counts, len(ALL_LIMBS))
        skill_coverage = len(self.skill_counts) / max(len(SKILL_LIMB_MAP), 1)

        if self.feedback_latencies:
            avg_lat = sum(self.feedback_latencies) / len(self.feedback_latencies)
            # 1s = 0.5 cohesion, 0s = 1.0
            latency_score = math.exp(-avg_lat)
        else:
            latency_score = 0.5

        score = (0.4 * limb_balance) + (0.3 * skill_coverage) + (0.3 * latency_score)
        # EWMA smoothing
        self._ewma_score = self.decay * self._ewma_score + (1.0 - self.decay) * score
        return {
            "cohesion_score": round(score, 4),
            "ewma_score": round(self._ewma_score, 4),
            "limb_balance": round(limb_balance, 4),
            "skill_coverage": round(skill_coverage, 4),
            "latency_score": round(latency_score, 4),
            "events_in_window": len(self.events),
            "limbs_active": len(self.limb_counts),
            "skills_active": len(self.skill_counts),
        }


# ───────────────────────────────────────────────────────────────────────────────
# Master Cognitive Cohesion Braid
# ───────────────────────────────────────────────────────────────────────────────

class CognitiveCohesionBraid:
    """The unified compound braid across all bridges, limbs, and skills.

    Public API
    ----------
    bind_simula(bridge)            register the SIMULA bridge / callback
    bind_euphan(bridge)            register the EUPHAN bridge / callback
    bind_hermes(bridge)            register the HERMES bridge / callback

    on_simula_data(event)          ingress for SIMULA events
    on_euphan_event(event)         ingress for EUPHAN limb events
    on_hermes_result(result)       ingress for HERMES task results

    cohesion_score()               current cohesion metrics
    generate_html_report(path)     write dashboard
    export_json(path)              write metrics JSON
    """

    def __init__(self, config: Optional[CohesionConfig] = None,
                 enable_all: bool = True):
        self.config = config or CohesionConfig(enabled=enable_all)
        self.scorer = CohesionScorer(
            window=self.config.history_window,
            decay=self.config.coherence_decay,
        )

        # Bridges (any object exposing .record_* / .queue_* methods, or None)
        self.simula_bridge: Any = None
        self.euphan_bridge: Any = None
        self.hermes_bridge: Any = None

        # Optional callback hooks (used when a bridge isn't a full object)
        self.simula_augment_cb: Optional[Callable[[Dict[str, Any]], None]] = None
        self.hermes_enqueue_cb: Optional[Callable[[Dict[str, Any]], None]] = None
        self.euphan_log_cb:    Optional[Callable[[Dict[str, Any]], None]] = None

        self.event_log: Deque[BraidEvent] = deque(maxlen=self.config.history_window)
        self.braid_stats = {
            "simula_to_euphan": 0,
            "euphan_to_hermes": 0,
            "hermes_to_simula": 0,
            "total_events": 0,
            "started_at": datetime.utcnow().isoformat() + "Z",
        }

        # Optional RSI HashGrid sub-system (attached via attach_rsi_hashgrid)
        self._rsi_hashgrid: Any = None

    # ── Binding ────────────────────────────────────────────────────────────
    def bind_simula(self, bridge: Any = None,
                    augment_cb: Optional[Callable] = None) -> "CognitiveCohesionBraid":
        self.simula_bridge = bridge
        if augment_cb is not None:
            self.simula_augment_cb = augment_cb
        return self

    def bind_euphan(self, bridge: Any = None,
                    log_cb: Optional[Callable] = None) -> "CognitiveCohesionBraid":
        self.euphan_bridge = bridge
        if log_cb is not None:
            self.euphan_log_cb = log_cb
        return self

    def bind_hermes(self, bridge: Any = None,
                    enqueue_cb: Optional[Callable] = None) -> "CognitiveCohesionBraid":
        self.hermes_bridge = bridge
        if enqueue_cb is not None:
            self.hermes_enqueue_cb = enqueue_cb
        return self

    # ── RSI HashGrid ───────────────────────────────────────────────────────
    def attach_rsi_hashgrid(self, rsi_hg: Any) -> "CognitiveCohesionBraid":
        """Attach a CompoundingCohesionRSIHashgrid instance.

        When attached, `gamma_cycle_step()` becomes available and RSI
        diagnostics are included in `cohesion_score()` output.
        """
        self._rsi_hashgrid = rsi_hg
        return self

    def gamma_cycle_step(
        self,
        limb_states: Any,       # torch.Tensor [B, num_limbs, hidden_dim]
        cohesion_override: Optional[float] = None,
    ) -> Tuple[Any, float]:
        """Run one RSI HashGrid gamma cycle.

        Args:
            limb_states        : limb hidden states tensor
            cohesion_override  : use this score instead of computing it now

        Returns
        -------
        combine_weight_deltas : [num_limbs] tensor
        rsi_value             : float
        """
        if self._rsi_hashgrid is None:
            raise RuntimeError(
                "No RSI HashGrid attached. Call attach_rsi_hashgrid() first."
            )
        score = cohesion_override
        if score is None:
            score = self.scorer.compute().get("cohesion_score", 0.5)
        return self._rsi_hashgrid.step(limb_states, float(score))

    # ── Ingress: SIMULA ────────────────────────────────────────────────────
    def on_simula_data(self, event: Dict[str, Any]) -> BraidEvent:
        ev = BraidEvent(
            source="simula",
            kind=event.get("kind", "data_generation"),
            skill=event.get("skill"),
            limb=SKILL_LIMB_MAP.get(event.get("skill", ""), event.get("limb")),
            confidence=float(event.get("avg_quality", event.get("confidence", 1.0))),
            success=bool(event.get("success", True)),
            payload=event,
        )
        self._ingest(ev)

        # Braid: SIMULA → EUPHAN (log every batch through EUPHAN timeline)
        if self.config.enabled and self.config.braid_simula_to_euphan:
            routed = self._route_to_euphan({
                "kind": "simula_data_batch",
                "limb": ev.limb or "perception",
                "confidence": ev.confidence,
                "num_examples": event.get("num_examples", 0),
                "ts": ev.timestamp,
            })
            if routed:
                ev.routed_to.append("euphan")
                self.braid_stats["simula_to_euphan"] += 1
        return ev

    # ── Ingress: EUPHAN ────────────────────────────────────────────────────
    def on_euphan_event(self, event: Dict[str, Any]) -> BraidEvent:
        ev = BraidEvent(
            source="euphan",
            kind=event.get("action", event.get("kind", "limb_event")),
            limb=event.get("limb"),
            skill=event.get("skill"),
            confidence=float(event.get("confidence", 1.0)),
            success=bool(event.get("success", True)),
            payload=event,
        )
        self._ingest(ev)

        # Braid: EUPHAN → HERMES (low-confidence events spawn solver tasks)
        if (self.config.enabled and self.config.braid_euphan_to_hermes
                and ev.confidence < self.config.weak_confidence_threshold):
            t0 = time.time()
            routed = self._route_to_hermes({
                "kind": "weak_limb_recovery",
                "limb": ev.limb,
                "confidence": ev.confidence,
                "trigger": ev.kind,
                "skills_needed": self._skills_for_limb(ev.limb),
                "ts": ev.timestamp,
            })
            if routed:
                self.scorer.record_feedback_latency(time.time() - t0)
                ev.routed_to.append("hermes")
                self.braid_stats["euphan_to_hermes"] += 1
        return ev

    # ── Ingress: HERMES ────────────────────────────────────────────────────
    def on_hermes_result(self, result: Dict[str, Any]) -> BraidEvent:
        ev = BraidEvent(
            source="hermes",
            kind=result.get("task_type", "solve"),
            confidence=float(result.get("confidence", 1.0)),
            success=bool(result.get("success", False)),
            payload=result,
        )
        # Map agent skills back to limbs
        for skill in result.get("skills_used", []):
            if skill in SKILL_LIMB_MAP:
                ev.skill = skill
                ev.limb = SKILL_LIMB_MAP[skill]
                break
        self._ingest(ev)

        # Braid: HERMES → SIMULA (failed solves trigger targeted augmentation)
        if (self.config.enabled and self.config.braid_hermes_to_simula
                and not ev.success):
            t0 = time.time()
            routed = self._route_to_simula({
                "kind": "failure_augmentation_request",
                "task_id": result.get("task_id"),
                "n_examples": self.config.failure_augmentation_count,
                "weak_limb": ev.limb,
                "ts": ev.timestamp,
            })
            if routed:
                self.scorer.record_feedback_latency(time.time() - t0)
                ev.routed_to.append("simula")
                self.braid_stats["hermes_to_simula"] += 1
        return ev

    # ── Internal: routing helpers (graceful no-op if bridge missing) ───────
    def _route_to_euphan(self, payload: Dict[str, Any]) -> bool:
        if self.euphan_log_cb:
            try:
                self.euphan_log_cb(payload)
                return True
            except Exception:
                return False
        if self.euphan_bridge and hasattr(self.euphan_bridge, "log_event"):
            try:
                self.euphan_bridge.log_event(payload)
                return True
            except Exception:
                return False
        return False

    def _route_to_hermes(self, payload: Dict[str, Any]) -> bool:
        if self.hermes_enqueue_cb:
            try:
                self.hermes_enqueue_cb(payload)
                return True
            except Exception:
                return False
        if self.hermes_bridge and hasattr(self.hermes_bridge, "queue_solve_task"):
            try:
                self.hermes_bridge.queue_solve_task(payload)
                return True
            except Exception:
                return False
        return False

    def _route_to_simula(self, payload: Dict[str, Any]) -> bool:
        if self.simula_augment_cb:
            try:
                self.simula_augment_cb(payload)
                return True
            except Exception:
                return False
        if self.simula_bridge and hasattr(self.simula_bridge, "augment"):
            try:
                self.simula_bridge.augment(payload)
                return True
            except Exception:
                return False
        return False

    def _ingest(self, ev: BraidEvent) -> None:
        self.scorer.record(ev)
        self.event_log.append(ev)
        self.braid_stats["total_events"] += 1

    def _skills_for_limb(self, limb: Optional[str]) -> List[str]:
        if not limb:
            return []
        return [s for s, l in SKILL_LIMB_MAP.items() if l == limb]

    # ── Reporting ──────────────────────────────────────────────────────────
    def cohesion_score(self) -> Dict[str, Any]:
        out = self.scorer.compute()
        out.update({
            "braid_stats": dict(self.braid_stats),
            "skills_per_source": {
                src: sum(1 for s in self.scorer.skill_counts
                         if SKILL_SOURCE.get(s) == src)
                for src in (
                    "SIMULA",
                    "EUPHAN",
                    "HERMES",
                    "VortexDisCode",
                    "OctoTetrahedral",
                )
            },
        })
        if self._rsi_hashgrid is not None:
            out["rsi_hashgrid"] = self._rsi_hashgrid.get_diagnostics()
        return out

    def export_json(self, path: Optional[str] = None) -> str:
        path = path or os.path.join(self.config.output_dir, "cohesion_metrics.json")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.cohesion_score(), f, indent=2)
        return path

    def generate_html_report(self, path: Optional[str] = None) -> str:
        path = path or os.path.join(self.config.output_dir, "cohesion_report.html")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        m = self.cohesion_score()

        rows_limbs = "".join(
            f"<tr><td>{l}</td><td>{self.scorer.limb_counts.get(l, 0)}</td></tr>"
            for l in ALL_LIMBS
        )
        rows_skills = "".join(
            f"<tr><td>{s}</td><td>{SKILL_SOURCE.get(s,'?')}</td>"
            f"<td>{SKILL_LIMB_MAP.get(s,'?')}</td>"
            f"<td>{self.scorer.skill_counts.get(s, 0)}</td></tr>"
            for s in SKILL_LIMB_MAP
        )
        bs = m["braid_stats"]
        html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Cognitive Cohesion Braid Report</title>
<style>
body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:2rem;background:#0b1020;color:#e6eefc}}
h1{{color:#7fb3ff}} h2{{color:#a8c8ff;border-bottom:1px solid #234;padding-bottom:.3rem}}
table{{border-collapse:collapse;width:100%;margin:.5rem 0 1.5rem}}
th,td{{padding:.4rem .7rem;border-bottom:1px solid #1d2842;text-align:left}}
th{{background:#162041;color:#9fbfff}}
.kpi{{display:inline-block;background:#142042;border:1px solid #28406b;
     border-radius:8px;padding:.7rem 1rem;margin:.3rem;min-width:160px}}
.kpi b{{display:block;color:#7fb3ff;font-size:1.4rem}}
.bar{{height:10px;background:#28406b;border-radius:5px;overflow:hidden}}
.bar>span{{display:block;height:100%;background:linear-gradient(90deg,#5db,#7af)}}
</style></head><body>
<h1>🧬 Cognitive Cohesion Braid</h1>
<p>Generated {datetime.utcnow().isoformat()}Z · Started {bs['started_at']}</p>

<h2>Cohesion KPIs</h2>
<div class="kpi"><b>{m['cohesion_score']:.3f}</b>cohesion score</div>
<div class="kpi"><b>{m['ewma_score']:.3f}</b>EWMA</div>
<div class="kpi"><b>{m['limb_balance']:.3f}</b>limb balance (entropy)</div>
<div class="kpi"><b>{m['skill_coverage']:.3f}</b>skill coverage</div>
<div class="kpi"><b>{m['latency_score']:.3f}</b>feedback latency score</div>
<div class="bar"><span style="width:{m['cohesion_score']*100:.1f}%"></span></div>

<h2>Braid Cross-Routing</h2>
<div class="kpi"><b>{bs['simula_to_euphan']}</b>SIMULA → EUPHAN</div>
<div class="kpi"><b>{bs['euphan_to_hermes']}</b>EUPHAN → HERMES</div>
<div class="kpi"><b>{bs['hermes_to_simula']}</b>HERMES → SIMULA</div>
<div class="kpi"><b>{bs['total_events']}</b>total events</div>

<h2>Limb Activation ({m['limbs_active']}/{len(ALL_LIMBS)})</h2>
<table><tr><th>Limb</th><th>Activations</th></tr>{rows_limbs}</table>

<h2>Skill Firing ({m['skills_active']}/{len(SKILL_LIMB_MAP)})</h2>
<table><tr><th>Skill</th><th>Source</th><th>Limb</th><th>Fires</th></tr>{rows_skills}</table>

</body></html>"""
        with open(path, "w") as f:
            f.write(html)
        return path


# ───────────────────────────────────────────────────────────────────────────────
# Convenience factory
# ───────────────────────────────────────────────────────────────────────────────

def build_default_braid(output_dir: str = "logs/cohesion") -> CognitiveCohesionBraid:
    """Build a braid with sensible defaults (everything on, no bridges bound)."""
    cfg = CohesionConfig(enabled=True, output_dir=output_dir)
    return CognitiveCohesionBraid(cfg)


# ───────────────────────────────────────────────────────────────────────────────
# Compound-transform reasoning helpers (V30+ ARC pipeline integration)
#
# Two skills bound into the braid above:
#   - scale-ratio-routing      → spatial limb
#   - compound-transform-order → reasoning limb
#
# These functions are pure orchestration utilities the rest of the system
# (compound solvers, router, hand-solver agents) calls. Each call also emits a
# braid event so cohesion_score reflects when these skills fire.
# ───────────────────────────────────────────────────────────────────────────────

INPUT_OP = "INPUT"   # transforms applied to the input domain (where to look)
OUTPUT_OP = "OUTPUT" # transforms applied to the output domain (what to draw)


def classify_scale_ratio(
    in_dims: Tuple[int, int],
    out_dims: Tuple[int, int],
    braid: Optional[CognitiveCohesionBraid] = None,
) -> Dict[str, Any]:
    """Fire scale-ratio-routing skill. Classify input/output dim relationship.

    Returns a dict the router uses to pick a solver family. Inspired by the
    Fibonacci/φ "stable ratio across scales" observation — when in/out dims
    share a clean integer or rational ratio, structural scaling solvers win;
    otherwise route to in-place transform solvers.
    """
    ih, iw = in_dims
    oh, ow = out_dims
    rh = oh / ih if ih else 0.0
    rw = ow / iw if iw else 0.0

    family = "in_place"
    if rh == rw == 1.0:
        family = "in_place"
    elif rh == rw and float(rh).is_integer() and rh > 1:
        family = "scale_up"
    elif rh == rw and rh > 0 and float(1.0 / rh).is_integer():
        family = "scale_down"
    elif rh == rw:
        family = "rational_scale"
    elif (oh % ih == 0 and ow % iw == 0):
        family = "tile_stamp"
    elif (ih % oh == 0 and iw % ow == 0):
        family = "downsample_extract"

    result = {
        "in_dims": (ih, iw),
        "out_dims": (oh, ow),
        "ratio_h": rh,
        "ratio_w": rw,
        "family": family,
    }
    if braid is not None:
        braid.on_euphan_event({
            "skill": "scale-ratio-routing",
            "limb": SKILL_LIMB_MAP["scale-ratio-routing"],
            "action": "classify",
            "confidence": 1.0 if rh == rw else 0.7,
        })
    return result


def order_compound_chain(
    ops: List[Tuple[str, str]],
    braid: Optional[CognitiveCohesionBraid] = None,
) -> List[str]:
    """Fire compound-transform-order skill. Order a list of (op_name, kind) pairs.

    Per the function-composition rule (h(g(f(x)))):
      - INPUT_OP transforms must be applied right-to-left (closest to input fires first)
      - OUTPUT_OP transforms must be applied left-to-right (closest to output fires last)

    This collapses naive O(n!) primitive search to O(k!·m!) where k=#input ops,
    m=#output ops. Returns the ordered op-name sequence to execute.
    """
    input_ops = [name for name, kind in ops if kind == INPUT_OP]
    output_ops = [name for name, kind in ops if kind == OUTPUT_OP]
    ordered = list(reversed(input_ops)) + output_ops
    if braid is not None:
        braid.on_euphan_event({
            "skill": "compound-transform-order",
            "limb": SKILL_LIMB_MAP["compound-transform-order"],
            "action": "order",
            "confidence": 1.0,
        })
    return ordered
