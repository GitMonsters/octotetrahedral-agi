# NGVT Braid — Safety Analysis & Contribution Notes

## What This Is

The NGVT (Nonlinear Geometric Vortexing Torus) Braid system is an **offline analysis
tool** that applies a torus manifold projection and attention-weighting algorithm to
recorded openpilot `modelV2` lead-detection output.

This document describes the safety boundaries and what would be required before any
part of this work could be proposed for inclusion in openpilot.

---

## What Is and Is Not in the Control Path

| Component | Location | Touches control path? |
|---|---|---|
| `ngvt_braid/` (Rust math) | `selfdrive/controls/lib/ngvt_braid/` | **No** — pure math library |
| `tools/ngvt_analysis.py` | `tools/` | **No** — reads `.rlog` files offline |
| `tools/ngvt_visualizer.py` | `tools/` | **No** — plots JSON output files |
| `selfdrive/test/test_ngvt_braid.py` | `selfdrive/test/` | **No** — synthetic unit tests |
| `cereal/custom.capnp.append` | `cereal/` (append-only) | **No** — analysis-only schema |

None of these components publish to any socket consumed by `controlsd`, `plannerd`,
`card`, or any other safety-critical openpilot daemon.

---

## Scope of the Torus Projection

The torus mapping is a coordinate transformation applied **only to logged data**:

```
image-space (x, y) → torus (θ, φ) → R³ coordinates (X, Y, Z)
```

The Braid amplification (3× boost near failure zones) modifies a **score used
for offline analysis only**. It does not feed back into the vehicle's control outputs.

---

## What Would Be Required Before a Live Integration PR

If a future PR proposes wiring `cognitived`/`strategicModelV2` into the control path,
comma.ai's stated priorities require (in order): **Safety → Stability → Quality → Features**.

Before such a PR could be submitted:

1. **No new safety-critical code paths without peer review and CI coverage.**
   Any daemon that publishes to a channel consumed by `controlsd` or `plannerd`
   must pass `process_replay` regression tests against reference logs.

2. **Score amplification must be bounded and validated.**
   The 3× Braid boost must be shown — via process-replay comparison plots — to not
   increase false-positive lead detections or shorten following distance unexpectedly.

3. **Custom cereal schema must not redefine existing fields.**
   `NgvtBraidAnalysis` uses a new struct with a unique ID. It must never shadow or
   redefine fields in `ModelDataV2`, `LeadDataV3`, or any stock struct.

4. **PR size must be under ~500 lines** (comma.ai's stated limit for review).
   Split the Rust crate, the schema addition, and any integration as separate PRs.

5. **Pass CI:**
   ```bash
   python selfdrive/test/process_replay/test_processes.py
   python selfdrive/test/process_replay/model_replay.py
   pytest selfdrive/test/test_ngvt_braid.py -v
   ```

6. **Before/after plots required.**
   Upload `ngvt_results.json` visualizer plots showing the manifold analysis
   on the standard test route (`8494c69d3c710e81|000001d4--2648a9a404`, segment 4).

---

## Recommended Contribution Path

1. Run `tools/ngvt_analysis.py` on public comma.ai log segments.
2. Generate visualizer plots and review for anomalies.
3. Open a **Discussion** (not a PR) on the openpilot GitHub to gauge interest.
4. Check the [bounty board](https://github.com/orgs/commaai/projects/26) for related
   open work that might provide a natural integration point.
5. Join `discord.comma.ai` (#dev channel) and share offline analysis results first.

---

## References

- openpilot CONTRIBUTING.md: `docs/CONTRIBUTING.md`
- openpilot SAFETY.md: `docs/SAFETY.md`
- cereal schema: `cereal/log.capnp` — `ModelDataV2`, `LeadDataV3`
- Process replay tests: `selfdrive/test/process_replay/`
- Standard test route: `8494c69d3c710e81|000001d4--2648a9a404` segment 4
