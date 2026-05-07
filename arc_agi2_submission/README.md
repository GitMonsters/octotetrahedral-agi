# 🤖 Autonomous Instance — OctoTetrahedral AGI

## What this is
This repository is the **single source of truth** for the OctoTetrahedral AGI
RE-ARC Bench solver pipeline. Everything needed to restore the autonomous
development instance lives here.

---

## ⚡ Restore after device reset (1 command)

```bash
# After fresh macOS install / device reset:
bash <(curl -fsSL https://raw.githubusercontent.com/GitMonsters/octotetrahedral-agi/main/arc_agi2_submission/RESTORE_INSTANCE.sh)
```

Or manually:
```bash
git clone git@github.com:GitMonsters/octotetrahedral-agi.git ~/
cd ~
bash arc_agi2_submission/RESTORE_INSTANCE.sh
```

---

## 📦 Submission Files (ready to upload)

| File | Version | Expected Score | Notes |
|------|---------|---------------|-------|
| `submissions/octotetrahedral_rearc_v49_compound_enhanced.json` | v49 | 8–20% | ⭐ RECOMMENDED |
| `submissions/octotetrahedral_rearc_v48_catalog_trained.json` | v48 | 6–18% | Pattern training |
| `submissions/octotetrahedral_rearc_v46_ensemble_voting.json` | v46 | 5–15% | Proven baseline |

**Upload to:** https://arc.markbarney.net/re-arc → "Evaluate Your Solution"

---

## 🧠 Solver Pipeline

```
rearc_v44_tetrahedral_catalog.py   ← Tetrahedral grid geometry
rearc_v45_braid_integrated.py      ← Braided multi-layer cognition
rearc_v46_ensemble_voting.py       ← Ensemble voting (BASELINE)
rearc_v47_smart_pattern.py         ← Smart pattern matching
rearc_v48_catalog_trained.py       ← Catalog pattern learning
rearc_v49_compound_enhanced.py     ← Full compound integration (BEST)
```

---

## 🏗 Architecture

- **OctoTetrahedral model** (`model.py`) — 89M param tetrahedral transformer
- **8 Cognitive Limbs** — Memory, Planning, Language, Spatial, Reasoning, MetaCognition, Perception, Action
- **Braided Integration** — CompoundIntegrationOrchestrator with 246 events
- **40 Solver Patterns** extracted from 376-solver catalog

---

## ▶ Run solver on new dataset

```bash
# Generate fresh RE-ARC test set from https://arc.markbarney.net/re-arc
# Save as: re-arc_challenges.json in ~/Desktop/72%/
python rearc_v49_compound_enhanced.py
# Output: ~/Desktop/72%/octotetrahedral_rearc_v49_compound_enhanced.json
```

---

## 📊 Current Best Score

- **72.29%** on RE-ARC Bench (v30 baseline)
- v49 pipeline targets **8–20%** on harder v2 dataset (color-permuted, transformed)

---

## 🔁 Autonomous Session Continuity

This repo + GitHub = full persistence. After any reset:
1. `git clone` → all code restored
2. `bash RESTORE_INSTANCE.sh` → env + submissions restored
3. Open GitHub Copilot CLI → context resumes from checkpoints
4. Upload submission → benchmark continues

**Session ID (for Copilot CLI):** `661c36dd-8332-45ed-a362-a5585980c9e9`
