#!/usr/bin/env bash
# ============================================================
#  AUTONOMOUS INSTANCE RESTORE SCRIPT
#  OctoTetrahedral AGI — RE-ARC Bench Solver Pipeline
#  Run this after any device reset to get back to full state.
# ============================================================

set -e
REPO="git@github.com:GitMonsters/octotetrahedral-agi.git"
WORKDIR="$HOME"

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║       AUTONOMOUS INSTANCE RESTORE — OctoTetrahedral     ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ── 0. SSH: ensure port 443 fallback is configured ──────────
# GitHub blocks port 22 in some networks; 443 always works.
if ! grep -q "ssh.github.com" "$HOME/.ssh/config" 2>/dev/null; then
  echo "🔑  Configuring SSH port-443 fallback for GitHub..."
  mkdir -p "$HOME/.ssh"
  cat >> "$HOME/.ssh/config" << 'SSHEOF'

Host github.com
  Hostname ssh.github.com
  Port 443
  User git
SSHEOF
  echo "   Done."
else
  echo "🔑  SSH config already has port-443 fallback."
fi

# ── 1. Clone / pull repo ────────────────────────────────────
if [ -d "$WORKDIR/.git" ]; then
  echo "✅  Repo already present — pulling latest..."
  cd "$WORKDIR" && git pull origin main
else
  echo "📦  Cloning repo into $WORKDIR ..."
  git clone "$REPO" "$WORKDIR"
  cd "$WORKDIR"
fi

# ── 2. Python env ───────────────────────────────────────────
echo ""
echo "🐍  Setting up Python environment..."
if [ ! -d "$WORKDIR/venv" ]; then
  python3 -m venv "$WORKDIR/venv"
fi
source "$WORKDIR/venv/bin/activate"
pip install --quiet --upgrade pip
if [ -f "$WORKDIR/requirements.txt" ]; then
  pip install --quiet -r "$WORKDIR/requirements.txt"
fi
# scipy needed by v50 rule learner
pip install --quiet scipy numpy 2>/dev/null || true

# ── 3. Restore Desktop/72% folder (submission files) ────────
echo ""
echo "📁  Restoring submission folder..."
mkdir -p "$HOME/Desktop/72%"
if [ -d "$WORKDIR/arc_agi2_submission/submissions" ]; then
  cp -n "$WORKDIR/arc_agi2_submission/submissions/"*.json "$HOME/Desktop/72%/" 2>/dev/null || true
  echo "   Submissions restored to ~/Desktop/72%/"
fi

# ── 4. Quick validation ──────────────────────────────────────
echo ""
echo "🔍  Validating key files..."
SOLVERS=(
  "arc_agi2_submission/rearc_v46_ensemble_voting.py"
  "arc_agi2_submission/rearc_v48_catalog_trained.py"
  "arc_agi2_submission/rearc_v49_compound_enhanced.py"
  "arc_agi2_submission/rearc_v50_rule_learner.py"
)
for f in "${SOLVERS[@]}"; do
  if [ -f "$WORKDIR/$f" ]; then
    echo "   ✅  $f"
  else
    echo "   ❌  MISSING: $f"
  fi
done

SUBS=(
  "octotetrahedral_rearc_v46_ensemble_voting.json"
  "octotetrahedral_rearc_v49_compound_enhanced.json"
  "octotetrahedral_rearc_v50_rule_learner.json"
)
echo ""
echo "   Submissions in ~/Desktop/72%/:"
for s in "${SUBS[@]}"; do
  if [ -f "$HOME/Desktop/72%/$s" ]; then
    echo "   ✅  $s"
  else
    echo "   ⚠️   $s (not yet generated — run solver)"
  fi
done

# ── 5. Print status ──────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  INSTANCE RESTORED ✨"
echo ""
echo "  Repo:        $WORKDIR"
echo "  Submissions: ~/Desktop/72%/"
echo ""
echo "  ⭐  BEST SOLVER (v50 — per-task rule learner):"
echo "      cd $WORKDIR && python arc_agi2_submission/rearc_v50_rule_learner.py"
echo "      Upload: ~/Desktop/72%/octotetrahedral_rearc_v50_rule_learner.json"
echo ""
echo "  📤  RE-ARC Bench: https://arc.markbarney.net/re-arc"
echo "  🐙  GitHub:       https://github.com/GitMonsters/octotetrahedral-agi"
echo ""
echo "  Solver chain: v46 → v47 → v48 → v49 → v50 (current best)"
echo "  Next:         build v51 based on v50 benchmark score"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
