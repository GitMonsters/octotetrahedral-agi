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
SOLVERS=("rearc_v46_ensemble_voting.py" "rearc_v48_catalog_trained.py" "rearc_v49_compound_enhanced.py")
for f in "${SOLVERS[@]}"; do
  if [ -f "$WORKDIR/$f" ]; then
    echo "   ✅  $f"
  else
    echo "   ❌  MISSING: $f"
  fi
done

# ── 5. Print status ──────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  INSTANCE RESTORED ✨"
echo ""
echo "  Repo:        $WORKDIR"
echo "  Submissions: ~/Desktop/72%/"
echo "  Solver:      python $WORKDIR/rearc_v49_compound_enhanced.py"
echo ""
echo "  RE-ARC Bench: https://arc.markbarney.net/re-arc"
echo "  Upload file:  ~/Desktop/72%/octotetrahedral_rearc_v49_compound_enhanced.json"
echo ""
echo "  GitHub:       https://github.com/GitMonsters/octotetrahedral-agi"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
