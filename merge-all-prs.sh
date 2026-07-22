#!/bin/bash

# Merge all open PRs in GitMonsters/octotetrahedral-agi
# Usage: bash merge-all-prs.sh

REPO="GitMonsters/octotetrahedral-agi"
PR_NUMBERS=(25 23 22 19 18 16 15 14 11 5 2)

echo "🚀 Starting merge of ${#PR_NUMBERS[@]} pull requests..."
echo ""

merged=0
failed=0

for pr_num in "${PR_NUMBERS[@]}"; do
  echo "📋 Merging PR #$pr_num..."
  
  if gh pr merge "$pr_num" --repo "$REPO" --merge; then
    echo "✅ PR #$pr_num merged successfully"
    ((merged++))
  else
    echo "❌ Failed to merge PR #$pr_num"
    ((failed++))
  fi
  echo ""
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 Summary:"
echo "   ✅ Merged: $merged"
echo "   ❌ Failed: $failed"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
