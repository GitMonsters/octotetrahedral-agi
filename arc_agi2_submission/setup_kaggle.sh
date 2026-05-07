#!/bin/bash
# Kaggle API Setup Script
# Run this after downloading kaggle.json from https://www.kaggle.com/settings

echo "=== Kaggle API Setup ==="

# Check if kaggle.json was provided as argument or exists in Downloads
if [ -n "$1" ]; then
    KAGGLE_JSON="$1"
elif [ -f ~/Downloads/kaggle.json ]; then
    KAGGLE_JSON=~/Downloads/kaggle.json
else
    echo "Usage: ./setup_kaggle.sh /path/to/kaggle.json"
    echo ""
    echo "To get kaggle.json:"
    echo "1. Go to https://www.kaggle.com/settings"
    echo "2. Scroll to 'API' section"
    echo "3. Click 'Create New Token'"
    echo "4. Run: ./setup_kaggle.sh ~/Downloads/kaggle.json"
    exit 1
fi

# Setup
mkdir -p ~/.kaggle
cp "$KAGGLE_JSON" ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

echo "✅ Kaggle API configured!"
echo ""
echo "Now submit with:"
echo "  cd /Users/evanpieser/arc_agi2_submission"
echo "  kaggle kernels push"
echo ""
echo "Or direct submit:"
echo "  kaggle competitions submit -c arc-prize-2026-arc-agi-2 -f submission.json -m 'TranscendPlexity'"
