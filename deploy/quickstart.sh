#!/bin/bash
# ================================================================
# OctoTetrahedral AGI — GPU Deploy Quick Reference
#
# Skip IONOS. Three faster options:
# ================================================================

cat << 'EOF'
╔══════════════════════════════════════════════════════════════╗
║  OctoTetrahedral AGI — GPU Deployment (Non-IONOS)           ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  OPTION 1: Google Colab (FREE, 30 seconds)                   ║
║  ─────────────────────────────────────────                   ║
║  1. Open deploy/OctoTetrahedral_MultiModal_Demo.ipynb        ║
║  2. Upload to https://colab.research.google.com              ║
║  3. Runtime → Change runtime type → T4 GPU                   ║
║  4. Runtime → Run all                                        ║
║  Done. Free T4 GPU, zero API keys.                           ║
║                                                              ║
║  OPTION 2: Lambda Labs ($2.49/hr H100, ~5 min setup)         ║
║  ────────────────────────────────────────────────             ║
║  1. Sign up: https://lambda.ai                               ║
║  2. Get API key: https://cloud.lambdalabs.com/api-keys       ║
║  3. Run:                                                     ║
║     export LAMBDA_API_KEY="your-key"                         ║
║     bash deploy/provision_lambda.sh                          ║
║                                                              ║
║  OPTION 3: RunPod ($1.99/hr H100, ~5 min setup)              ║
║  ───────────────────────────────────────────────              ║
║  1. Sign up: https://runpod.io                               ║
║  2. Get API key: Settings → API Keys                         ║
║  3. Run:                                                     ║
║     export RUNPOD_API_KEY="your-key"                         ║
║     bash deploy/provision_runpod.sh                          ║
║                                                              ║
╠══════════════════════════════════════════════════════════════╣
║  After any GPU is running, deploy the model:                 ║
║     ssh root@<IP> 'bash -s' < deploy/ionos_setup.sh          ║
║  Or manually:                                                ║
║     ssh root@<IP>                                            ║
║     git clone https://github.com/GitMonsters/                ║
║       octotetrahedral-agi.git /opt/octo && cd /opt/octo      ║
║     pip install -r requirements.txt                          ║
║     python serve.py --scale tiny --device cuda:0             ║
╚══════════════════════════════════════════════════════════════╝
EOF
