# 🚀 OctoTetrahedral AGI — Command Reference

Complete guide to all commands for the OctoTetrahedral AGI system.

---

## 🏃 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start the API server (Metal GPU acceleration)
python3 -m uvicorn api:app --host 0.0.0.0 --port 8000

# OR enable auto-start via LaunchAgent
launchctl load ~/Library/LaunchAgents/com.octotetrahedral.plist
```

---

## 📊 API Endpoints

### Health & Monitoring

```bash
# Health check with device info
curl http://localhost:8000/health

# Performance statistics
curl http://localhost:8000/stats

# Prometheus metrics
curl http://localhost:8000/metrics
```

### Inference (Requires API Key)

```bash
# Single inference
curl -X POST http://localhost:8000/predict \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input_ids": [1, 2, 3, 4, 5]}'
```

---

## 🔑 API Key Management

```bash
# Generate new API key
python3 -c "from auth import generate_api_key; print(generate_api_key('label'))"

# Or use the script
./scripts/generate_api_key.sh [label]

# View all API keys
cat ~/.octotetrahedral/api_keys.json
```

---

## 🍎 Metal GPU Setup

```bash
# Enable Apple Silicon Metal (MPS) GPU acceleration
./scripts/enable_metal.sh

# Verify Metal is enabled
python3 -c "import torch; print(torch.backends.mps.is_available())"
```

---

## 🧠 Model Training

### Resume from checkpoint

```bash
# Continue training from step 2500
python train_arc.py \
    --resume checkpoints/arc/arc_step_2500.pt \
    --max-steps 7500 \
    --batch-size 8

# Full 60 epoch training
python train_arc.py \
    --resume checkpoints/arc/arc_step_2500.pt \
    --max-steps 7500 \
    --batch-size 16
```

### New training run

```bash
python train_arc.py \
    --data-dir /path/to/ARC-AGI/data \
    --batch-size 16 \
    --learning-rate 1e-4 \
    --max-steps 7500
```

---

## 📊 Benchmarking

### Full Benchmark Suite

```bash
# Install benchmark dependencies
pip install -r requirements_benchmark.txt

# Run all benchmarks
python benchmark_suite.py --all

# Benchmark specific models
python benchmark_suite.py \
  --models octotetrahedral,claude,chatgpt \
  --tasks single_inference,batch_10,reasoning

# Generate HTML report
python benchmark_suite.py \
  --models octotetrahedral,llama \
  --tasks single_inference,batch_10,concurrent_10 \
  --html \
  --output benchmark_results/
```

### Public Benchmarks (No API Keys)

```bash
# Run benchmarks using public/free models
python benchmark_public.py

# Output files: benchmark_public_YYYYMMDD_HHMMSS.{json,csv,md,html}
```

### Quick Performance Test

```bash
# Run rapid performance tests
python test_performance.py [API_KEY]

# With default key
python test_performance.py

# Tests: latency, throughput, concurrency, health, stats
```

---

## 🧩 ARC Prize Solver

### Solve Specific Tasks

```bash
# Solve task a32d8b75
python arc_task_a32d8b75_solver.py

# Run with visualization
python arc_task_a32d8b75_solver.py --visualize

# Save results
python arc_task_a32d8b75_solver.py --output results.json
```

### View Puzzle Catalog

```bash
# List all 514 available solvers
ls arc-puzzle-catalog/solves/

# Run specific solver
python3 -c "
import json, importlib.util

task_id = '0934a4d8'
with open(f'arc-puzzle-catalog/dataset/{task_id}.json') as f:
    task = json.load(f)

spec = importlib.util.spec_from_file_location('solver', f'arc-puzzle-catalog/solves/{task_id}/solver.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

for pair in task['test']:
    result = mod.solve(pair['input'])
    print(f'{task_id}: PASS' if result == pair['output'] else 'FAIL')
"

# Generate visualizations
python arc-puzzle-catalog/generate_viz.py

# View catalog in browser
open arc-puzzle-catalog/index.html
```

---

## 📈 RE-ARC Submission Pipeline

```bash
# Run RE-ARC v50 Rule Learner (BEST)
python arc_agi2_submission/rearc_v50_rule_learner.py

# Output: octotetrahedral_rearc_v50_rule_learner.json

# Check other versions
ls arc_agi2_submission/submissions/

# Restore autonomous instance after device reset
bash arc_agi2_submission/RESTORE_INSTANCE.sh
```

---

## 🔧 Service Management

### LaunchAgent (Auto-Start)

```bash
# Load service (auto-start enabled)
launchctl load ~/Library/LaunchAgents/com.octotetrahedral.plist

# Unload service (disable auto-start)
launchctl unload ~/Library/LaunchAgents/com.octotetrahedral.plist

# Restart service
launchctl unload ~/Library/LaunchAgents/com.octotetrahedral.plist
sleep 1
launchctl load ~/Library/LaunchAgents/com.octotetrahedral.plist

# View service status
launchctl list | grep octotetrahedral

# View service logs
tail -f ~/Library/Logs/octotetrahedral.log
```

### Manual Service Control

```bash
# Kill running processes
pkill -f "uvicorn api:app"

# Start fresh
pkill -f "uvicorn api:app"
sleep 1
python3 -m uvicorn api:app --host 0.0.0.0 --port 8000

# Start on different port
python3 -m uvicorn api:app --host 0.0.0.0 --port 8001
```

---

## 📝 Logging & Debugging

```bash
# View application logs
tail -f ~/Library/Logs/octotetrahedral.log

# Check system resources
ps aux | grep uvicorn

# Monitor memory usage
python3 -c "import psutil; print(psutil.Process().memory_info())"

# Test API connectivity
curl -v http://localhost:8000/health

# Check GPU status
python3 -c "import torch; print(f'Device: {torch.device(\"mps\" if torch.backends.mps.is_available() else \"cpu\")}')"
```

---

## 🏗️ Development

### Code Structure

```bash
# Main files
api.py                  # FastAPI server
model.py                # OctoTetrahedral model
auth.py                 # API authentication
monitoring.py           # Performance tracking
config.py               # Configuration
cognition.py            # AGI cognition module

# Directories
core/                   # Tetrahedral attention
limbs/                  # 8 specialized limbs
adaptation/             # RNA editing & LoRA
sync/                   # Hub synchronization
data/                   # Dataset loaders
arc-puzzle-catalog/     # 514 ARC solvers
arc_agi2_submission/    # RE-ARC pipeline
```

### Testing

```bash
# Run integration tests
python test_performance.py

# Test ARC solvers
python test_arc_solver.py

# Run all tests
pytest

# Test specific module
pytest test_performance.py -v
```

---

## 📦 Dependencies

### Core Requirements

```bash
pip install -r requirements.txt
```

### Benchmarking

```bash
pip install -r requirements_benchmark.txt
```

### Optional (for specific features)

```bash
# Anthropic Claude
pip install anthropic

# OpenAI GPT
pip install openai

# Google Gemini
pip install google-generativeai

# Mistral
pip install mistralai
```

---

## 🌐 Environment Variables

```bash
# API Keys
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
export GOOGLE_API_KEY="..."
export MISTRAL_API_KEY="..."

# OctoTetrahedral
export OCTO_API_KEY="qU62MH7IOkLzFDUHCVJoRlrc41nzzNNa8-Hhnm2YwVQ"

# Model paths
export MODEL_PATH="checkpoints/arc/arc_final.pt"
export DATA_DIR="/path/to/ARC-AGI/data"

# Server
export SERVER_HOST="0.0.0.0"
export SERVER_PORT="8000"
```

---

## 📊 Performance Targets

| Component | Target | Current |
|-----------|--------|---------|
| Single Inference | <100ms | 3.7s (first run) |
| Throughput | 100+ req/s | 270 req/s |
| Memory Usage | <4GB | 2.9GB |
| Metal GPU | 5-10x speedup | ✅ Enabled |
| API Key Auth | All endpoints | ✅ Implemented |

---

## 🚀 Production Deployment

```bash
# Docker image
docker build -t octotetrahedral-agi .
docker run -p 8000:8000 octotetrahedral-agi

# Kubernetes
kubectl apply -f k8s/deployment.yaml

# Cloud (AWS/GCP/Azure)
# See deployment guides in docs/

# GitHub Actions CI/CD
# See .github/workflows/
```

---

## 📞 Support & Troubleshooting

### Common Issues

```bash
# Port already in use
pkill -f "uvicorn api:app"
sleep 2
python3 -m uvicorn api:app --host 0.0.0.0 --port 8000

# Model not loading
python3 -c "import torch; from model import OctoTetrahedralModel; m = OctoTetrahedralModel()"

# Metal GPU not detected
./scripts/enable_metal.sh

# API key invalid
./scripts/generate_api_key.sh mykey

# Benchmark fails
pip install -r requirements_benchmark.txt
```

---

## 📚 Additional Resources

- [ARC Prize](https://arcprize.org)
- [GitHub Repository](https://github.com/GitMonsters/octotetrahedral-agi)
- [Benchmarking Guide](./BENCHMARKING.md)
- [Architecture Details](./ARCHITECTURE.md)
- [Contributing Guide](./CONTRIBUTING.md)

---

**Last Updated:** 2026-07-24  
**Version:** 1.0.0
