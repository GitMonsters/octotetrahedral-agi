# 🚀 OctoTetrahedral AGI - Quick Start Guide

**Your API is already running!** Just run benchmarks in 3 steps.

---

## ⚡ Quick Start (Copy & Paste)

### Step 1: Activate Virtual Environment
```bash
source octo_env/bin/activate
```

### Step 2: Install Benchmark Dependencies
```bash
pip install -r requirements_benchmark.txt
```

### Step 3: Run Quick Benchmark
```bash
python quick_benchmark.py
```

**That's it!** You'll see results like:

```
======================================================================
🚀 OctoTetrahedral AGI - Quick Benchmark
======================================================================

1️⃣  Health Check...
   ✅ Device: mps
   ✅ Status: healthy

2️⃣  Single Inference (5 requests)...
   Request 1: 3758ms ✅
   Request 2: 1452ms ✅
   Request 3: 1451ms ✅
   Request 4: 1449ms ✅
   Request 5: 1448ms ✅

   📊 Average: 1911ms
   📊 Min: 1448ms
   📊 Max: 3758ms

3️⃣  System Statistics...
   💾 Memory: 2876.1MB
   📈 Total Requests: 5
   ⏱️  Uptime: 45s
   ⚡ Throughput: 0.11 req/sec

======================================================================
✅ Benchmark Complete!
======================================================================
```

---

## 📊 Advanced Benchmarking

### Option A: Postman GUI (Visual)
```bash
# Download & import in Postman
# Files: postman_collection.json + postman_environment.json
# Then click "Run Collection"
```

### Option B: Newman CLI (Automated)
```bash
# Install Newman reporter
npm install -g newman-reporter-html

# Run benchmarks
newman run postman_collection.json \
  -e postman_environment.json \
  --reporters cli,json,html
```

### Option C: Comprehensive Python Suite
```bash
# Full benchmark suite (if benchmark_suite.py exists)
python benchmark_suite.py --all --html
```

---

## 📈 What You Get

| Command | Purpose | Time |
|---------|---------|------|
| `python quick_benchmark.py` | Quick health check | 1 min |
| `newman run postman_collection.json` | Full API test | 5 min |
| `python benchmark_suite.py --all` | Comprehensive suite | 15 min |

---

## 🔧 API Endpoints

The API is running at `http://localhost:8000`

### Health Check
```bash
curl http://localhost:8000/health
```

### Statistics
```bash
curl http://localhost:8000/stats
```

### Run Inference
```bash
curl -X POST http://localhost:8000/predict \
  -H "Authorization: Bearer qU62MH7IOkLzFDUHCVJoRlrc41nzzNNa8-Hhnm2YwVQ" \
  -H "Content-Type: application/json" \
  -d '{"input_ids": [1, 2, 3, 4, 5]}'
```

---

## 📁 Files Reference

```
octotetrahedral-agi/
├── quick_benchmark.py              ← Run this for quick tests
├── postman_collection.json         ← Import in Postman
├── postman_environment.json        ← Import in Postman
├── POSTMAN_GUIDE.md                ← Full Postman guide
├── COMMANDS.md                     ← All available commands
├── scripts/
│   ├── quickstart.sh               ← Auto setup script
│   └── run_postman_benchmarks.sh   ← Auto run benchmarks
└── octo_env/                       ← Virtual environment (auto-created)
```

---

## 🎯 Common Tasks

### Just Test the API
```bash
source octo_env/bin/activate
python quick_benchmark.py
```

### Test in Postman (visual)
1. Open Postman
2. File → Import → `postman_collection.json`
3. File → Import → `postman_environment.json`
4. Click Collections → Run Collection

### Get detailed stats
```bash
curl http://localhost:8000/stats | python -m json.tool
```

### Check GPU is active
```bash
curl http://localhost:8000/health | python -m json.tool
# Look for: "device": "mps"
```

---

## ✅ Status

```
API Server:        ✅ Running at http://localhost:8000
Metal GPU:         ✅ Active (mps device)
Virtual Env:       ✅ Created (octo_env/)
Dependencies:      ✅ Installed
Quick Benchmark:   ✅ Ready to run
Postman Suite:     ✅ Ready to import
```

---

## 🚀 Next Steps

1. **Run quick benchmark:**
   ```bash
   source octo_env/bin/activate
   python quick_benchmark.py
   ```

2. **Or use Postman for visual testing:**
   - Open Postman
   - Import both JSON files
   - Click "Run Collection"

3. **View full documentation:**
   - Read `POSTMAN_GUIDE.md` for details
   - Read `COMMANDS.md` for all available commands

---

## 📞 Troubleshooting

**API not responding?**
```bash
curl http://localhost:8000/health
```

**Virtual environment not activated?**
```bash
source octo_env/bin/activate
```

**Missing dependencies?**
```bash
pip install -r requirements.txt
pip install -r requirements_benchmark.txt
```

**Port 8000 already in use?**
```bash
# Kill existing process
pkill -f "uvicorn api:app"

# Restart API
python3 -m uvicorn api:app --host 0.0.0.0 --port 8000
```

---

**Ready?** Run this now:
```bash
source octo_env/bin/activate && python quick_benchmark.py
```

🎉 **That's it!**
