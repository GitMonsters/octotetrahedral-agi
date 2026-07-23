# macOS Production Deployment Guide

This guide covers deploying **OctoTetrahedral AGI** as a production service on macOS with Apple Silicon Metal GPU acceleration.

---

## Prerequisites

| Requirement | Minimum |
|---|---|
| macOS | 12.3 (Monterey) or later |
| Chip | Apple Silicon (M1 / M2 / M3+) |
| Python | 3.9+ |
| Disk | 2 GB free |
| RAM | 8 GB (16 GB recommended) |

---

## Quick Start

```bash
# 1. Clone and enter the repository
git clone https://github.com/GitMonsters/octotetrahedral-agi.git
cd octotetrahedral-agi

# 2. Run the production setup script
bash scripts/setup_macos_production.sh

# 3. Verify the service is healthy
bash scripts/health_check_macos.sh
```

---

## Detailed Setup

### 1. Run the Setup Script

```bash
bash scripts/setup_macos_production.sh [--skip-brew] [--port PORT]
```

The script performs these steps automatically:

1. Verifies macOS 12.3+
2. Detects Apple Silicon architecture
3. Installs Homebrew (skip with `--skip-brew`)
4. Installs Python 3.11 via Homebrew
5. Creates a Python virtual environment at `.venv/`
6. Installs PyTorch with Metal backend and all project dependencies
7. Verifies Metal MPS backend functionality
8. Creates the log directory at `~/Library/Logs/OctoTetrahedralAGI/`
9. Installs the LaunchAgent for auto-start on login
10. Configures the macOS application firewall
11. Validates the installation

### 2. Environment Variables

Override defaults by setting environment variables before running the script or in your shell profile:

| Variable | Default | Description |
|---|---|---|
| `OCTOAGI_ENV` | `prod` | Environment tag |
| `OCTOTETRAHEDRAL_DEVICE` | `auto` | Compute device: `metal`, `cuda`, `cpu`, or `auto` |
| `OCTOAGI_LOG_LEVEL` | `WARNING` | Log verbosity |
| `PORT` | `8000` | API listen port |

### 3. Production Configuration

Edit `config/production.yaml` to tune the service:

```yaml
server:
  port: 8000
  workers: auto      # defaults to physical CPU core count
  timeout: 30

device:
  preference: "metal"
  fallback: "cpu"

rate_limiting:
  enabled: true
  requests_per_minute: 1000
```

---

## Service Management

The LaunchAgent is installed at `~/Library/LaunchAgents/com.octotetrahedral.plist`.

```bash
# Start the service
launchctl start com.octotetrahedral.agi

# Stop the service
launchctl stop com.octotetrahedral.agi

# Check status
launchctl list com.octotetrahedral.agi

# Reload after config changes
launchctl unload ~/Library/LaunchAgents/com.octotetrahedral.plist
launchctl load -w ~/Library/LaunchAgents/com.octotetrahedral.plist
```

### Logs

```bash
# Standard output
tail -f ~/Library/Logs/OctoTetrahedralAGI/stdout.log

# Standard error
tail -f ~/Library/Logs/OctoTetrahedralAGI/stderr.log
```

---

## Health Checks

Run the health check script against the running service:

```bash
bash scripts/health_check_macos.sh [--port PORT] [--host HOST]
```

The script checks:

1. **Endpoint availability** – `/health` is reachable
2. **Health response** – service reports `"status": "healthy"`
3. **Metal / MPS detection** – GPU backend is functional
4. **Model loading** – model is loaded and reported
5. **Inference accuracy** – `/predict` returns a valid response
6. **Latency baseline** – response time is below 2 000 ms
7. **Memory usage** – sufficient free memory available

Exit code `0` = healthy; exit code `1` = one or more checks failed.

---

## Metal GPU Performance

### Expected Throughput

| Device | Avg Latency | Throughput |
|---|---|---|
| CPU (baseline) | ~65 ms | ~16 req/s |
| Metal MPS (M1) | ~8–13 ms | ~80–120 req/s |
| Metal MPS (M2/M3) | ~6–10 ms | ~100–160 req/s |

### Validating Metal Usage

```python
import torch
print(torch.backends.mps.is_available())   # True on supported hardware
print(torch.backends.mps.is_built())       # True when PyTorch was built with MPS
```

Or check the `/health` endpoint response:

```bash
curl -s http://127.0.0.1:8000/health | python3 -m json.tool
```

Expected output includes `"device": "mps"` when Metal is active.

---

## Monitoring Setup

### Prometheus Metrics

The `/metrics` endpoint exposes Prometheus-compatible metrics when `monitoring.metrics_endpoint` is configured. Scrape it with:

```yaml
# prometheus.yml
scrape_configs:
  - job_name: octotetrahedral_agi
    static_configs:
      - targets: ['127.0.0.1:8000']
    metrics_path: /metrics
```

### Error Tracking

All errors are written to `~/Library/Logs/OctoTetrahedralAGI/stderr.log` in JSON format. Pipe to a log aggregator (e.g. Datadog, Splunk) by tailing the file.

---

## Pre-deployment Checklist

Run through this checklist before going live:

- [ ] macOS 12.3+ confirmed (`sw_vers -productVersion`)
- [ ] Apple Silicon detected (`uname -m` → `arm64`)
- [ ] Setup script completed without errors
- [ ] Metal MPS backend verified (`torch.backends.mps.is_available()` → `True`)
- [ ] Virtual environment exists at `.venv/`
- [ ] All dependencies installed (`pip list | grep torch`)
- [ ] Production config reviewed (`config/production.yaml`)
- [ ] Log directory exists (`~/Library/Logs/OctoTetrahedralAGI/`)
- [ ] LaunchAgent installed (`launchctl list com.octotetrahedral.agi`)
- [ ] `/health` returns `"status": "healthy"`
- [ ] `/predict` returns valid predictions
- [ ] Health check script exits 0 (`bash scripts/health_check_macos.sh`)

---

## Troubleshooting

### Service fails to start

```bash
# Check launch agent logs
cat ~/Library/Logs/OctoTetrahedralAGI/stderr.log | tail -50
```

Common causes:
- Port 8000 already in use → change `PORT` in the plist or set `--port`
- Virtual environment missing → re-run `bash scripts/setup_macos_production.sh`
- Missing checkpoint file → set `OCTOAGI_MODEL_PATH` or leave empty for in-memory mode

### Metal MPS not available

Ensure:
- macOS 12.3+ (`sw_vers -productVersion`)
- Apple Silicon chip (`uname -m` → `arm64`)
- PyTorch built with MPS support (`pip show torch` → version ≥ 1.12)

The service automatically falls back to CPU if MPS is unavailable.

### Permission errors on launchctl

```bash
# Ensure the plist is owned by your user
ls -la ~/Library/LaunchAgents/com.octotetrahedral.plist
# Should be owned by you, mode 644
chmod 644 ~/Library/LaunchAgents/com.octotetrahedral.plist
```

### Verifying the setup script completed successfully

```bash
# Check that the venv and key packages exist
.venv/bin/python -c "import torch; print(torch.__version__)"
.venv/bin/python -c "import production_config; print('OK')"
```

---

## Performance Validation

After deployment, run a quick performance baseline:

```bash
# Single request
curl -w "\nTime: %{time_total}s\n" -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"input_ids": [1,2,3,4,5,6,7,8]}' | python3 -m json.tool

# 10-request throughput test
for i in $(seq 1 10); do
  curl -s -o /dev/null -w "%{time_total}\n" \
    -X POST http://127.0.0.1:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"input_ids": [1,2,3,4,5,6,7,8]}'
done
```

Latency under 100 ms per request on Apple Silicon with Metal indicates a healthy production configuration.
