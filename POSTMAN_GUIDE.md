# 📮 Postman API Benchmarking Guide

Complete guide to using Postman for benchmarking the OctoTetrahedral AGI API.

---

## 🚀 Quick Start

### 1. Import the Collection

**Method A: Direct Import**
```bash
# Download and open in Postman
curl -o postman_collection.json \
  https://raw.githubusercontent.com/GitMonsters/octotetrahedral-agi/main/postman_collection.json

# Open Postman → Import → Select file
```

**Method B: Via Link**
- Open Postman
- Click **Import**
- Paste: `https://raw.githubusercontent.com/GitMonsters/octotetrahedral-agi/main/postman_collection.json`
- Click **Import**

### 2. Set Environment Variables

In Postman:
1. Click **Environments** (bottom left)
2. Create new environment: `OctoTetrahedral-Dev`
3. Add variables:

```json
{
  "base_url": "http://localhost:8000",
  "api_key": "qU62MH7IOkLzFDUHCVJoRlrc41nzzNNa8-Hhnm2YwVQ"
}
```

### 3. Ensure API is Running

```bash
python3 -m uvicorn api:app --host 0.0.0.0 --port 8000
```

---

## 📊 Available Tests

### **Health & Status** ✅

#### Health Check
- **Endpoint:** `GET /health`
- **Purpose:** Verify system is running, check device (GPU/CPU)
- **Response:** Device type, model name, status
- **Use:** Pre-test verification

#### Get Statistics
- **Endpoint:** `GET /stats`
- **Purpose:** Real-time performance metrics
- **Metrics:** 
  - Total requests
  - Average/min/max latency
  - Throughput (req/sec)
  - Memory usage
  - Error count
- **Use:** Monitor system health

#### Get Metrics (Prometheus)
- **Endpoint:** `GET /metrics`
- **Purpose:** Prometheus-compatible metrics export
- **Format:** Standard Prometheus format
- **Use:** Integration with monitoring tools

---

### **Inference Tests** 🧠

#### Single Inference - Basic
- **Endpoint:** `POST /predict`
- **Input:** `{"input_ids": [1, 2, 3, 4, 5]}`
- **Auth:** Required (Bearer token)
- **Response:** Predictions, latency, device info
- **Use:** Test basic functionality

#### Single Inference - Extended
- **Endpoint:** `POST /predict`
- **Input:** Longer sequence (20 tokens)
- **Purpose:** Test with larger inputs
- **Use:** Stress test single request

#### Batch Processing (10 Requests)
- **Endpoint:** `POST /predict`
- **Purpose:** Sequential batch throughput
- **Use:** Run in Postman Runner (10 iterations)

#### Error Handling Tests
- **Invalid API Key:** Test 401 authentication
- **Missing Auth Header:** Test missing credentials
- **Use:** Security validation

---

### **Performance Benchmarks** 📈

#### Latency Test (Single)
- **Measures:** Single request latency
- **Threshold:** < 5000ms
- **Test Script:** Validates latency range
- **Console Output:** Latency in milliseconds
- **Use:** Measure response time

#### Throughput Test (Concurrent x5)
- **Measures:** Concurrent request handling
- **Method:** Use Postman Runner (5 iterations)
- **Tracking:** Aggregates total and average latency
- **Console Output:** Per-request and running average
- **Use:** Benchmark parallel processing

#### Stress Test (Large Batch)
- **Input Size:** 32 tokens (maximum)
- **Purpose:** Test system limits
- **Use:** Identify breaking point

---

### **Environment Tests** 🌍

#### Check Device (GPU)
- **Purpose:** Verify Metal GPU is active
- **Assertion:** Device contains "mps"
- **Output:** Device type and backend info
- **Use:** Confirm Apple Silicon acceleration

#### Check Memory Usage
- **Purpose:** Monitor memory consumption
- **Threshold:** < 4GB
- **Output:** Memory in MB, uptime, request count
- **Use:** Performance monitoring

#### Performance Summary
- **Purpose:** Comprehensive performance report
- **Output:** Formatted summary table
- **Metrics:** All key performance indicators
- **Use:** Final benchmark results

---

## 🎯 Benchmark Scenarios

### Scenario 1: Quick Health Check (1 min)

```
1. Health Check
2. Get Statistics
3. Single Inference - Basic
```

**Expected Output:**
```
✅ System healthy
✅ Device: mps (Metal GPU)
✅ Latency: ~3750ms (first run)
```

---

### Scenario 2: API Validation (5 min)

```
1. Health Check
2. Single Inference - Basic
3. Invalid API Key (should fail)
4. Missing Auth Header (should fail)
5. Single Inference - Extended
```

**Expected Output:**
```
✅ Valid requests succeed
✅ Invalid requests return 401
✅ Auth validation working
```

---

### Scenario 3: Performance Benchmark (10 min)

```
1. Health Check
2. Check Device
3. Latency Test (1x)
4. Throughput Test (5x in Runner)
5. Stress Test (1x large)
6. Performance Summary
```

**Expected Output:**
```
📊 Performance Summary
═══════════════════════════════════════════════════
⏱️  Uptime: 600s
📈 Total Requests: 8
🔄 Avg Latency: 3245.50ms
📊 Min Latency: 1200.00ms
📈 Max Latency: 3750.00ms
⚡ Throughput: 0.27 req/sec
❌ Errors: 0
💾 Memory: 2876.1 MB
═══════════════════════════════════════════════════
```

---

### Scenario 4: Stress Test (30 min)

```
1. Health Check
2. Throughput Test (50x in Runner - concurrent simulation)
3. Performance Summary
```

**Expected Output:**
```
✅ Handles 50+ sequential requests
✅ Maintains < 5000ms per request
✅ No memory leaks
```

---

## 📋 Running Tests in Postman

### Method 1: Manual Testing

1. Select a request
2. Click **Send**
3. View response in **Body** tab
4. Check **Tests** tab for assertions
5. View **Console** for detailed output

### Method 2: Collection Runner (Batch)

1. Click **Collections** (left sidebar)
2. Right-click collection → **Run collection**
3. Select environment
4. Set iterations (e.g., 10)
5. Configure delay between requests (e.g., 500ms)
6. Click **Run**
7. View results in Summary tab

### Method 3: Automated Runs (CI/CD)

```bash
# Install Newman (Postman CLI)
npm install -g newman

# Run collection
newman run postman_collection.json \
  -e postman_environment.json \
  --reporters cli,json \
  --reporter-json-export results.json

# Run specific folder
newman run postman_collection.json \
  -e postman_environment.json \
  --folder "Performance Benchmarks"
```

---

## 📊 Interpreting Results

### Response Time Analysis

```
Min Latency:     < 1000ms  → Excellent
Avg Latency:     1000-3000ms → Good
Max Latency:     3000-5000ms → Acceptable
> 5000ms         → Needs investigation
```

### Throughput Analysis

```
0.5+ req/sec     → Good
0.2-0.5 req/sec  → Acceptable
< 0.2 req/sec    → Check system
```

### Memory Analysis

```
< 2GB            → Optimal
2-3GB            → Good
3-4GB            → Monitor
> 4GB            → Investigate
```

---

## 🔧 Advanced Features

### Custom Test Scripts

Edit request → **Tests** tab to add assertions:

```javascript
// Example: Check latency
pm.test('Latency under 5s', function() {
    var latency = pm.response.json().latency_ms;
    pm.expect(latency).to.be.below(5000);
});

// Example: Verify predictions
pm.test('Has predictions', function() {
    pm.expect(pm.response.json()).to.have.property('predictions');
});

// Example: Chain variables
pm.collectionVariables.set('last_latency', 
    pm.response.json().latency_ms);
```

### Environment Switching

- **Dev:** `http://localhost:8000` (local)
- **Staging:** `http://staging-api.example.com:8000`
- **Production:** `https://api.octotetrahedral.com`

Switch in top-right dropdown before running tests.

### Monitoring with Console

Open **Postman Console** (Ctrl+Alt+C) to view:
- Request/response details
- Console.log output from test scripts
- Network timing breakdown

---

## 📈 Export Results

### Export Run Summary

1. After running collection, click **Export Results**
2. Choose format: **JSON** or **CSV**
3. Save to file for analysis

### Parse Results Programmatically

```python
import json

with open('results.json') as f:
    data = json.load(f)
    
for run in data['run']['executions']:
    request = run['request']['url']['path']
    response_code = run['response']['code']
    response_time = run['response']['responseTime']
    print(f"{request}: {response_code} ({response_time}ms)")
```

---

## 🚨 Troubleshooting

### Issue: "Could not connect to localhost:8000"

**Solution:**
```bash
# Check if API is running
curl http://localhost:8000/health

# Start API if needed
python3 -m uvicorn api:app --host 0.0.0.0 --port 8000
```

### Issue: "Unauthorized (401)"

**Solution:**
```bash
# Verify API key
# Check environment variable: {{api_key}}
# Generate new key if needed:
./scripts/generate_api_key.sh newkey
```

### Issue: "Request timeout"

**Solution:**
- Increase Postman timeout: Settings → General → Request Timeout (default 0 = infinite)
- Or increase per-request timeout in request settings

### Issue: "Certificate verification failed"

**Solution:**
- For HTTPS only: Settings → General → SSL certificate verification (toggle OFF for dev/testing)

---

## 📚 Next Steps

1. **Run Health Check** - Verify system is up
2. **Run Single Test** - Test basic functionality
3. **Run Collection** - Execute all tests
4. **Export Results** - Save for analysis
5. **Automate** - Use Newman for CI/CD

---

## 🔗 References

- [Postman Documentation](https://learning.postman.com/)
- [Newman CLI](https://learning.postman.com/docs/running-collections/using-newman-cli/command-line-integration-with-newman/)
- [API Reference](./api.py)
- [Commands Reference](./COMMANDS.md)

---

**Last Updated:** 2026-07-24  
**Version:** 1.0.0
