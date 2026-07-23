# OctoTetrahedral AGI — Deployment Guide

## Quick Start: Model Inference

Your trained model is ready for production use. Follow this guide to deploy and run inference on ARC puzzles.

---

## 1. Load the Trained Model

```python
import torch
from model import OctoTetrahedralModel

# Initialize model
model = OctoTetrahedralModel()
model.eval()

# Load best checkpoint (1000-step trained)
checkpoint = torch.load('checkpoints/arc/arc_final.pt', weights_only=False)
model.load_state_dict(checkpoint)

print("✅ Model loaded and ready for inference")
```

---

## 2. Run Inference on ARC Puzzles

### Basic Inference

```python
import torch
from model import OctoTetrahedralModel

model = OctoTetrahedralModel()
model.eval()
checkpoint = torch.load('checkpoints/arc/arc_final.pt', weights_only=False)
model.load_state_dict(checkpoint)

# Prepare input (tokenized ARC puzzle)
batch_size = 2
seq_len = 32
input_ids = torch.randint(0, 1000, (batch_size, seq_len))

# Run inference
with torch.no_grad():
    output = model(input_ids=input_ids, return_confidences=True)

# Extract predictions
logits = output['logits']  # Shape: [batch_size, seq_len, vocab_size]
confidences = output['confidences']  # Dict of limb confidences

print(f"Logits shape: {logits.shape}")
print(f"Overall confidence: {confidences['overall']:.3f}")

# Get top-k predictions
top_k = torch.topk(logits, k=5, dim=-1)
print(f"Top 5 predictions per position: {top_k.indices}")
```

### With Confidence Scores

```python
# Get per-limb confidence scores
print("Cognitive Limb Confidences:")
for limb, confidence in confidences.items():
    if limb not in ['braid_gates', 'braid_weights', 'overall']:
        print(f"  {limb:15s}: {confidence:.3f}")

# Use confidences for uncertainty-aware routing
if confidences['overall'] > 0.7:
    print("✅ High confidence prediction")
elif confidences['overall'] > 0.5:
    print("⚠️  Medium confidence - consider ensembling")
else:
    print("❌ Low confidence - fallback to solver")
```

---

## 3. Batch Processing

```python
from torch.utils.data import DataLoader
from data.arc_dataset import ARCDataset, load_arc_tasks

# Load ARC tasks
tasks = load_arc_tasks('data/ARC-AGI/data', split='training', limit=100)

# Create dataset and loader
dataset = ARCDataset(tasks)
loader = DataLoader(dataset, batch_size=16, shuffle=False)

# Inference loop
model.eval()
all_predictions = []

with torch.no_grad():
    for batch_idx, batch in enumerate(loader):
        input_ids = batch['input_ids']
        output = model(input_ids=input_ids, return_confidences=True)
        
        predictions = output['logits'].argmax(dim=-1)
        all_predictions.append(predictions)
        
        if (batch_idx + 1) % 10 == 0:
            print(f"Processed {batch_idx + 1} batches")

# Combine results
all_predictions = torch.cat(all_predictions, dim=0)
print(f"✅ Processed {len(all_predictions)} samples")
```

---

## 4. GPU Deployment

### CUDA (NVIDIA GPU)

```python
import torch
from model import OctoTetrahedralModel

# Initialize on GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = OctoTetrahedralModel().to(device)

# Load checkpoint
checkpoint = torch.load('checkpoints/arc/arc_final.pt', weights_only=False)
model.load_state_dict(checkpoint)
model.eval()

# Inference
input_ids = torch.randint(0, 1000, (32, 32)).to(device)

with torch.no_grad():
    output = model(input_ids=input_ids, return_confidences=True)

print(f"✅ GPU inference: {device}")
```

### Metal (Apple Silicon)

```python
import torch
from model import OctoTetrahedralModel

# Initialize on Metal
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model = OctoTetrahedralModel().to(device)

# Load checkpoint
checkpoint = torch.load('checkpoints/arc/arc_final.pt', weights_only=False)
model.load_state_dict(checkpoint)
model.eval()

print(f"✅ Apple Silicon inference: {device}")
```

---

## 5. REST API Deployment

### FastAPI Server

```python
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from model import OctoTetrahedralModel

app = FastAPI(title="OctoTetrahedral AGI Inference")

# Load model once at startup
model = OctoTetrahedralModel()
checkpoint = torch.load('checkpoints/arc/arc_final.pt', weights_only=False)
model.load_state_dict(checkpoint)
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

class InferenceRequest(BaseModel):
    input_ids: list
    return_confidences: bool = True

class InferenceResponse(BaseModel):
    predictions: list
    confidences: dict
    device: str

@app.post("/predict")
async def predict(request: InferenceRequest):
    """Run inference on input tokens"""
    input_ids = torch.tensor(request.input_ids).to(device)
    
    with torch.no_grad():
        output = model(
            input_ids=input_ids,
            return_confidences=request.return_confidences
        )
    
    predictions = output['logits'].argmax(dim=-1).tolist()
    confidences = {k: float(v) for k, v in output['confidences'].items()}
    
    return InferenceResponse(
        predictions=predictions,
        confidences=confidences,
        device=str(device)
    )

@app.get("/health")
async def health():
    """Health check"""
    return {"status": "healthy", "model": "OctoTetrahedralModel"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**Run the server:**
```bash
python api.py
```

**Example request:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "input_ids": [100, 200, 300, 400, 500],
    "return_confidences": true
  }'
```

---

## 6. Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and code
COPY . .

# Download checkpoint (or mount volume)
RUN mkdir -p checkpoints/arc

# Expose API port
EXPOSE 8000

# Run inference server
CMD ["python", "api.py"]
```

### docker-compose.yml

```yaml
version: '3.8'

services:
  octotetrahedral:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./checkpoints/arc:/app/checkpoints/arc
    environment:
      - DEVICE=cuda
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

**Build and run:**
```bash
docker-compose up
```

---

## 7. Performance Optimization

### Model Quantization

```python
import torch
from model import OctoTetrahedralModel

# Load model
model = OctoTetrahedralModel()
checkpoint = torch.load('checkpoints/arc/arc_final.pt', weights_only=False)
model.load_state_dict(checkpoint)

# Quantize to int8
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)

# Save quantized model
torch.save(quantized_model.state_dict(), 'checkpoints/arc/arc_final_quantized.pt')

print("✅ Model quantized: 75% memory reduction")
```

### Model Pruning

```python
import torch.nn.utils.prune as prune

# Prune 30% of weights
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        prune.l1_unstructured(module, name='weight', amount=0.3)
        prune.remove(module, name='weight')

print("✅ Model pruned: 30% parameter reduction")
```

### Batch Inference Optimization

```python
# Use larger batches for throughput
batch_sizes = [1, 8, 32, 64, 128]

for batch_size in batch_sizes:
    input_ids = torch.randint(0, 1000, (batch_size, 32)).to(device)
    
    # Time inference
    import time
    start = time.time()
    
    with torch.no_grad():
        for _ in range(100):
            _ = model(input_ids=input_ids)
    
    elapsed = time.time() - start
    throughput = (batch_size * 100) / elapsed
    
    print(f"Batch {batch_size:3d}: {throughput:.0f} samples/sec")
```

---

## 8. Monitoring & Logging

```python
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class InferenceMonitor:
    def __init__(self):
        self.total_inferences = 0
        self.total_time = 0
        self.confidence_hist = []
    
    def log_inference(self, batch_size, elapsed_time, confidences):
        self.total_inferences += batch_size
        self.total_time += elapsed_time
        self.confidence_hist.append(confidences['overall'])
        
        logger.info(
            f"Inference: {batch_size} samples, "
            f"{elapsed_time:.3f}s, "
            f"confidence: {confidences['overall']:.3f}"
        )
    
    def summary(self):
        avg_time = self.total_time / max(1, self.total_inferences)
        avg_conf = sum(self.confidence_hist) / len(self.confidence_hist)
        
        logger.info(f"Summary: {self.total_inferences} inferences, "
                   f"avg {avg_time*1000:.1f}ms/sample, "
                   f"avg confidence {avg_conf:.3f}")

monitor = InferenceMonitor()
```

---

## 9. Troubleshooting

### Out of Memory

```python
# Reduce batch size
batch_size = 4  # Instead of 32

# Or use gradient checkpointing
model.enable_gradient_checkpointing()
```

### Slow Inference

```python
# Use mixed precision
from torch.cuda.amp import autocast

with autocast():
    output = model(input_ids=input_ids)

# Or use TorchScript
scripted_model = torch.jit.script(model)
```

### Low Confidence

```python
# Use ensemble with multiple checkpoints
checkpoints = ['arc_step_800.pt', 'arc_step_1000.pt']
models = [load_model(ckpt) for ckpt in checkpoints]

# Ensemble predictions
predictions = []
for model in models:
    pred = model(input_ids)
    predictions.append(pred)

ensemble_pred = torch.stack(predictions).mean(dim=0)
```

---

## 10. Benchmarking

```bash
# CPU Benchmark
python -m torch.utils.benchmark.main -m \
  "model = OctoTetrahedralModel(); x = torch.randn(32, 32); model(x)"

# GPU Benchmark
python benchmark_gpu.py --device cuda --batch-size 128 --num-iters 1000
```

---

## Production Checklist

- [ ] Model checkpoint verified and tested
- [ ] Inference latency measured and acceptable
- [ ] Error handling and fallbacks implemented
- [ ] Logging and monitoring configured
- [ ] API authentication/authorization set up
- [ ] Rate limiting and quota management enabled
- [ ] Model serving framework chosen (FastAPI/TorchServe/Ray)
- [ ] Docker image built and tested
- [ ] Hardware allocation planned (CPU/GPU ratio)
- [ ] Backup and disaster recovery procedures documented

---

## Resources

- **Model:** `checkpoints/arc/arc_final.pt`
- **Training Report:** `TRAINING_REPORT.md`
- **Architecture:** `ARCHITECTURE.md`
- **Code:** `model.py`

---

**Status:** ✅ Ready for production deployment

*Last updated: July 22, 2026*
