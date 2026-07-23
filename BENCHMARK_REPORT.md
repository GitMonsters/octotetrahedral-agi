# Benchmark Report

## Latency Metrics

- Average latency: **65.29 ms**
- Minimum latency: **63.36 ms**
- Maximum latency: **70.09 ms**
- Standard deviation: **1.84 ms**
- Consistency validation: **same input = same output ✅**

## Throughput Analysis

- Baseline throughput: **16.20 req/s**
- Concurrency profile: **50 concurrent requests**
- Stability note: low latency variance indicates a stable CPU inference path under the sampled workload.

## Batch Size Scaling

| Batch size | Latency (ms) | Predictions returned |
| --- | ---: | ---: |
| 5 | 66.81 | 5 |
| 10 | 77.30 | 10 |
| 25 | 105.68 | 25 |
| 50 | 155.65 | 50 |
| 100 | 229.79 | 100 |

## Performance Characteristics

- CPU inference is deterministic and stable for repeated requests.
- Latency scales roughly linearly with batch size, which indicates batching overhead is predictable but still CPU-bound.
- Baseline throughput is suitable for low-volume traffic, but sustained higher concurrency will need acceleration or horizontal scaling.

## Optimization Opportunities

1. **GPU acceleration**: expected batch latency drops from ~65 ms to **6-7 ms** with CUDA/Metal support.
2. **Response caching**: repeated requests can avoid redundant tensor execution for highly duplicated token sequences.
3. **Quantization**: reduced precision inference can lower memory pressure and improve throughput on both GPU and CPU.
4. **Adaptive batching**: cap batches near the sweet spot where throughput improves faster than tail latency grows.

## Recommendations

- Prefer **CUDA** when available; use **Metal/MPS** on Apple Silicon; keep CPU fallback enabled for portability.
- Add request-level caching for repeated prompts and health-safe cache invalidation.
- Evaluate **8-bit / 4-bit quantization** for production deployments with tight latency or memory budgets.
- Track p95/p99 latency during sustained load before increasing concurrency beyond 50 requests.
