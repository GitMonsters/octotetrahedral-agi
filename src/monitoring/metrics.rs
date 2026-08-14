//! Ring-buffer latency / error metrics for the inference service.

use std::collections::VecDeque;
use std::sync::Mutex;
use std::time::Instant;

use serde::{Deserialize, Serialize};

const MAX_SAMPLES: usize = 1000;

/// Snapshot of current metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSnapshot {
    pub uptime_seconds: u64,
    pub total_requests: u64,
    pub avg_latency_ms: f64,
    pub min_latency_ms: f64,
    pub max_latency_ms: f64,
    pub error_count: u64,
    pub throughput_req_per_sec: f64,
}

/// Thread-safe rolling-window metrics collector.
pub struct InferenceMonitor {
    inner: Mutex<MonitorInner>,
}

struct MonitorInner {
    latencies: VecDeque<f64>,
    errors: u64,
    total: u64,
    start: Instant,
}

impl InferenceMonitor {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(MonitorInner {
                latencies: VecDeque::with_capacity(MAX_SAMPLES),
                errors: 0,
                total: 0,
                start: Instant::now(),
            }),
        }
    }

    /// Record a completed request.
    pub fn record(&self, latency_ms: f64, is_error: bool) {
        let mut inner = self.inner.lock().expect("metrics mutex poisoned");
        inner.total += 1;
        if is_error {
            inner.errors += 1;
        }
        if inner.latencies.len() == MAX_SAMPLES {
            inner.latencies.pop_front();
        }
        inner.latencies.push_back(latency_ms);
    }

    /// Return a current metrics snapshot.
    pub fn snapshot(&self) -> MetricsSnapshot {
        let inner = self.inner.lock().expect("metrics mutex poisoned");
        let uptime = inner.start.elapsed().as_secs();
        if inner.latencies.is_empty() {
            return MetricsSnapshot {
                uptime_seconds: uptime,
                total_requests: inner.total,
                avg_latency_ms: 0.0,
                min_latency_ms: 0.0,
                max_latency_ms: 0.0,
                error_count: inner.errors,
                throughput_req_per_sec: 0.0,
            };
        }
        let lats: Vec<f64> = inner.latencies.iter().cloned().collect();
        let avg = lats.iter().sum::<f64>() / lats.len() as f64;
        let min = lats.iter().cloned().fold(f64::MAX, f64::min);
        let max = lats.iter().cloned().fold(f64::MIN, f64::max);
        let throughput = if uptime > 0 {
            inner.total as f64 / uptime as f64
        } else {
            0.0
        };
        MetricsSnapshot {
            uptime_seconds: uptime,
            total_requests: inner.total,
            avg_latency_ms: avg,
            min_latency_ms: min,
            max_latency_ms: max,
            error_count: inner.errors,
            throughput_req_per_sec: throughput,
        }
    }
}

impl Default for InferenceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_requests_and_errors() {
        let m = InferenceMonitor::new();
        m.record(10.0, false);
        m.record(20.0, true);
        let snap = m.snapshot();
        assert_eq!(snap.total_requests, 2);
        assert_eq!(snap.error_count, 1);
        assert!((snap.avg_latency_ms - 15.0).abs() < 0.01);
    }
}
