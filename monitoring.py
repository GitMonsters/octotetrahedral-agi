import time
import psutil
from collections import deque
from datetime import datetime

class PerformanceMonitor:
    def __init__(self, max_samples=1000):
        self.latencies = deque(maxlen=max_samples)
        self.errors = 0
        self.total_requests = 0
        self.start_time = datetime.now()
    
    def record_request(self, latency_ms, error=False):
        """Record a request with latency"""
        self.total_requests += 1
        self.latencies.append(latency_ms)
        if error:
            self.errors += 1
    
    def get_stats(self):
        """Get current performance statistics"""
        if not self.latencies:
            return {
                "uptime_seconds": 0,
                "total_requests": self.total_requests,
                "avg_latency_ms": 0,
                "min_latency_ms": 0,
                "max_latency_ms": 0,
                "error_count": self.errors,
                "throughput_req_per_sec": 0,
                "memory_mb": round(psutil.Process().memory_info().rss / 1024 / 1024, 2)
            }
        
        latencies = list(self.latencies)
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        return {
            "uptime_seconds": int(uptime),
            "total_requests": self.total_requests,
            "avg_latency_ms": round(sum(latencies) / len(latencies), 2),
            "min_latency_ms": round(min(latencies), 2),
            "max_latency_ms": round(max(latencies), 2),
            "error_count": self.errors,
            "throughput_req_per_sec": round(self.total_requests / uptime if uptime > 0 else 0, 2),
            "memory_mb": round(psutil.Process().memory_info().rss / 1024 / 1024, 2)
        }

# Global monitor instance
monitor = PerformanceMonitor()
