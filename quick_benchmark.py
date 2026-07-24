#!/usr/bin/env python3
"""
Quick Performance Benchmark for OctoTetrahedral AGI
Tests the local API with simple performance metrics
"""

import asyncio
import time
import json
import requests
from datetime import datetime
from pathlib import Path

class QuickBenchmark:
    def __init__(self, api_url="http://localhost:8000", api_key="qU62MH7IOkLzFDUHCVJoRlrc41nzzNNa8-Hhnm2YwVQ"):
        self.api_url = api_url
        self.api_key = api_key
        self.results = []

    async def run_benchmarks(self):
        """Run quick benchmarks"""
        print("\n" + "="*70)
        print("🚀 OctoTetrahedral AGI - Quick Benchmark")
        print("="*70 + "\n")

        # Test 1: Health Check
        print("1️⃣  Health Check...")
        try:
            resp = requests.get(f"{self.api_url}/health", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                print(f"   ✅ Device: {data.get('device')}")
                print(f"   ✅ Status: {data.get('status')}\n")
            else:
                print(f"   ❌ Error: {resp.status_code}\n")
                return
        except Exception as e:
            print(f"   ❌ Error: {e}\n")
            return

        # Test 2: Single Inference (5 requests)
        print("2️⃣  Single Inference (5 requests)...")
        latencies = []
        for i in range(5):
            try:
                start = time.time()
                resp = requests.post(
                    f"{self.api_url}/predict",
                    json={"input_ids": [1, 2, 3, 4, 5]},
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=60
                )
                latency = (time.time() - start) * 1000

                if resp.status_code == 200:
                    latencies.append(latency)
                    print(f"   Request {i+1}: {latency:.0f}ms ✅")
                else:
                    print(f"   Request {i+1}: Error {resp.status_code} ❌")
            except Exception as e:
                print(f"   Request {i+1}: {str(e)[:30]} ❌")

        if latencies:
            print(f"\n   📊 Average: {sum(latencies)/len(latencies):.0f}ms")
            print(f"   📊 Min: {min(latencies):.0f}ms")
            print(f"   📊 Max: {max(latencies):.0f}ms\n")

        # Test 3: Memory & Stats
        print("3️⃣  System Statistics...")
        try:
            resp = requests.get(f"{self.api_url}/stats", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                print(f"   💾 Memory: {data.get('memory_mb', 0):.1f}MB")
                print(f"   📈 Total Requests: {data.get('total_requests', 0)}")
                print(f"   ⏱️  Uptime: {data.get('uptime_seconds', 0)}s")
                print(f"   ⚡ Throughput: {data.get('throughput_req_per_sec', 0):.2f} req/sec\n")
            else:
                print(f"   ❌ Error: {resp.status_code}\n")
        except Exception as e:
            print(f"   ❌ Error: {e}\n")

        print("="*70)
        print("✅ Benchmark Complete!")
        print("="*70 + "\n")


if __name__ == "__main__":
    benchmark = QuickBenchmark()
    asyncio.run(benchmark.run_benchmarks())
