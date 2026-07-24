#!/usr/bin/env python3
"""
Quick Performance Test Runner
Tests OctoTetrahedral AGI with real-world workloads
"""

import asyncio
import time
import json
import requests
from datetime import datetime
from typing import Dict, List, Any

class QuickPerformanceTest:
    """Run quick performance tests"""
    
    def __init__(self, api_url: str = "http://localhost:8000", api_key: str = ""):
        self.api_url = api_url
        self.api_key = api_key
        self.results = []
    
    async def test_latency(self, num_requests: int = 10) -> Dict[str, Any]:
        """Test single request latency"""
        print(f"\n⏱️  Testing Latency ({num_requests} requests)...")
        
        latencies = []
        errors = 0
        
        for i in range(num_requests):
            try:
                start = time.time()
                response = requests.post(
                    f"{self.api_url}/predict",
                    json={"input_ids": [1, 2, 3, 4, 5]},
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=60
                )
                latency = (time.time() - start) * 1000
                
                if response.status_code == 200:
                    latencies.append(latency)
                    print(f"  Request {i+1}/{num_requests}: {latency:.0f}ms ✅")
                else:
                    errors += 1
                    print(f"  Request {i+1}/{num_requests}: Error {response.status_code} ❌")
            except Exception as e:
                errors += 1
                print(f"  Request {i+1}/{num_requests}: {str(e)[:30]} ❌")
        
        if latencies:
            return {
                "test": "latency",
                "total_requests": num_requests,
                "successful": len(latencies),
                "failed": errors,
                "success_rate": len(latencies) / num_requests,
                "min_ms": min(latencies),
                "max_ms": max(latencies),
                "avg_ms": sum(latencies) / len(latencies),
                "median_ms": sorted(latencies)[len(latencies)//2]
            }
        return {"test": "latency", "error": "All requests failed"}
    
    async def test_throughput(self, duration_sec: int = 10) -> Dict[str, Any]:
        """Test throughput (requests per second)"""
        print(f"\n📈 Testing Throughput ({duration_sec}s test)...")
        
        start_time = time.time()
        request_count = 0
        error_count = 0
        latencies = []
        
        while time.time() - start_time < duration_sec:
            try:
                req_start = time.time()
                response = requests.post(
                    f"{self.api_url}/predict",
                    json={"input_ids": [1, 2, 3]},
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=30
                )
                latency = (time.time() - req_start) * 1000
                
                if response.status_code == 200:
                    request_count += 1
                    latencies.append(latency)
                else:
                    error_count += 1
            except:
                error_count += 1
        
        elapsed = time.time() - start_time
        throughput = request_count / elapsed if elapsed > 0 else 0
        
        print(f"  Requests: {request_count} ✅ | Errors: {error_count} ❌")
        print(f"  Throughput: {throughput:.2f} req/sec")
        
        return {
            "test": "throughput",
            "duration_sec": elapsed,
            "total_requests": request_count,
            "errors": error_count,
            "throughput_req_per_sec": throughput,
            "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0
        }
    
    async def test_concurrent(self, num_concurrent: int = 5) -> Dict[str, Any]:
        """Test concurrent request handling"""
        print(f"\n⚡ Testing Concurrent Requests ({num_concurrent} parallel)...")
        
        async def make_request():
            try:
                start = time.time()
                response = requests.post(
                    f"{self.api_url}/predict",
                    json={"input_ids": [1, 2, 3, 4, 5]},
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=60
                )
                latency = (time.time() - start) * 1000
                return {"status": response.status_code, "latency": latency}
            except Exception as e:
                return {"status": 500, "error": str(e)}
        
        start = time.time()
        
        # Run requests concurrently
        tasks = [make_request() for _ in range(num_concurrent)]
        results = await asyncio.gather(*tasks)
        
        total_time = (time.time() - start) * 1000
        successful = sum(1 for r in results if r["status"] == 200)
        
        latencies = [r["latency"] for r in results if "latency" in r]
        
        print(f"  Completed: {successful}/{num_concurrent} successful")
        print(f"  Total time: {total_time:.0f}ms")
        print(f"  Avg latency: {sum(latencies)/len(latencies) if latencies else 0:.0f}ms")
        
        return {
            "test": "concurrent",
            "concurrent_count": num_concurrent,
            "successful": successful,
            "failed": num_concurrent - successful,
            "total_time_ms": total_time,
            "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0
        }
    
    async def test_health(self) -> Dict[str, Any]:
        """Test health check endpoint"""
        print(f"\n🏥 Testing Health Check...")
        
        try:
            start = time.time()
            response = requests.get(f"{self.api_url}/health", timeout=5)
            latency = (time.time() - start) * 1000
            
            if response.status_code == 200:
                data = response.json()
                print(f"  Device: {data.get('device')}")
                print(f"  Status: {data.get('status')} ✅")
                print(f"  Latency: {latency:.1f}ms")
                return {
                    "test": "health",
                    "status": "ok",
                    "device": data.get("device"),
                    "latency_ms": latency
                }
        except Exception as e:
            print(f"  Error: {e} ❌")
            return {"test": "health", "status": "error", "error": str(e)}
    
    async def test_stats(self) -> Dict[str, Any]:
        """Test stats endpoint"""
        print(f"\n📊 Testing Stats Endpoint...")
        
        try:
            response = requests.get(f"{self.api_url}/stats", timeout=5)
            if response.status_code == 200:
                stats = response.json()
                print(f"  Total Requests: {stats.get('total_requests')}")
                print(f"  Avg Latency: {stats.get('avg_latency_ms'):.1f}ms")
                print(f"  Memory: {stats.get('memory_mb'):.1f}MB")
                print(f"  Uptime: {stats.get('uptime_seconds')}s ✅")
                return {
                    "test": "stats",
                    "status": "ok",
                    "data": stats
                }
        except Exception as e:
            print(f"  Error: {e} ❌")
            return {"test": "stats", "status": "error"}
    
    async def run_all_tests(self, api_key: str) -> Dict[str, Any]:
        """Run all performance tests"""
        self.api_key = api_key
        
        print("\n" + "="*70)
        print("🚀 OctoTetrahedral AGI - Performance Test Suite")
        print("="*70)
        
        all_results = {
            "timestamp": datetime.now().isoformat(),
            "api_url": self.api_url,
            "tests": []
        }
        
        # Run tests
        health = await self.test_health()
        all_results["tests"].append(health)
        
        if health.get("status") == "ok":
            stats = await self.test_stats()
            all_results["tests"].append(stats)
            
            latency = await self.test_latency(10)
            all_results["tests"].append(latency)
            
            throughput = await self.test_throughput(10)
            all_results["tests"].append(throughput)
            
            concurrent = await self.test_concurrent(5)
            all_results["tests"].append(concurrent)
        
        return all_results
    
    def print_summary(self, results: Dict[str, Any]):
        """Print test summary"""
        print("\n" + "="*70)
        print("📊 PERFORMANCE TEST RESULTS")
        print("="*70)
        
        for test in results["tests"]:
            test_name = test.get("test", "unknown")
            print(f"\n{test_name.upper()}:")
            
            if test.get("status") == "error":
                print(f"  ❌ Error: {test.get('error')}")
            elif test_name == "latency":
                print(f"  ✅ Min:     {test.get('min_ms', 0):.0f}ms")
                print(f"  ✅ Max:     {test.get('max_ms', 0):.0f}ms")
                print(f"  ✅ Avg:     {test.get('avg_ms', 0):.0f}ms")
                print(f"  ✅ Median:  {test.get('median_ms', 0):.0f}ms")
                print(f"  ✅ Success: {test.get('success_rate', 0):.1%}")
            elif test_name == "throughput":
                print(f"  ✅ Throughput: {test.get('throughput_req_per_sec', 0):.2f} req/sec")
                print(f"  ✅ Avg Latency: {test.get('avg_latency_ms', 0):.0f}ms")
            elif test_name == "concurrent":
                print(f"  ✅ Success: {test.get('successful')}/{test.get('concurrent_count')}")
                print(f"  ✅ Total Time: {test.get('total_time_ms', 0):.0f}ms")
            elif test_name == "health":
                print(f"  ✅ Device: {test.get('device')}")
                print(f"  ✅ Latency: {test.get('latency_ms', 0):.1f}ms")
            elif test_name == "stats":
                data = test.get("data", {})
                print(f"  ✅ Requests: {data.get('total_requests')}")
                print(f"  ✅ Memory: {data.get('memory_mb', 0):.1f}MB")
        
        print("\n" + "="*70)
        print("✅ Performance Testing Complete!")
        print("="*70 + "\n")


async def main():
    """Run performance tests"""
    import sys
    
    api_key = sys.argv[1] if len(sys.argv) > 1 else "qU62MH7IOkLzFDUHCVJoRlrc41nzzNNa8-Hhnm2YwVQ"
    
    tester = QuickPerformanceTest()
    results = await tester.run_all_tests(api_key)
    tester.print_summary(results)
    
    # Save results
    with open("performance_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("📄 Results saved to performance_test_results.json")


if __name__ == "__main__":
    asyncio.run(main())
