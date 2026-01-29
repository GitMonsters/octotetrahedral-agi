#!/usr/bin/env python3
"""
NGVT Load Testing & Benchmark Suite
Comprehensive performance evaluation of both servers
"""

import subprocess
import json
import time
import requests
import statistics
from typing import Dict, List, Any
from datetime import datetime

class NGVTLoadTest:
    def __init__(self, standard_url: str = "http://127.0.0.1:8080",
                 ultra_url: str = "http://127.0.0.1:8081"):
        self.standard_url = standard_url
        self.ultra_url = ultra_url
        self.results = {}
    
    def print_header(self, text: str):
        """Print formatted header"""
        print(f"\n{'='*70}")
        print(f"  {text}")
        print(f"{'='*70}\n")
    
    def run_apache_bench(self, url: str, name: str, num_requests: int = 1000, 
                        concurrency: int = 10, payload: str = None):
        """Run Apache Bench load test"""
        print(f"🔄 Running Apache Bench on {name}...")
        print(f"   Requests: {num_requests}, Concurrency: {concurrency}")
        
        cmd = [
            'ab',
            '-n', str(num_requests),
            '-c', str(concurrency),
            '-t', '60'  # 60 second timeout
        ]
        
        # Add POST data if provided
        if payload:
            cmd.extend(['-p', payload, '-T', 'application/json'])
        
        cmd.append(url)
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            output = result.stdout
            
            # Parse results
            results = {
                'raw_output': output,
                'completed': 'Finished' in output or 'requests completed' in output
            }
            
            # Extract key metrics
            for line in output.split('\n'):
                if 'Requests per second' in line:
                    results['requests_per_sec'] = float(line.split(':')[1].strip().split()[0])
                elif 'Time per request' in line and 'mean' in line:
                    results['mean_response_time_ms'] = float(line.split(':')[1].strip().split()[0])
                elif 'Transfer rate' in line:
                    results['transfer_rate_kbps'] = float(line.split(':')[1].strip().split()[0])
                elif 'Failed requests' in line:
                    results['failed_requests'] = int(line.split(':')[1].strip())
                elif 'Non-2xx responses' in line:
                    results['non_2xx'] = int(line.split(':')[1].strip())
            
            if results.get('completed'):
                print(f"   ✅ Completed: {results.get('requests_per_sec', 0):.2f} req/sec")
                print(f"   ⏱️  Mean latency: {results.get('mean_response_time_ms', 0):.2f}ms")
            else:
                print(f"   ⚠️  Test may have issues")
            
            return results
        
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return {'error': str(e)}
    
    def run_custom_load_test(self, url: str, name: str, num_requests: int = 100,
                            concurrency: int = 10):
        """Run custom Python-based load test"""
        print(f"🔄 Running custom load test on {name}...")
        print(f"   Requests: {num_requests}, Concurrency: {concurrency}")
        
        import concurrent.futures
        import threading
        
        payload = {
            "prompt": "Performance test prompt for NGVT system",
            "max_tokens": 50,
            "temperature": 0.7
        }
        
        latencies = []
        errors = 0
        lock = threading.Lock()
        
        def make_request():
            nonlocal errors
            try:
                start = time.time()
                response = requests.post(
                    f"{url}/inference",
                    json=payload,
                    timeout=30
                )
                latency = (time.time() - start) * 1000
                
                with lock:
                    if response.status_code == 200:
                        latencies.append(latency)
                    else:
                        errors += 1
                
                return latency
            except Exception as e:
                with lock:
                    errors += 1
                return None
        
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(make_request) for _ in range(num_requests)]
            concurrent.futures.wait(futures)
        
        total_time = time.time() - start_time
        
        if latencies:
            results = {
                'total_requests': num_requests,
                'successful_requests': len(latencies),
                'failed_requests': errors,
                'total_time_seconds': total_time,
                'requests_per_sec': num_requests / total_time,
                'mean_latency_ms': statistics.mean(latencies),
                'median_latency_ms': statistics.median(latencies),
                'min_latency_ms': min(latencies),
                'max_latency_ms': max(latencies),
                'stddev_latency_ms': statistics.stdev(latencies) if len(latencies) > 1 else 0,
                'p95_latency_ms': sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0,
                'p99_latency_ms': sorted(latencies)[int(len(latencies) * 0.99)] if latencies else 0,
            }
            
            print(f"   ✅ Completed: {results['requests_per_sec']:.2f} req/sec")
            print(f"   ⏱️  Mean latency: {results['mean_latency_ms']:.2f}ms")
            print(f"   📊 Median latency: {results['median_latency_ms']:.2f}ms")
            print(f"   ⚠️  Errors: {errors}/{num_requests}")
            
            return results
        else:
            print(f"   ❌ All requests failed")
            return {'error': 'All requests failed'}
    
    def test_caching_performance(self):
        """Test caching performance on ultra server"""
        self.print_header("CACHING PERFORMANCE TEST")
        
        prompts = [
            "What is NGVT?",
            "How does NGVT work?",
            "What are the benefits of NGVT?",
            "What is NGVT?",  # Repeat for cache hit
            "How does NGVT work?",  # Repeat for cache hit
        ]
        
        results = {
            'cache_misses': 0,
            'cache_hits': 0,
            'miss_latencies': [],
            'hit_latencies': []
        }
        
        print("Testing response caching with repeated prompts...")
        
        for prompt in prompts:
            payload = {"prompt": prompt, "max_tokens": 50, "temperature": 0.7}
            
            start = time.time()
            response = requests.post(
                f"{self.ultra_url}/inference",
                json=payload,
                timeout=30
            )
            latency = (time.time() - start) * 1000
            
            if response.status_code == 200:
                data = response.json()
                is_cached = data.get('cache_hit', False)
                
                if is_cached:
                    results['cache_hits'] += 1
                    results['hit_latencies'].append(latency)
                    print(f"  ✅ Cache HIT  - '{prompt[:30]}...' - {latency:.2f}ms")
                else:
                    results['cache_misses'] += 1
                    results['miss_latencies'].append(latency)
                    print(f"  ❌ Cache MISS - '{prompt[:30]}...' - {latency:.2f}ms")
        
        if results['miss_latencies']:
            avg_miss = statistics.mean(results['miss_latencies'])
            print(f"\n📊 Cache Miss Average: {avg_miss:.2f}ms")
        
        if results['hit_latencies']:
            avg_hit = statistics.mean(results['hit_latencies'])
            print(f"📊 Cache Hit Average: {avg_hit:.2f}ms")
            
            if results['miss_latencies']:
                speedup = avg_miss / avg_hit
                print(f"💨 Speedup: {speedup:.1f}x faster with caching!")
        
        results['hit_rate'] = (results['cache_hits'] / (results['cache_hits'] + results['cache_misses']) * 100) if (results['cache_hits'] + results['cache_misses']) > 0 else 0
        print(f"📈 Cache Hit Rate: {results['hit_rate']:.1f}%")
        
        return results
    
    def compare_servers(self):
        """Compare performance between standard and ultra servers"""
        self.print_header("SERVER COMPARISON TEST")
        
        num_requests = 100
        payload = {"prompt": "Compare performance", "max_tokens": 50}
        
        comparison = {}
        
        # Standard Server
        print("Testing Standard Server...")
        standard_results = self.run_custom_load_test(
            self.standard_url, "Standard Server",
            num_requests=num_requests, concurrency=5
        )
        comparison['standard'] = standard_results
        
        time.sleep(2)
        
        # Ultra Server
        print("\nTesting Ultra Server...")
        ultra_results = self.run_custom_load_test(
            self.ultra_url, "Ultra Server",
            num_requests=num_requests, concurrency=5
        )
        comparison['ultra'] = ultra_results
        
        # Comparison
        print("\n" + "-"*70)
        print("PERFORMANCE COMPARISON:")
        print("-"*70)
        
        if 'requests_per_sec' in standard_results and 'requests_per_sec' in ultra_results:
            std_req = standard_results['requests_per_sec']
            ultra_req = ultra_results['requests_per_sec']
            speedup = ultra_req / std_req if std_req > 0 else 0
            
            print(f"Standard: {std_req:.2f} req/sec")
            print(f"Ultra:    {ultra_req:.2f} req/sec")
            print(f"Speedup:  {speedup:.2f}x")
        
        if 'mean_latency_ms' in standard_results and 'mean_latency_ms' in ultra_results:
            std_lat = standard_results['mean_latency_ms']
            ultra_lat = ultra_results['mean_latency_ms']
            improvement = ((std_lat - ultra_lat) / std_lat * 100) if std_lat > 0 else 0
            
            print(f"\nStandard Latency: {std_lat:.2f}ms")
            print(f"Ultra Latency:    {ultra_lat:.2f}ms")
            print(f"Improvement:      {improvement:.1f}%")
        
        return comparison
    
    def run_full_benchmark(self):
        """Run complete benchmark suite"""
        print("\n" + "█"*70)
        print("  NGVT LOAD TESTING & BENCHMARK SUITE")
        print("  Starting comprehensive performance evaluation...")
        print("█"*70)
        
        self.results['timestamp'] = datetime.now().isoformat()
        
        # 1. Health check
        self.print_header("HEALTH CHECK")
        try:
            std_health = requests.get(f"{self.standard_url}/health", timeout=2).status_code
            ultra_health = requests.get(f"{self.ultra_url}/health", timeout=2).status_code
            print(f"✅ Standard Server: {std_health}")
            print(f"✅ Ultra Server: {ultra_health}")
        except Exception as e:
            print(f"❌ Error: {e}")
            return
        
        # 2. Custom load test - light
        self.print_header("LIGHT LOAD TEST (100 requests, 5 concurrent)")
        light_results = self.compare_servers()
        self.results['light_load'] = light_results
        
        # 3. Caching test
        self.print_header("CACHING PERFORMANCE TEST")
        caching_results = self.test_caching_performance()
        self.results['caching'] = caching_results
        
        # 4. Custom load test - heavy
        self.print_header("HEAVY LOAD TEST (500 requests, 20 concurrent)")
        print("Standard Server...")
        standard_heavy = self.run_custom_load_test(
            self.standard_url, "Standard",
            num_requests=500, concurrency=20
        )
        self.results['standard_heavy'] = standard_heavy
        
        time.sleep(3)
        
        print("\nUltra Server...")
        ultra_heavy = self.run_custom_load_test(
            self.ultra_url, "Ultra",
            num_requests=500, concurrency=20
        )
        self.results['ultra_heavy'] = ultra_heavy
        
        # 5. Apache Bench test (if available)
        self.print_header("APACHE BENCH STRESS TEST (ab)")
        try:
            print("Standard Server (1000 requests, 10 concurrent)...")
            ab_standard = self.run_apache_bench(
                f"{self.standard_url}/health",
                "Standard Server Health",
                num_requests=1000,
                concurrency=10
            )
            self.results['apache_bench_standard'] = ab_standard
            
            time.sleep(3)
            
            print("\nUltra Server (1000 requests, 10 concurrent)...")
            ab_ultra = self.run_apache_bench(
                f"{self.ultra_url}/health",
                "Ultra Server Health",
                num_requests=1000,
                concurrency=10
            )
            self.results['apache_bench_ultra'] = ab_ultra
        except Exception as e:
            print(f"⚠️  Apache Bench unavailable: {e}")
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print final benchmark summary"""
        self.print_header("BENCHMARK SUMMARY")
        
        print(f"Timestamp: {self.results.get('timestamp')}")
        
        # Light load results
        if 'light_load' in self.results:
            print("\n📊 LIGHT LOAD TEST (100 req, 5 concurrent):")
            std = self.results['light_load'].get('standard', {})
            ultra = self.results['light_load'].get('ultra', {})
            
            if std.get('requests_per_sec'):
                print(f"  Standard: {std['requests_per_sec']:.2f} req/sec, {std['mean_latency_ms']:.2f}ms")
            if ultra.get('requests_per_sec'):
                print(f"  Ultra:    {ultra['requests_per_sec']:.2f} req/sec, {ultra['mean_latency_ms']:.2f}ms")
        
        # Caching results
        if 'caching' in self.results:
            cache = self.results['caching']
            print(f"\n💾 CACHING TEST:")
            print(f"  Cache Hit Rate: {cache.get('hit_rate', 0):.1f}%")
            print(f"  Hits: {cache.get('cache_hits', 0)}, Misses: {cache.get('cache_misses', 0)}")
        
        # Heavy load results
        if 'standard_heavy' in self.results:
            std = self.results['standard_heavy']
            print(f"\n⚡ HEAVY LOAD TEST (500 req, 20 concurrent):")
            print(f"  Standard: {std.get('requests_per_sec', 0):.2f} req/sec")
        
        if 'ultra_heavy' in self.results:
            ultra = self.results['ultra_heavy']
            print(f"  Ultra:    {ultra.get('requests_per_sec', 0):.2f} req/sec")
        
        # Apache Bench results
        if 'apache_bench_standard' in self.results:
            print(f"\n🔥 APACHE BENCH (1000 requests):")
            ab_std = self.results['apache_bench_standard']
            ab_ultra = self.results['apache_bench_ultra']
            
            if ab_std.get('requests_per_sec'):
                print(f"  Standard: {ab_std['requests_per_sec']:.2f} req/sec")
            if ab_ultra.get('requests_per_sec'):
                print(f"  Ultra:    {ab_ultra['requests_per_sec']:.2f} req/sec")
        
        print("\n" + "█"*70)
        print("  BENCHMARK COMPLETE")
        print("█"*70 + "\n")
        
        # Save results
        self.save_results()
    
    def save_results(self):
        """Save benchmark results to file"""
        try:
            with open('/tmp/ngvt_benchmark_results.json', 'w') as f:
                json.dump(self.results, f, indent=2)
            print(f"✅ Results saved to: /tmp/ngvt_benchmark_results.json")
        except Exception as e:
            print(f"⚠️  Could not save results: {e}")

if __name__ == "__main__":
    tester = NGVTLoadTest()
    tester.run_full_benchmark()
