#!/usr/bin/env python3
"""
NGVT Demo Servers - Test Client
Tests both standard and ultra-optimized servers
"""

import requests
import json
import time
from typing import List, Dict, Any

class NGVTClient:
    def __init__(self, standard_url: str = "http://127.0.0.1:8080", 
                 ultra_url: str = "http://127.0.0.1:8081"):
        self.standard_url = standard_url
        self.ultra_url = ultra_url
    
    def test_health(self):
        """Test health endpoints"""
        print("\n" + "="*60)
        print("HEALTH CHECK")
        print("="*60)
        
        try:
            resp = requests.get(f"{self.standard_url}/health", timeout=2)
            print(f"✅ Standard Server (8080): {resp.status_code}")
            print(json.dumps(resp.json(), indent=2))
        except Exception as e:
            print(f"❌ Standard Server: {e}")
        
        try:
            resp = requests.get(f"{self.ultra_url}/health", timeout=2)
            print(f"\n✅ Ultra Server (8081): {resp.status_code}")
            print(json.dumps(resp.json(), indent=2))
        except Exception as e:
            print(f"❌ Ultra Server: {e}")
    
    def test_inference(self, prompt: str = "What is NGVT?", max_tokens: int = 50):
        """Test single inference"""
        print("\n" + "="*60)
        print("INFERENCE TEST")
        print("="*60)
        
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.7
        }
        
        # Standard Server
        try:
            start = time.time()
            resp = requests.post(f"{self.standard_url}/inference", 
                               json=payload, timeout=10)
            latency = (time.time() - start) * 1000
            
            if resp.status_code == 200:
                data = resp.json()
                print(f"\n✅ Standard Server:")
                print(f"   Latency: {data['latency_ms']:.2f}ms")
                print(f"   Tokens: {data['tokens_generated']}")
                print(f"   Response: {data['response'][:60]}...")
            else:
                print(f"❌ Standard Server: {resp.status_code}")
        except Exception as e:
            print(f"❌ Standard Server: {e}")
        
        # Ultra Server
        try:
            start = time.time()
            resp = requests.post(f"{self.ultra_url}/inference", 
                               json=payload, timeout=10)
            latency = (time.time() - start) * 1000
            
            if resp.status_code == 200:
                data = resp.json()
                print(f"\n✅ Ultra Server:")
                print(f"   Latency: {data['latency_ms']:.2f}ms")
                print(f"   Tokens: {data['tokens_generated']}")
                print(f"   Cache Hit: {data.get('cache_hit', False)}")
                print(f"   Response: {data['response'][:60]}...")
            else:
                print(f"❌ Ultra Server: {resp.status_code}")
        except Exception as e:
            print(f"❌ Ultra Server: {e}")
    
    def test_caching(self, prompt: str = "Test caching"):
        """Test response caching on ultra server"""
        print("\n" + "="*60)
        print("CACHING TEST (Ultra Server)")
        print("="*60)
        
        payload = {
            "prompt": prompt,
            "max_tokens": 50,
            "temperature": 0.7
        }
        
        # First request (cache miss)
        print(f"\n🔄 Request 1 (Cache MISS):")
        try:
            start = time.time()
            resp = requests.post(f"{self.ultra_url}/inference", 
                               json=payload, timeout=10)
            latency1 = (time.time() - start) * 1000
            
            if resp.status_code == 200:
                data = resp.json()
                print(f"   Latency: {data['latency_ms']:.2f}ms")
                print(f"   Cache Hit: {data.get('cache_hit', False)}")
            else:
                print(f"❌ Error: {resp.status_code}")
        except Exception as e:
            print(f"❌ Error: {e}")
            latency1 = 0
        
        time.sleep(1)
        
        # Second request (cache hit)
        print(f"\n⚡ Request 2 (Cache HIT):")
        try:
            start = time.time()
            resp = requests.post(f"{self.ultra_url}/inference", 
                               json=payload, timeout=10)
            latency2 = (time.time() - start) * 1000
            
            if resp.status_code == 200:
                data = resp.json()
                print(f"   Latency: {data['latency_ms']:.2f}ms")
                print(f"   Cache Hit: {data.get('cache_hit', False)}")
                
                if latency1 > 0:
                    speedup = latency1 / latency2
                    print(f"\n💨 Speedup: {speedup:.1f}x faster!")
            else:
                print(f"❌ Error: {resp.status_code}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def test_metrics(self):
        """Test metrics endpoints"""
        print("\n" + "="*60)
        print("METRICS")
        print("="*60)
        
        try:
            resp = requests.get(f"{self.standard_url}/metrics", timeout=2)
            if resp.status_code == 200:
                data = resp.json()
                print(f"\n✅ Standard Server Metrics:")
                print(f"   Total Requests: {data['total_requests']}")
                print(f"   Total Tokens: {data['total_tokens_generated']}")
                print(f"   Throughput: {data['throughput_tokens_per_sec']:.2f} tokens/sec")
                print(f"   Model: {data['model_info']['name']}")
                print(f"   Accuracy (SWE-bench): {data['model_info']['performance']['swe_bench_lite']}")
        except Exception as e:
            print(f"❌ Standard Server: {e}")
        
        try:
            resp = requests.get(f"{self.ultra_url}/metrics", timeout=2)
            if resp.status_code == 200:
                data = resp.json()
                print(f"\n✅ Ultra Server Metrics:")
                print(f"   Total Requests: {data['total_requests']}")
                print(f"   Total Tokens: {data['total_tokens']}")
                print(f"   Cache Hits: {data['cache_hits']} ({data['cache_hit_rate_percent']:.1f}%)")
                print(f"   Throughput: {data['throughput_tokens_sec']:.2f} tokens/sec")
                print(f"   Cache Size: {data['cache_size']} entries")
                print(f"   Performance with Caching: {data['performance']['with_caching']}")
        except Exception as e:
            print(f"❌ Ultra Server: {e}")
    
    def run_all_tests(self):
        """Run all tests"""
        print("\n" + "█"*60)
        print("  NGVT DEMO SERVERS - COMPREHENSIVE TEST")
        print("█"*60)
        
        self.test_health()
        self.test_inference("What makes NGVT different?", 50)
        self.test_caching("NGVT is the fastest LLM")
        self.test_metrics()
        
        print("\n" + "█"*60)
        print("  ✅ TESTS COMPLETE")
        print("█"*60 + "\n")

if __name__ == "__main__":
    client = NGVTClient()
    client.run_all_tests()
