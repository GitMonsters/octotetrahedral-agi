#!/usr/bin/env python3
"""
Batch ARC-AGI Puzzle Solver
Solves multiple puzzles via the /solve-arc API endpoint
"""

import json
import argparse
import requests
import time
from pathlib import Path
from typing import Dict, List, Any
from collections import Counter
import sys

class ARCBatchSolver:
    """Batch solver for ARC puzzles via HTTP API"""
    
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def solve_task(self, task_id: str, task: Dict[str, Any], method: str = "auto") -> Dict:
        """Solve a single task via API"""
        try:
            payload = {
                "task": task,
                "method": method,
                "task_id": task_id
            }
            
            response = requests.post(
                f"{self.base_url}/solve-arc",
                json=payload,
                headers=self.headers,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "task_id": task_id,
                    "success": True,
                    "method": result.get("method", "unknown"),
                    "confidence": result.get("confidence", 0),
                    "predictions": result.get("predictions", []),
                    "verified": result.get("verified_on_training", False),
                    "latency_ms": result.get("latency_ms", 0),
                    "reasoning": result.get("reasoning", "")
                }
            else:
                return {
                    "task_id": task_id,
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}"
                }
        except requests.exceptions.RequestException as e:
            return {
                "task_id": task_id,
                "success": False,
                "error": f"Request failed: {str(e)}"
            }
        except Exception as e:
            return {
                "task_id": task_id,
                "success": False,
                "error": f"Error: {str(e)}"
            }
    
    def solve_batch(self, tasks: Dict[str, Dict], method: str = "auto", 
                   limit: int = None) -> List[Dict]:
        """Solve multiple tasks"""
        results = []
        task_ids = list(tasks.keys())
        
        if limit:
            task_ids = task_ids[:limit]
        
        for i, task_id in enumerate(task_ids, 1):
            print(f"[{i}/{len(task_ids)}] Solving {task_id}...", end=" ", flush=True)
            
            result = self.solve_task(task_id, tasks[task_id], method=method)
            results.append(result)
            
            if result["success"]:
                print(f"✅ ({result['method']}, conf={result['confidence']:.2f}, {result['latency_ms']:.1f}ms)")
            else:
                print(f"❌ {result['error'][:50]}")
        
        return results
    
    def format_results(self, results: List[Dict]) -> Dict[str, Any]:
        """Aggregate and format results"""
        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]
        
        method_counts = Counter(r["method"] for r in successful)
        confidence_scores = [r["confidence"] for r in successful]
        latencies = [r["latency_ms"] for r in successful]
        
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        
        return {
            "summary": {
                "total_tasks": len(results),
                "successful": len(successful),
                "failed": len(failed),
                "success_rate": f"{len(successful) / len(results) * 100:.1f}%" if results else "0%"
            },
            "performance": {
                "avg_confidence": round(avg_confidence, 3),
                "avg_latency_ms": round(avg_latency, 2),
                "min_latency_ms": round(min(latencies), 2) if latencies else 0,
                "max_latency_ms": round(max(latencies), 2) if latencies else 0
            },
            "methods": dict(method_counts.most_common()),
            "results": results
        }


def main():
    parser = argparse.ArgumentParser(description="Batch solve ARC-AGI puzzles")
    parser.add_argument("--input", required=True, help="Input JSON file with ARC tasks")
    parser.add_argument("--output", required=True, help="Output JSON file for results")
    parser.add_argument("--api-key", required=True, help="API key for authentication")
    parser.add_argument("--host", default="localhost", help="API host (default: localhost)")
    parser.add_argument("--port", type=int, default=8001, help="API port (default: 8001)")
    parser.add_argument("--method", default="auto", help="Solve method (auto/rule_learner/neural/mistral)")
    parser.add_argument("--limit", type=int, help="Limit number of tasks to solve")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Load input
    print(f"\n📂 Loading tasks from {args.input}...")
    try:
        with open(args.input) as f:
            tasks = json.load(f)
        print(f"✅ Loaded {len(tasks)} tasks")
    except FileNotFoundError:
        print(f"❌ File not found: {args.input}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"❌ Invalid JSON: {args.input}")
        sys.exit(1)
    
    # Solve
    base_url = f"http://{args.host}:{args.port}"
    print(f"🚀 Solving via {base_url} (method={args.method})")
    print(f"{'='*70}")
    
    solver = ARCBatchSolver(base_url, args.api_key)
    t0 = time.time()
    results = solver.solve_batch(tasks, method=args.method, limit=args.limit)
    elapsed = time.time() - t0
    
    # Format and save
    print(f"{'='*70}")
    output = solver.format_results(results)
    output["total_time_seconds"] = round(elapsed, 2)
    
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n{'='*70}")
    print("📊 RESULTS")
    print(f"{'='*70}")
    print(f"Total Tasks:      {output['summary']['total_tasks']}")
    print(f"Successful:       {output['summary']['successful']}")
    print(f"Failed:           {output['summary']['failed']}")
    print(f"Success Rate:     {output['summary']['success_rate']}")
    print(f"\nAvg Confidence:   {output['performance']['avg_confidence']:.3f}")
    print(f"Avg Latency:      {output['performance']['avg_latency_ms']:.2f}ms")
    print(f"Total Time:       {output['total_time_seconds']:.1f}s")
    print(f"\nTop Methods:")
    for method, count in sorted(output['methods'].items(), key=lambda x: x[1], reverse=True)[:5]:
        pct = count / output['summary']['successful'] * 100 if output['summary']['successful'] else 0
        print(f"  {method:30s} {count:4d} ({pct:.1f}%)")
    
    print(f"\n✅ Results saved to {args.output}")
    print(f"{'='*70}\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
