#!/usr/bin/env python3
"""
ARC-AGI Dataset Builder
Fetches all puzzle data directly from arcprize.org public API.
Endpoint: https://arcprize.org/media/data/task/task-data/{task_id}.json
"""

import json, os, sys, time, urllib.request, urllib.error, ssl
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = "https://arcprize.org/media/data/task/task-data"
INDEX_URL = "https://arcprize.org/media/data/datasets"
OUTPUT_DIR = Path(__file__).parent / "dataset"

DATASETS = {
    "v2_public_evaluation_set": f"{INDEX_URL}/v2_public_evaluation_set.json",
    "v2_public_training_set": f"{INDEX_URL}/v2_public_training_set.json",
    "v1_public_evaluation_set": f"{INDEX_URL}/v1_public_evaluation_set.json",
    "v1_public_training_set": f"{INDEX_URL}/v1_public_training_set.json",
}

ctx = ssl.create_default_context()

def fetch_json(url: str, timeout: int = 15) -> dict | list | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ARC-Dataset-Builder/1.0"})
        with urllib.request.urlopen(req, context=ctx, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        return None

def fetch_task(task_id: str) -> tuple[str, dict | None]:
    url = f"{BASE_URL}/{task_id}.json"
    data = fetch_json(url)
    return task_id, data

def main():
    sets_to_fetch = sys.argv[1:] if len(sys.argv) > 1 else list(DATASETS.keys())
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Fetch dataset indexes
    all_task_ids = {}
    for name in sets_to_fetch:
        if name not in DATASETS:
            print(f"❌ Unknown dataset: {name}")
            continue
        print(f"📋 Fetching index: {name}...")
        task_ids = fetch_json(DATASETS[name])
        if task_ids:
            all_task_ids[name] = task_ids
            print(f"   → {len(task_ids)} tasks")
            # Save index
            with open(OUTPUT_DIR / f"{name}.json", "w") as f:
                json.dump(task_ids, f, indent=2)
        else:
            print(f"   ❌ Failed to fetch index")
    
    # Deduplicate task IDs across all sets
    unique_tasks = {}
    for name, ids in all_task_ids.items():
        for tid in ids:
            if tid not in unique_tasks:
                unique_tasks[tid] = []
            unique_tasks[tid].append(name)
    
    print(f"\n📊 Total unique tasks: {len(unique_tasks)}")
    
    # Check what we already have
    tasks_dir = OUTPUT_DIR / "tasks"
    tasks_dir.mkdir(exist_ok=True)
    
    already_have = set()
    for f in tasks_dir.glob("*.json"):
        already_have.add(f.stem)
    
    to_fetch = [tid for tid in unique_tasks if tid not in already_have]
    print(f"   Already downloaded: {len(already_have)}")
    print(f"   Need to fetch: {len(to_fetch)}")
    
    if not to_fetch:
        print("✅ All tasks already downloaded!")
    else:
        print(f"\n🚀 Fetching {len(to_fetch)} tasks (8 threads)...")
        
        success = 0
        failed = []
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(fetch_task, tid): tid for tid in to_fetch}
            
            for i, future in enumerate(as_completed(futures), 1):
                tid, data = future.result()
                if data:
                    with open(tasks_dir / f"{tid}.json", "w") as f:
                        json.dump(data, f)
                    success += 1
                else:
                    failed.append(tid)
                
                if i % 50 == 0 or i == len(to_fetch):
                    print(f"   [{i}/{len(to_fetch)}] ✅ {success} fetched, ❌ {len(failed)} failed")
        
        if failed:
            print(f"\n⚠️  Failed tasks ({len(failed)}): {', '.join(failed[:20])}")
            with open(OUTPUT_DIR / "failed.json", "w") as f:
                json.dump(failed, f, indent=2)
    
    # Build summary
    print(f"\n📦 Building dataset summary...")
    summary = {
        "source": "https://arcprize.org",
        "endpoint": f"{BASE_URL}/{{task_id}}.json",
        "datasets": {},
        "total_unique_tasks": len(unique_tasks),
        "tasks_downloaded": len(already_have) + (success if to_fetch else 0),
    }
    
    for name, ids in all_task_ids.items():
        downloaded = sum(1 for tid in ids if (tasks_dir / f"{tid}.json").exists())
        summary["datasets"][name] = {
            "count": len(ids),
            "downloaded": downloaded,
        }
    
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(json.dumps(summary, indent=2))
    print("\n✅ Dataset saved to:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
