"""Run the synthesizer over a slice of the eval set and aggregate scores.

Workers run in parallel processes; each worker builds its own LLM client
and library index (OpenAI clients are not picklable).
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from .config import HarnessConfig
from .dataset import eval_ids, load_task
from .compounding import CompoundingStore


def _solved_ids(cfg: HarnessConfig) -> set[str]:
    solves_dir = os.path.join(cfg.solvers_root, "solves")
    if os.path.isdir(solves_dir):
        return {
            d for d in os.listdir(solves_dir)
            if os.path.isdir(os.path.join(solves_dir, d))
        }
    return set()


def _target_subset(cfg: HarnessConfig, subset: str, limit: int) -> list[str]:
    ids = sorted(eval_ids(cfg, subset))
    if cfg.exclude_solved:
        ids = [i for i in ids if i not in _solved_ids(cfg)]
    if limit and limit > 0:
        ids = ids[:limit]
    return ids


def _worker_solve(cfg: HarnessConfig, task_id: str) -> dict:
    """Top-level worker: solve one task and return a picklable result."""
    from .synthesizer import Synthesizer

    task = load_task(cfg, task_id)
    # compounding store for cross-task learning (read-only in worker)
    store = CompoundingStore(os.path.join(cfg.run_dir, "compounding"))
    synth = Synthesizer(cfg, store=store)
    t0 = time.time()
    out = synth.solve_task(task_id, task)
    dt = time.time() - t0

    # generalisation check: run the best code on the held-out test pairs
    test_passed = None
    test_total = None
    if out.get("code") and task.get("test"):
        tv = synth.apply_solver(out["code"], task["test"])
        test_passed, test_total = tv["test_passed"], tv["test_total"]

    # extract features for compounding retrieval
    features = _task_features(task)
    rules = out.get("rules", [])
    level = _task_level(task)

    return {
        "task_id": task_id,
        "status": out["status"],
        "score": out["score"],
        "attempts": out.get("attempts"),
        "time_s": round(dt, 1),
        "kfold": out.get("kfold"),
        "consensus": out.get("consensus"),
        "test_passed": test_passed,
        "test_total": test_total,
        "code": out.get("code", ""),
        "features": features,
        "rules": rules,
        "level": level,
        "synth_tokens": synth.synth.usage.input_tokens + synth.synth.usage.output_tokens,
        "candidate_tokens": synth.candidate.usage.input_tokens + synth.candidate.usage.output_tokens,
        "synth_cost": synth.synth.cost_so_far(),
        "candidate_cost": synth.candidate.cost_so_far(),
    }


def _task_features(task: dict) -> dict:
    """Extract a sparse feature vector from a task for compounding retrieval."""
    grids = [tr["input"] for tr in task.get("train", [])]
    grids += [tr["output"] for tr in task.get("train", [])]
    if not task.get("train"):
        return {}
    features = {}
    # grid dimensions
    h_set, w_set = set(), set()
    color_set = set()
    for g in grids:
        if g and g[0]:
            h_set.add(len(g))
            w_set.add(len(g[0]))
            for row in g:
                color_set.update(row)
    features["n_colors"] = len(color_set)
    features["max_h"] = max(h_set) if h_set else 0
    features["max_w"] = max(w_set) if w_set else 0
    features["n_train"] = len(task.get("train", []))
    features["n_test"] = len(task.get("test", []))
    # color frequency signature
    for c in sorted(color_set)[:10]:
        features[f"color_{c}"] = 1.0
    return features


def _task_level(task: dict) -> int | None:
    """Infer task level from metadata if available."""
    meta = task.get("metadata", {})
    return meta.get("level")


def run_eval(cfg: HarnessConfig, subset: str = "v1", limit: int = 5) -> dict:
    """Evaluate the synthesizer over `limit` tasks from `subset` (v1 or v2)."""
    os.makedirs(cfg.run_dir, exist_ok=True)
    task_ids = _target_subset(cfg, subset, limit)
    t0 = time.time()

    # compounding store: verified results compound across tasks
    store = CompoundingStore(os.path.join(cfg.run_dir, "compounding"))

    results = []
    with ProcessPoolExecutor(max_workers=cfg.num_workers) as pool:
        futs = {pool.submit(_worker_solve, cfg, tid): tid for tid in task_ids}
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            solved = sum(1 for x in results if x["status"] == "solved")

            # compound: record verified result
            store.record_task(
                task_id=r["task_id"],
                solved=r["status"] == "solved",
                code=r.get("code", ""),
                features=r.get("features"),
                rules=r.get("rules"),
                level=r.get("level"),
                config={"synth_model": cfg.synth_model, "candidate_model": cfg.candidate_model},
                cost_usd=r["synth_cost"] + r["candidate_cost"],
                time_s=r["time_s"],
                test_passed=bool(r.get("test_passed") == r.get("test_total") and r.get("test_total")),
            )

            extra = ""
            if r["status"] == "rejected":
                reason = (r.get("kfold") or {}).get("reject_reason") or (r.get("consensus") or {}).get("reject_reason")
                if reason:
                    extra = f"  [{reason}]"
            cap = store.capability
            print(
                f"[{time.strftime('%H:%M:%S')}] {r['task_id']}: {r['status']} "
                f"(train {r['score']:.2f}) {r['time_s']:.0f}s  "
                f"acc={solved}/{len(results)}  cap={cap:.3f}{extra}",
                flush=True,
            )
    results.sort(key=lambda x: x["task_id"])
    solved = sum(1 for r in results if r["status"] == "solved")
    rejected = sum(1 for r in results if r["status"] == "rejected")
    test_solved = sum(1 for r in results if r["test_passed"] == r["test_total"] and r["test_total"])
    report = {
        "subset": subset,
        "n": len(task_ids),
        "solved": solved,
        "kfold_rejected": rejected,
        "accuracy": solved / len(task_ids) if task_ids else 0.0,
        "test_accuracy": test_solved / len(task_ids) if task_ids else 0.0,
        "elapsed_s": round(time.time() - t0, 1),
        "usage": {
            "synth_tokens": round(sum(r["synth_tokens"] for r in results)),
            "candidate_tokens": round(sum(r["candidate_tokens"] for r in results)),
            "cost_usd": round(sum(r["synth_cost"] + r["candidate_cost"] for r in results), 4),
        },
        "compounding": store.summary(),
        "results": results,
    }
    report_path = os.path.join(cfg.run_dir, f"eval_{subset}_{int(time.time())}.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved: {report_path}", flush=True)
    return report
