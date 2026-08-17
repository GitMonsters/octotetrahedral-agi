"""Load ARC-AGI tasks from the verified-solvers dataset and official splits."""

from __future__ import annotations

import json
import os

from .config import HarnessConfig


def load_task(cfg: HarnessConfig, task_id: str) -> dict:
    """Load a task by ID from dataset/tasks/{task_id}.json."""
    path = os.path.join(cfg.data_root, "tasks", f"{task_id}.json")
    if not os.path.exists(path):
        # fall back to the solvers repo's dataset
        path = os.path.join(cfg.solvers_root, "dataset", "tasks", f"{task_id}.json")
    with open(path) as f:
        return json.load(f)


def eval_ids(cfg: HarnessConfig, version: str) -> list[str]:
    """Return the official evaluation task IDs for ARC-AGI-1 or ARC-AGI-2."""
    fname = "v1_public_evaluation_set.json" if version == "1" else "v2_public_evaluation_set.json"
    for root in (cfg.data_root, cfg.solvers_root):
        p = os.path.join(root, "dataset", fname)
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f)
    raise FileNotFoundError(f"{fname} not found")


def setup_data_dir(cfg: HarnessConfig) -> None:
    """Symlink dataset/tasks from the solvers repo into data/ if missing."""
    os.makedirs(cfg.data_root, exist_ok=True)
    tasks_dir = os.path.join(cfg.data_root, "tasks")
    if not os.path.isdir(tasks_dir):
        src = os.path.join(cfg.solvers_root, "dataset", "tasks")
        if os.path.isdir(src):
            os.symlink(src, tasks_dir)
    for fname in ("v1_public_evaluation_set.json", "v2_public_evaluation_set.json"):
        dst = os.path.join(cfg.data_root, fname)
        if not os.path.exists(dst):
            src = os.path.join(cfg.solvers_root, "dataset", fname)
            if os.path.exists(src):
                os.symlink(src, dst)
