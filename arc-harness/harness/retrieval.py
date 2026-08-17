"""Retrieve relevant library helpers for a given task.

Matches on structural grid features (dims, colors, connected-component
stats) so the prompt only includes the few helpers most likely to apply.
"""

from __future__ import annotations

import json
import os

from .config import HarnessConfig
from .grids import Grid, colors_used, grid_dims


def _features(grids: list[Grid]) -> dict:
    colors = set()
    sizes = []
    for g in grids:
        colors.update(colors_used(g))
        sizes.append(grid_dims(g))
    return {
        "colors": tuple(sorted(colors)),
        "n_colors": len(colors),
        "max_h": max(h for h, w in sizes),
        "max_w": max(w for h, w in sizes),
        "n_grids": len(grids),
    }


def _score(a: dict, b: dict) -> float:
    """Feature-overlap score between a task and a library task."""
    s = 0.0
    inter = set(a["colors"]) & set(b["colors"])
    union = set(a["colors"]) | set(b["colors"])
    if union:
        s += 3.0 * len(inter) / len(union)  # color palette overlap (weighted high)
    s += 1.0 if abs(a["n_colors"] - b["n_colors"]) <= 1 else 0.0
    s += 1.0 if abs(a["max_h"] - b["max_h"]) <= 3 else 0.0
    s += 1.0 if abs(a["max_w"] - b["max_w"]) <= 3 else 0.0
    s += 1.0 if a["n_grids"] == b["n_grids"] else 0.0
    return s


class LibraryIndex:
    """In-memory index over the extracted concept library."""

    def __init__(self, cfg: HarnessConfig):
        self.cfg = cfg
        self.entries = self._load_index()
        self._task_cache = {}

    def _load_index(self) -> list[dict]:
        idx_path = os.path.join(self.cfg.library_root, "index.json")
        if os.path.exists(idx_path):
            return json.load(open(idx_path))
        return []

    def _task_features(self, task_id: str) -> dict:
        if task_id not in self._task_cache:
            from .dataset import load_task

            task = load_task(self.cfg, task_id)
            self._task_cache[task_id] = _features(
                [p["input"] for p in task["train"]] + [p["output"] for p in task["train"]]
            )
        return self._task_cache[task_id]

    def top_k_for_task(self, task_id: str, k: int = 3) -> list[dict]:
        """Return the k most relevant library entries for a task, deduped by name.
        Excludes helpers mined from the task's own solver to avoid trivial copying."""
        target = self._task_features(task_id)
        scored = []
        seen_names = set()
        for e in self.entries:
            if e["task_id"] == task_id:
                continue  # self-exclusion
            if e["name"] in seen_names:
                continue
            feats = self._task_features(e["task_id"])
            score = _score(target, feats)
            scored.append((score, e))
            seen_names.add(e["name"])
        scored.sort(key=lambda x: -x[0])
        return [e for s, e in scored[:k] if s > 0]
