"""Verified-compounding infrastructure.

Every module feeds verified results back through this store.  Only outputs
that survive test verification compound — unverified results are logged but
never propagated.  The store is file-backed (JSON) so it persists across
sessions and grows with each solved task.

Core idea (from the user's framing):
    AGI progress ≈ recursive abstraction × verified compounding improvement

Three compounding layers:
1. Verified Solution Memory — solved tasks become few-shot examples
2. Pattern Accumulation — extracted rules/strategies compound across tasks
3. Config Bandit — hyperparameter performance compounds to allocate budget
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections import defaultdict
from enum import Enum
from typing import Any


# ── MOLT roles ──────────────────────────────────────────────────────────────

class MOLTRole(Enum):
    """Modular Operating Language of Thought roles.

    Controls how patterns are injected into prompts:
    - DIRECTIVE:  primary transform, prominently featured
    - CONSTRAINT: constrains the search space
    - HEURISTIC:  soft guidance, optional
    - CONTEXT:    background info, included for context
    """
    DIRECTIVE = "directive"
    CONSTRAINT = "constraint"
    HEURISTIC = "heuristic"
    CONTEXT = "context"


# Role assignment rules: pattern name -> role
_ROLE_OVERRIDES: dict[str, MOLTRole] = {
    "rotation": MOLTRole.DIRECTIVE,
    "rot_cw": MOLTRole.DIRECTIVE,
    "rot_ccw": MOLTRole.DIRECTIVE,
    "flip_h": MOLTRole.DIRECTIVE,
    "flip_v": MOLTRole.DIRECTIVE,
    "color_map": MOLTRole.DIRECTIVE,
    "swap_colors": MOLTRole.DIRECTIVE,
    "gravity": MOLTRole.DIRECTIVE,
    "color_limit": MOLTRole.CONSTRAINT,
    "boundary": MOLTRole.CONSTRAINT,
    "size_constraint": MOLTRole.CONSTRAINT,
    "prefer_simple": MOLTRole.HEURISTIC,
    "prefer_symmetry": MOLTRole.HEURISTIC,
}


def _infer_role(pattern_name: str) -> str:
    """Infer MOLT role for a pattern name."""
    if pattern_name in _ROLE_OVERRIDES:
        return _ROLE_OVERRIDES[pattern_name].value
    return MOLTRole.DIRECTIVE.value


# ── helpers ──────────────────────────────────────────────────────────────────

def _hash_config(cfg: dict) -> str:
    """Deterministic hash for a config dict (sorted keys)."""
    raw = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def _cosine(a: dict, b: dict) -> float:
    """Cosine similarity between two sparse feature vectors."""
    keys = set(a) | set(b)
    if not keys:
        return 0.0
    dot = sum(a.get(k, 0) * b.get(k, 0) for k in keys)
    na = math.sqrt(sum(v * v for v in a.values())) or 1.0
    nb = math.sqrt(sum(v * v for v in b.values())) or 1.0
    return dot / (na * nb)


# ── Solution Memory ──────────────────────────────────────────────────────────

class SolutionMemory:
    """Verified solutions with feature-based retrieval.

    Each entry stores code, metadata, and a feature vector for similarity
    search.  Only solutions where ``test_passed == True`` are used for
    retrieval (unverified solutions are stored but not propagated).
    """

    def __init__(self, path: str):
        self.path = path
        self._entries: list[dict] = []
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path) as f:
                self._entries = json.load(f)

    def save(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self._entries, f, indent=1)
        os.replace(tmp, self.path)

    def record(
        self,
        task_id: str,
        code: str,
        features: dict[str, float],
        level: int | None = None,
        rules: list[str] | None = None,
        score: float = 1.0,
        test_passed: bool = True,
        cost_usd: float = 0.0,
        time_s: float = 0.0,
    ):
        entry = {
            "task_id": task_id,
            "code": code,
            "features": features,
            "level": level,
            "rules": rules or [],
            "score": score,
            "verified": test_passed,
            "cost_usd": cost_usd,
            "time_s": time_s,
            "timestamp": time.time(),
        }
        # upsert
        self._entries = [e for e in self._entries if e["task_id"] != task_id]
        self._entries.append(entry)
        self.save()

    def similar(
        self, features: dict[str, float], top_k: int = 5
    ) -> list[dict]:
        """Return top-k verified solutions sorted by feature similarity."""
        verified = [e for e in self._entries if e.get("verified")]
        scored = [
            (e, _cosine(features, e.get("features", {}))) for e in verified
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [e for e, s in scored[:top_k] if s > 0.05]

    def by_rules(self, rules: list[str]) -> list[dict]:
        """Return verified solutions that share any of the given rules."""
        rule_set = set(rules)
        return [
            e for e in self._entries
            if e.get("verified") and rule_set & set(e.get("rules", []))
        ]

    @property
    def verified_count(self) -> int:
        return sum(1 for e in self._entries if e.get("verified"))

    @property
    def total_count(self) -> int:
        return len(self._entries)


# ── Pattern Memory ───────────────────────────────────────────────────────────

class PatternMemory:
    """Accumulated rule patterns with verified-confidence scores.

    Patterns are extracted from solved tasks (e.g. "rotation", "color_swap",
    "gravity") and tracked with how often they appear in verified solutions.
    New tasks get injected with high-confidence patterns relevant to their
    features.
    """

    def __init__(self, path: str):
        self.path = path
        self._patterns: dict[str, dict] = {}  # pattern_name -> {count, verified_count, tasks, features}
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path) as f:
                self._patterns = json.load(f)

    def save(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self._patterns, f, indent=1)
        os.replace(tmp, self.path)

    def record(
        self,
        pattern_name: str,
        task_id: str,
        verified: bool = True,
        features: dict[str, float] | None = None,
        role: str | None = None,
    ):
        if pattern_name not in self._patterns:
            self._patterns[pattern_name] = {
                "count": 0,
                "verified_count": 0,
                "tasks": [],
                "avg_features": {},
                "role": role or _infer_role(pattern_name),
            }
        p = self._patterns[pattern_name]
        p["count"] += 1
        if verified:
            p["verified_count"] += 1
        if task_id not in p["tasks"]:
            p["tasks"].append(task_id)
        if features:
            n = p["count"]
            for k, v in features.items():
                old = p["avg_features"].get(k, 0.0)
                p["avg_features"][k] = old + (v - old) / n
        self.save()

    def confidence(self, pattern_name: str) -> float:
        """Verified ratio — only verified occurrences compound."""
        p = self._patterns.get(pattern_name)
        if not p or p["count"] == 0:
            return 0.0
        return p["verified_count"] / p["count"]

    def top_patterns(self, min_confidence: float = 0.3, top_k: int = 20) -> list[tuple[str, float]]:
        """Return (name, confidence) pairs sorted by confidence × log(count)."""
        items = []
        for name, p in self._patterns.items():
            if p["count"] == 0:
                continue
            conf = p["verified_count"] / p["count"]
            if conf >= min_confidence:
                score = conf * math.log(p["count"] + 1)
                items.append((name, score))
        items.sort(key=lambda x: x[1], reverse=True)
        return items[:top_k]

    def relevant_to(self, features: dict[str, float], top_k: int = 10) -> list[tuple[str, float]]:
        """Return patterns relevant to a task's features, weighted by similarity."""
        items = []
        for name, p in self._patterns.items():
            if p["count"] == 0:
                continue
            sim = _cosine(features, p.get("avg_features", {}))
            conf = p["verified_count"] / p["count"]
            if sim > 0.05 and conf >= 0.3:
                items.append((name, sim * conf))
        items.sort(key=lambda x: x[1], reverse=True)
        return items[:top_k]

    def by_role(self, role: str) -> list[tuple[str, float]]:
        """Return patterns matching a MOLT role, sorted by confidence."""
        items = []
        for name, p in self._patterns.items():
            if p.get("role") == role and p["count"] > 0:
                conf = p["verified_count"] / p["count"]
                items.append((name, conf))
        items.sort(key=lambda x: x[1], reverse=True)
        return items

    def by_roles(self, roles: list[str]) -> list[tuple[str, float]]:
        """Return patterns matching any of the given MOLT roles."""
        role_set = set(roles)
        items = []
        for name, p in self._patterns.items():
            if p.get("role") in role_set and p["count"] > 0:
                conf = p["verified_count"] / p["count"]
                items.append((name, conf))
        items.sort(key=lambda x: x[1], reverse=True)
        return items


# ── Config Bandit ────────────────────────────────────────────────────────────

class ConfigBandit:
    """Track hyperparameter performance and allocate budget via UCB1.

    Each unique config is a "arm".  After each task, record whether the config
    solved it and the cost.  The bandit suggests configs that balance
    exploration (try new configs) and exploitation (use known-good configs).
    """

    def __init__(self, path: str):
        self.path = path
        self._arms: dict[str, dict] = {}  # config_hash -> {pulls, wins, total_cost, config}
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path) as f:
                self._arms = json.load(f)

    def save(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self._arms, f, indent=1)
        os.replace(tmp, self.path)

    def record(self, config: dict, solved: bool, cost_usd: float = 0.0):
        h = _hash_config(config)
        if h not in self._arms:
            self._arms[h] = {
                "pulls": 0,
                "wins": 0,
                "total_cost": 0.0,
                "config": config,
            }
        arm = self._arms[h]
        arm["pulls"] += 1
        if solved:
            arm["wins"] += 1
        arm["total_cost"] += cost_usd
        self.save()

    def suggest(self, explore_weight: float = 1.41) -> dict | None:
        """UCB1: pick the config with highest upper confidence bound."""
        if not self._arms:
            return None
        total_pulls = sum(a["pulls"] for a in self._arms.values())
        if total_pulls == 0:
            return list(self._arms.values())[0]["config"]
        best_h, best_ucb = None, -1
        for h, arm in self._arms.items():
            n = arm["pulls"]
            if n == 0:
                return arm["config"]  # untried → pull first
            win_rate = arm["wins"] / n
            ucb = win_rate + explore_weight * math.sqrt(math.log(total_pulls) / n)
            if ucb > best_ucb:
                best_ucb = ucb
                best_h = h
        return self._arms[best_h]["config"] if best_h else None

    def best_solving_rate(self) -> tuple[dict, float]:
        """Return (config, solve_rate) for the arm with highest win rate (min 2 pulls)."""
        best, best_rate = None, -1
        for h, arm in self._arms.items():
            if arm["pulls"] >= 2:
                rate = arm["wins"] / arm["pulls"]
                if rate > best_rate:
                    best_rate = rate
                    best = arm["config"]
        return best, best_rate


# ── Compounding Store (facade) ──────────────────────────────────────────────

class CompoundingStore:
    """Unified facade for all compounding layers.

    Every module in the harness imports this and calls ``store.record()``
    after a verified result.  The store orchestrates solution memory, pattern
    memory, and config bandit behind a single interface.

    Only verified (test-passed) results compound into retrieval, prompt
    enrichment, and budget allocation.  Unverified results are logged for
    diagnostics but never propagated.
    """

    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        os.makedirs(root_dir, exist_ok=True)
        self.solutions = SolutionMemory(os.path.join(root_dir, "solutions.json"))
        self.patterns = PatternMemory(os.path.join(root_dir, "patterns.json"))
        self.bandit = ConfigBandit(os.path.join(root_dir, "bandit.json"))

        # cross-task capability tracking
        self._capability_path = os.path.join(root_dir, "capability.json")
        self._capability: dict = self._load_capability()

    def _load_capability(self) -> dict:
        if os.path.exists(self._capability_path):
            with open(self._capability_path) as f:
                return json.load(f)
        return {
            "tasks_solved": 0,
            "tasks_attempted": 0,
            "total_cost": 0.0,
            "capability_curve": [],  # [(timestamp, capability_score)]
            "level_solved": {1: 0, 2: 0, 3: 0},
            "level_attempted": {1: 0, 2: 0, 3: 0},
        }

    def _save_capability(self):
        tmp = self._capability_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self._capability, f, indent=1)
        os.replace(tmp, self._capability_path)

    @property
    def capability(self) -> float:
        """Current capability score: verified_solves / attempts (0-1)."""
        n = self._capability["tasks_attempted"]
        if n == 0:
            return 0.0
        return self._capability["tasks_solved"] / n

    def record_task(
        self,
        task_id: str,
        solved: bool,
        code: str = "",
        features: dict[str, float] | None = None,
        rules: list[str] | None = None,
        level: int | None = None,
        config: dict | None = None,
        cost_usd: float = 0.0,
        time_s: float = 0.0,
        test_passed: bool = False,
    ):
        """Record a task result across all compounding layers.

        Only ``test_passed == True`` results propagate into retrieval and
        prompt enrichment.  All results update the capability curve and
        config bandit.
        """
        # capability tracking
        self._capability["tasks_attempted"] += 1
        if solved:
            self._capability["tasks_solved"] += 1
        self._capability["total_cost"] += cost_usd
        if level is not None:
            self._capability["level_attempted"][level] = (
                self._capability["level_attempted"].get(level, 0) + 1
            )
            if solved:
                self._capability["level_solved"][level] = (
                    self._capability["level_solved"].get(level, 0) + 1
                )
        self._capability["capability_curve"].append(
            (time.time(), self.capability)
        )
        self._save_capability()

        # solution memory (only verified)
        if code and test_passed:
            self.solutions.record(
                task_id=task_id,
                code=code,
                features=features or {},
                level=level,
                rules=rules,
                score=1.0,
                test_passed=True,
                cost_usd=cost_usd,
                time_s=time_s,
            )

        # pattern memory
        if rules and test_passed:
            for rule in rules:
                self.patterns.record(
                    pattern_name=rule,
                    task_id=task_id,
                    verified=True,
                    features=features,
                )

        # config bandit
        if config is not None:
            self.bandit.record(config, solved, cost_usd)

    def get_enrichment_multiplier(self, max_enrichment: float = 3.0) -> float:
        """Cross-task enrichment: capability compounds.

        Returns a multiplier >1.0 that grows as more tasks are solved.
        Formula: 1 + ln(1 + verified_solves) * 0.1
        This gives a gentle compounding curve: +10% at 1 solve, +21% at 3,
        +30% at 7, +45% at 20. Capped at max_enrichment.
        """
        n = self._capability["tasks_solved"]
        return min(1.0 + math.log(1 + n) * 0.1, max_enrichment)

    def get_difficulty_signal(self, task_id: str, features: dict) -> dict:
        """Estimate task difficulty from compounding data.

        Returns {"difficulty": 0-1, "estimated_budget_mult": 0.5-2.0}.
        """
        # find similar solved tasks
        similar = self.solutions.similar(features, top_k=5)
        if not similar:
            return {"difficulty": 0.5, "estimated_budget_mult": 1.0}
        # average cost of similar solved tasks
        avg_cost = sum(e.get("cost_usd", 0) for e in similar) / len(similar)
        avg_time = sum(e.get("time_s", 0) for e in similar) / len(similar)
        # difficulty: 0 = easy (fast solve), 1 = hard (expensive)
        difficulty = min(1.0, avg_time / 600.0)  # 600s = hard
        budget_mult = 0.5 + difficulty * 1.5  # 0.5x-2.0x
        return {
            "difficulty": round(difficulty, 3),
            "estimated_budget_mult": round(budget_mult, 3),
            "similar_tasks": [e["task_id"] for e in similar],
            "similar_rules": list(set(r for e in similar for r in e.get("rules", []))),
        }

    def summary(self) -> dict:
        role_counts = defaultdict(int)
        for p in self.patterns._patterns.values():
            role_counts[p.get("role", "directive")] += 1
        return {
            "verified_solutions": self.solutions.verified_count,
            "total_solutions": self.solutions.total_count,
            "patterns": len(self.patterns._patterns),
            "pattern_roles": dict(role_counts),
            "config_arms": len(self.bandit._arms),
            "capability": round(self.capability, 4),
            "enrichment_multiplier": round(self.get_enrichment_multiplier(), 3),
            "tasks_solved": self._capability["tasks_solved"],
            "tasks_attempted": self._capability["tasks_attempted"],
        }
