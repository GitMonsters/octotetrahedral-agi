"""System 2 → System 1 knowledge transfer cache.

When System 2 (slow deliberation) produces a high-confidence output,
store a fingerprint of the input and the resulting hidden state so
System 1 (fast path) can retrieve and blend it on similar future inputs.

Based on Ye et al. 2022 'Parallel Cognition', Section 3.1:
'explicit knowledge in system 2 tends to be converted into implicit
knowledge in system 1, because it saves energy.'
"""

from __future__ import annotations

import time
from collections import OrderedDict
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import torch


class S2ToS1Cache:
    """LRU cache that converts high-confidence System-2 outputs into
    System-1 priors, reducing deliberation energy on repeated inputs.

    Args:
        capacity: Maximum number of entries to keep (LRU eviction).
        confidence_threshold: Minimum slow-path confidence to store.
    """

    def __init__(self, capacity: int = 512, confidence_threshold: float = 0.75) -> None:
        self.capacity: int = capacity
        self.confidence_threshold: float = confidence_threshold
        # OrderedDict gives O(1) LRU eviction: most-recently-used at end.
        self._store: OrderedDict[int, tuple] = OrderedDict()
        self._hits: int = 0
        self._misses: int = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fingerprint(self, input_ids: "torch.Tensor") -> int:
        """Cheap locality-sensitive fingerprint based on token sum."""
        return hash(input_ids.sum().item())

    def _is_near(self, fp_a: int, fp_b: int, pct: float = 0.10) -> bool:
        """Return True if two fingerprints are within *pct* of each other."""
        if fp_a == 0 and fp_b == 0:
            return True
        denom = max(abs(fp_a), abs(fp_b), 1)
        return abs(fp_a - fp_b) / denom <= pct

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def store(
        self,
        input_ids: "torch.Tensor",
        hidden: "torch.Tensor",
        confidence: float,
        path: str,
    ) -> None:
        """Store a slow-path hidden state if confidence is high enough.

        Args:
            input_ids: Raw token IDs (used only for fingerprinting).
            hidden: Hidden-state tensor to cache [B, L, D] or [B, D].
            confidence: Scalar confidence in [0, 1].
            path: 'slow' or 'fast' — only 'slow' entries are stored.
        """
        if path != "slow" or confidence < self.confidence_threshold:
            return

        fp = self._fingerprint(input_ids)

        # Evict oldest entry when at capacity (LRU)
        while len(self._store) >= self.capacity:
            self._store.popitem(last=False)

        # Move existing entry to end (most-recent) on collision
        if fp in self._store:
            self._store.move_to_end(fp)

        self._store[fp] = (fp, hidden.detach().cpu().float(), confidence, time.monotonic())

    def retrieve(
        self,
        input_ids: "torch.Tensor",
        top_k: int = 3,
    ) -> "Optional[torch.Tensor]":
        """Look up cached System-2 hidden states for a given input.

        Matches on exact fingerprint first, then near-misses (within 10%).

        Args:
            input_ids: Current token IDs to look up.
            top_k: Return mean of the top-k highest-confidence matches.

        Returns:
            Mean of matched hiddens (on CPU, caller responsible for device
            placement), or ``None`` if no matches found.
        """
        import torch

        fp = self._fingerprint(input_ids)

        candidates: list[tuple[float, "torch.Tensor"]] = []
        for stored_fp, (_, hidden, conf, _ts) in self._store.items():
            if stored_fp == fp or self._is_near(fp, stored_fp):
                candidates.append((conf, hidden))

        if not candidates:
            self._misses += 1
            return None

        self._hits += 1
        # Sort descending by confidence and take top-k
        candidates.sort(key=lambda c: c[0], reverse=True)
        top = [h for _, h in candidates[:top_k]]
        return torch.stack(top).mean(dim=0)

    def stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        total = self._hits + self._misses
        stored_confs = [c for _, (_, _, c, _) in self._store.items()]
        return {
            "size": len(self._store),
            "capacity": self.capacity,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
            "avg_confidence_stored": sum(stored_confs) / len(stored_confs) if stored_confs else 0.0,
        }

    def clear(self) -> None:
        """Evict all entries and reset statistics."""
        self._store.clear()
        self._hits = 0
        self._misses = 0
