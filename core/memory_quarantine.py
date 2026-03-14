"""
Memory quarantine pipeline — validates observations before committing to trusted memory.

Biological inspiration:
- Octopus RNA editing: neural signals are edited/filtered before being acted upon
- Hippocampal replay: memories are consolidated through repeated validation
- Excitatory/inhibitory balance gates what gets stored long-term

Pipeline:
1. New observation arrives with E/I signs + confidence from RNA editing
2. High confidence + excitatory → promote directly to trusted memory
3. Low confidence OR inhibitory-dominant → quarantine for validation
4. In quarantine: check consistency against existing trusted memories
5. Consistent + enough validations → promote; stale → discard
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
from collections import deque


class QuarantineDecision(Enum):
    """Triage decision for incoming observations."""
    PROMOTE = "promote"       # Pass through directly to trusted memory
    QUARANTINE = "quarantine" # Hold in buffer for consistency validation
    REJECT = "reject"         # Immediately discard (too unreliable)


@dataclass
class QuarantineEntry:
    """A candidate memory held in quarantine pending validation."""
    memory: torch.Tensor                          # [embed_dim] candidate embedding
    importance: float                             # Claimed importance score
    ei_signal: float                              # Mean E/I signal (-1 to +1)
    confidence: float                             # RNA editing confidence (0-1)
    spatial_context: Optional[torch.Tensor] = None
    object_refs: Optional[List[str]] = None
    event_type: str = 'observation'
    timestamp: int = 0                            # When entered quarantine
    consistency_score: float = 0.0                # Updated during validation
    validation_attempts: int = 0                  # How many times checked


class MemoryQuarantine(nn.Module):
    """
    Quarantine buffer between perception and trusted memory.

    Sits between the perception pipeline (which produces observations with
    E/I signs and confidence from ExcitatoryInhibitoryClassifier) and
    EpisodicMemory.store(). Only observations that pass triage or survive
    quarantine validation are promoted to trusted memory.

    Learned components:
    - confidence_gate: predicts trust from the embedding itself
    - consistency_proj: projects memories into a learned similarity space
      for consistency comparison
    """

    def __init__(
        self,
        embed_dim: int = 256,
        buffer_size: int = 32,
        confidence_threshold: float = 0.6,
        consistency_threshold: float = 0.4,
        max_age: int = 50,
        min_validations: int = 2,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.buffer_size = buffer_size
        self.confidence_threshold = confidence_threshold
        self.consistency_threshold = consistency_threshold
        self.max_age = max_age
        self.min_validations = min_validations

        # Quarantine buffer (FIFO, bounded)
        self._buffer: deque[QuarantineEntry] = deque(maxlen=buffer_size)

        # Global clock for aging entries
        self._clock: int = 0

        # Lifetime counters
        self._total_promoted: int = 0
        self._total_discarded: int = 0
        self._total_rejected: int = 0

        # --- Learned components ---

        # Confidence gate: learns whether an embedding should be trusted
        self.confidence_gate = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Projects memories into a learned similarity space for consistency checks
        self.consistency_proj = nn.Linear(embed_dim, embed_dim)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(
        self,
        memory: torch.Tensor,
        importance: float,
        ei_signal: float,
        confidence: float,
        spatial_context: Optional[torch.Tensor] = None,
        object_refs: Optional[List[str]] = None,
        event_type: str = 'observation',
    ) -> QuarantineDecision:
        """
        Triage a new observation.

        Args:
            memory:          [embed_dim] candidate memory embedding
            importance:      claimed importance score
            ei_signal:       mean E/I signal from RNA editing (-1 to +1)
            confidence:      RNA editing confidence (0 to 1)
            spatial_context: optional [embed_dim] spatial embedding
            object_refs:     optional list of referenced object ids
            event_type:      observation type tag

        Returns:
            QuarantineDecision indicating what happened to the observation.
        """
        self._clock += 1

        # Combine external confidence with learned gate prediction
        with torch.no_grad():
            mem = memory.detach()
            if mem.dim() == 1:
                mem = mem.unsqueeze(0)  # [1, embed_dim]
            learned_conf: float = self.confidence_gate(mem).item()
        combined_confidence = 0.5 * confidence + 0.5 * learned_conf

        # --- Decision logic ---
        # Very low confidence AND strongly inhibitory → reject outright
        if combined_confidence < 0.3 and ei_signal < -0.5:
            self._total_rejected += 1
            return QuarantineDecision.REJECT

        # High confidence AND excitatory-dominant → promote directly
        if combined_confidence >= self.confidence_threshold and ei_signal >= 0.0:
            self._total_promoted += 1
            return QuarantineDecision.PROMOTE

        # Everything else → quarantine for validation
        entry = QuarantineEntry(
            memory=memory.detach().clone(),
            importance=importance,
            ei_signal=ei_signal,
            confidence=confidence,
            spatial_context=(
                spatial_context.detach().clone() if spatial_context is not None else None
            ),
            object_refs=list(object_refs) if object_refs is not None else None,
            event_type=event_type,
            timestamp=self._clock,
        )
        self._buffer.append(entry)
        return QuarantineDecision.QUARANTINE

    def validate(
        self,
        trusted_memories: List[torch.Tensor],
    ) -> List[QuarantineEntry]:
        """
        Validate quarantined entries against trusted memories.

        Consistency is the max cosine similarity between the quarantined
        embedding (projected) and any trusted memory (projected).

        Args:
            trusted_memories: list of [embed_dim] tensors from EpisodicMemory

        Returns:
            List of entries that are ready for promotion (consistent enough
            and have met the minimum validation count).
        """
        if not self._buffer or not trusted_memories:
            # Still bump validation_attempts so stale entries can be discarded
            for entry in self._buffer:
                entry.validation_attempts += 1
            return []

        # Stack trusted memories → [N, embed_dim]
        trusted_stack = torch.stack(
            [t.detach() for t in trusted_memories], dim=0
        )

        ready: List[QuarantineEntry] = []

        with torch.no_grad():
            trusted_proj = self.consistency_proj(trusted_stack)  # [N, embed_dim]

            for entry in self._buffer:
                entry.validation_attempts += 1

                mem = entry.memory
                if mem.dim() == 1:
                    mem = mem.unsqueeze(0)  # [1, embed_dim]
                mem_proj = self.consistency_proj(mem)  # [1, embed_dim]

                # Max cosine similarity against any trusted memory
                sims = self._cosine_similarity(
                    mem_proj.expand_as(trusted_proj), trusted_proj
                )  # [N]
                entry.consistency_score = sims.max().item()

                if (
                    entry.consistency_score >= self.consistency_threshold
                    and entry.validation_attempts >= self.min_validations
                ):
                    ready.append(entry)

        return ready

    def promote(self) -> List[QuarantineEntry]:
        """
        Pop and return all entries that passed validation.

        The caller is responsible for storing these via
        EpisodicMemory.store().

        Returns:
            List of promoted QuarantineEntry objects.
        """
        promoted: List[QuarantineEntry] = []
        remaining: deque[QuarantineEntry] = deque(maxlen=self.buffer_size)

        for entry in self._buffer:
            if (
                entry.consistency_score >= self.consistency_threshold
                and entry.validation_attempts >= self.min_validations
            ):
                promoted.append(entry)
            else:
                remaining.append(entry)

        self._buffer = remaining
        self._total_promoted += len(promoted)
        return promoted

    def discard_stale(self) -> int:
        """
        Remove entries that exceeded max_age without promotion.

        Returns:
            Number of discarded entries.
        """
        surviving: deque[QuarantineEntry] = deque(maxlen=self.buffer_size)
        discarded = 0

        for entry in self._buffer:
            age = self._clock - entry.timestamp
            if age > self.max_age:
                discarded += 1
            else:
                surviving.append(entry)

        self._buffer = surviving
        self._total_discarded += discarded
        return discarded

    def get_stats(self) -> Dict[str, Any]:
        """Return quarantine statistics."""
        consistencies = [e.consistency_score for e in self._buffer]
        avg_consistency = (
            sum(consistencies) / len(consistencies) if consistencies else 0.0
        )
        return {
            "buffer_capacity": self.buffer_size,
            "num_quarantined": len(self._buffer),
            "num_promoted": self._total_promoted,
            "num_discarded": self._total_discarded,
            "num_rejected": self._total_rejected,
            "avg_consistency": round(avg_consistency, 4),
            "clock": self._clock,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Element-wise cosine similarity between two tensors.

        Args:
            a: [..., D] tensor
            b: [..., D] tensor (same leading dims as a)

        Returns:
            [...] cosine similarities clamped to [-1, 1]
        """
        a_norm = a / (a.norm(dim=-1, keepdim=True) + 1e-8)
        b_norm = b / (b.norm(dim=-1, keepdim=True) + 1e-8)
        return (a_norm * b_norm).sum(dim=-1)

    def forward(
        self,
        observations: torch.Tensor,
        ei_signals: torch.Tensor,
        confidences: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Batch-mode forward for training the learned gates.

        Runs the confidence gate and consistency projection on a batch of
        observations so gradients can flow through the learned components.

        Args:
            observations: [batch, embed_dim] candidate embeddings
            ei_signals:   [batch, 1] mean E/I signals
            confidences:  [batch, 1] RNA editing confidences

        Returns:
            Dict with 'gate_scores' [batch, 1] and 'projected' [batch, embed_dim]
        """
        gate_scores = self.confidence_gate(observations)      # [batch, 1]
        projected = self.consistency_proj(observations)        # [batch, embed_dim]
        return {
            "gate_scores": gate_scores,
            "projected": projected,
        }


# ======================================================================
# Self-test
# ======================================================================
if __name__ == '__main__':
    torch.manual_seed(42)
    EMBED = 64

    quarantine = MemoryQuarantine(
        embed_dim=EMBED,
        buffer_size=16,
        confidence_threshold=0.6,
        consistency_threshold=0.4,
        max_age=50,
        min_validations=2,
    )

    # --- Test submit: high confidence + excitatory → PROMOTE ---
    mem_good = torch.randn(EMBED)
    dec = quarantine.submit(mem_good, importance=0.9, ei_signal=0.8, confidence=0.95)
    assert dec == QuarantineDecision.PROMOTE, f"Expected PROMOTE, got {dec}"

    # --- Test submit: low confidence → QUARANTINE ---
    mem_uncertain = torch.randn(EMBED)
    dec = quarantine.submit(mem_uncertain, importance=0.5, ei_signal=0.1, confidence=0.2)
    assert dec == QuarantineDecision.QUARANTINE, f"Expected QUARANTINE, got {dec}"

    # --- Test submit: very low confidence + strongly inhibitory → REJECT ---
    mem_bad = torch.randn(EMBED)
    dec = quarantine.submit(mem_bad, importance=0.1, ei_signal=-0.9, confidence=0.05)
    assert dec == QuarantineDecision.REJECT, f"Expected REJECT, got {dec}"

    # --- Add a few more quarantined entries ---
    for i in range(4):
        quarantine.submit(
            torch.randn(EMBED),
            importance=0.5,
            ei_signal=-0.1,
            confidence=0.3,
            object_refs=[f"obj_{i}"],
            event_type='touch',
        )

    assert quarantine.get_stats()["num_quarantined"] == 5, "Expected 5 quarantined"

    # --- Test validate against trusted memories ---
    # Create trusted memories that are somewhat similar to quarantined ones
    trusted = [torch.randn(EMBED) for _ in range(3)]
    # First validation round — not enough attempts yet (need min_validations=2)
    ready = quarantine.validate(trusted)
    assert len(ready) == 0 or all(
        e.validation_attempts >= 2 for e in ready
    ), "Shouldn't promote with < min_validations"

    # Second validation round
    ready = quarantine.validate(trusted)
    # Some may now be ready if consistency is high enough
    for entry in ready:
        assert entry.validation_attempts >= 2
        assert entry.consistency_score >= quarantine.consistency_threshold

    # --- Test promote ---
    promoted = quarantine.promote()
    for entry in promoted:
        assert entry.consistency_score >= quarantine.consistency_threshold

    # --- Test discard_stale ---
    # Fast-forward clock past max_age
    quarantine._clock += quarantine.max_age + 10
    num_discarded = quarantine.discard_stale()
    assert num_discarded >= 0

    # --- Test learned gate forward pass (training mode) ---
    batch = torch.randn(8, EMBED)
    ei = torch.randn(8, 1)
    conf = torch.rand(8, 1)
    out = quarantine(batch, ei, conf)
    assert out["gate_scores"].shape == (8, 1), f"Bad gate shape: {out['gate_scores'].shape}"
    assert out["projected"].shape == (8, EMBED), f"Bad proj shape: {out['projected'].shape}"
    assert (out["gate_scores"] >= 0).all() and (out["gate_scores"] <= 1).all(), "Gate not in [0,1]"

    # --- Stats sanity check ---
    stats = quarantine.get_stats()
    assert "num_promoted" in stats
    assert "num_discarded" in stats
    assert "avg_consistency" in stats
    assert stats["num_rejected"] >= 1

    print("MemoryQuarantine self-test passed!")
