"""
core/knowledge_graph.py
Differentiable Knowledge Graph for OctoTetrahedral AGI.

Architecture:
  Entity Bank     : Learned entity embeddings  [num_entities × D]
  Relation Bank   : Learned relation embeddings [num_relations × D]
                    (each becomes one attention head in message passing)
  Message Passing : R hops of multi-relational entity self-attention (R-GAT)
  Soft Retrieval  : Given a query hidden state → attention-weighted entity sum
  Residual Fuse   : LayerNorm(x + project(KG-retrieved))

The KG enriches relational reasoning with a persistent, differentiable
entity store that is updated through backpropagation. An optional
EMA-write path (write_ema) blends the forward query into top-K entity
slots to incrementally ground entities in the current context.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class KGConfig:
    """Hyperparameters for the KnowledgeGraphModule."""
    enabled: bool = True

    # Entity / relation bank sizes
    num_entities: int = 256      # learnable entity slots
    num_relations: int = 8       # = num_mp_heads in message passing

    # Message-passing depth
    num_hops: int = 2

    # Retrieval top-K (for EMA write and diagnostic reporting)
    top_k_retrieval: int = 8

    # EMA write strength (0 = disabled; small value ≈ 0.02 recommended)
    ema_write_alpha: float = 0.02

    # Dropout
    dropout: float = 0.1


# ──────────────────────────────────────────────────────────────────────────────
# Relation-aware Message Passing (R-GAT single hop)
# ──────────────────────────────────────────────────────────────────────────────

class _RGATHop(nn.Module):
    """
    One hop of Relational Graph Attention.

    For each relation r (one attention head), compute softmax attention
    over all entity pairs and propagate messages.  The per-relation
    outputs are concatenated, projected back to D, and added as a
    residual to the input entity representations.
    """

    def __init__(self, hidden_dim: int, num_relations: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads=num_relations,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.ff_norm = nn.LayerNorm(hidden_dim)

    def forward(self, E: torch.Tensor) -> torch.Tensor:
        """
        Args:
            E: Entity representations  [B, num_entities, D]
        Returns:
            E': Updated entity representations  [B, num_entities, D]
        """
        msg, _ = self.attn(E, E, E)
        E = self.norm(E + msg)
        E = self.ff_norm(E + self.ff(E))
        return E


# ──────────────────────────────────────────────────────────────────────────────
# Main Module
# ──────────────────────────────────────────────────────────────────────────────

class KnowledgeGraphModule(nn.Module):
    """
    Differentiable Knowledge Graph limb for OctoTetrahedral AGI.

    Forward:
        1. Pool query:      q = mean_pool(x)              [B, D]
        2. Expand entities: E = entity_emb + rel_bias(q)  [B, N, D]
        3. Message passing: R × _RGATHop(E)               [B, N, D]
        4. Soft retrieval:  kg = attn(q, E_mp)            [B, D]
        5. Fuse:            out = LayerNorm(x + proj(kg))  [B, L, D]
        6. EMA write (opt): top-K entity slots ← EMA(q)

    The KG stream is designed to be added as an extra limb in the
    CompoundBraid so that the braid can attend to structured entity
    knowledge alongside perceptual, spatial and linguistic streams.
    """

    def __init__(self, config: KGConfig, hidden_dim: int):
        super().__init__()
        self.config = config
        self.hidden_dim = hidden_dim

        # ── Entity & relation banks ──
        self.entity_emb = nn.Embedding(config.num_entities, hidden_dim)
        self.relation_emb = nn.Embedding(config.num_relations, hidden_dim)

        # Relation-conditioned entity bias: given query, compute a small
        # per-entity bias via learned relation attention so the entity
        # representations are contextually modulated.
        self.rel_query_proj = nn.Linear(hidden_dim, config.num_relations, bias=False)
        self.rel_entity_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # ── Message passing ──
        self.mp_hops = nn.ModuleList([
            _RGATHop(hidden_dim, config.num_relations, config.dropout)
            for _ in range(config.num_hops)
        ])

        # ── Retrieval ──
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.scale = hidden_dim ** -0.5

        # ── Output fusion ──
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(config.dropout)

        # ── Confidence head (for limb protocol compatibility) ──
        self.conf_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    # ──────────────────────────────────────────────────────────────────────
    # Weight init
    # ──────────────────────────────────────────────────────────────────────

    def _init_weights(self) -> None:
        nn.init.normal_(self.entity_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.relation_emb.weight, mean=0.0, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────

    def _build_entity_repr(self, q: torch.Tensor) -> torch.Tensor:
        """
        Build contextually-modulated entity representations.

        Adds a relation-conditioned bias to the base entity embeddings
        so that the entity bank is gated by the incoming query.

        Args:
            q: Query vector  [B, D]
        Returns:
            E: Entity representations  [B, num_entities, D]
        """
        B = q.size(0)
        E_base = self.entity_emb.weight  # [N, D]

        # Relation weights conditioned on query  [B, num_relations]
        rel_w = self.rel_query_proj(q).softmax(dim=-1)

        # Weighted sum of relation embeddings as a gating bias  [B, D]
        rel_bias = rel_w @ self.relation_emb.weight  # [B, D]

        # Apply bias to entity base embeddings via projection
        E = E_base.unsqueeze(0).expand(B, -1, -1)           # [B, N, D]
        E = E + self.rel_entity_proj(rel_bias).unsqueeze(1)  # [B, N, D]
        return E

    @torch.no_grad()
    def _ema_write(
        self,
        q: torch.Tensor,
        attn_weights: torch.Tensor,
    ) -> None:
        """
        EMA-write top-K query content into the most-attended entity slots.

        This is done *without gradients* (detached), so it acts as a
        persistent context update that does not interfere with the
        computational graph.

        Args:
            q:            Query vectors        [B, D]
            attn_weights: Retrieval attention  [B, num_entities]
        """
        alpha = self.config.ema_write_alpha
        if alpha <= 0.0:
            return

        top_k = min(self.config.top_k_retrieval, self.config.num_entities)
        top_idx = attn_weights.topk(top_k, dim=-1).indices  # [B, top_k]

        # Average query across the batch to get a single write vector
        q_mean = q.detach().mean(dim=0)  # [D]

        # Average the top-k entity indices across the batch
        idx_flat, _ = top_idx.view(-1).sort()
        unique_idx = idx_flat.unique()

        self.entity_emb.weight.data[unique_idx] = (
            (1.0 - alpha) * self.entity_emb.weight.data[unique_idx]
            + alpha * q_mean.unsqueeze(0)
        )

    # ──────────────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        update_entities: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Args:
            x:               Hidden states  [B, L, D]
            update_entities: Whether to EMA-write top-K entity slots
                             (only meaningful after initialisation).

        Returns:
            out:        KG-enriched hidden states  [B, L, D]
            confidence: Per-sample confidence      [B, 1]
            info:       Diagnostic dict
        """
        B, L, D = x.shape

        # 1. Pool query
        q = self.query_proj(x.mean(dim=1))   # [B, D]

        # 2. Contextual entity representations
        E = self._build_entity_repr(q)       # [B, N, D]

        # 3. Multi-hop message passing
        for hop in self.mp_hops:
            E = hop(E)

        # 4. Soft retrieval via scaled dot-product attention
        keys = self.key_proj(E)              # [B, N, D]
        logits = torch.bmm(
            q.unsqueeze(1),                  # [B, 1, D]
            keys.transpose(1, 2)             # [B, D, N]
        ) * self.scale                       # [B, 1, N]
        attn_w = logits.softmax(dim=-1)      # [B, 1, N]

        kg_retrieved = torch.bmm(attn_w, E).squeeze(1)  # [B, D]

        # 5. Project and expand to sequence length
        kg_seq = self.dropout(
            self.out_proj(kg_retrieved)
        ).unsqueeze(1).expand(-1, L, -1)     # [B, L, D]

        # 6. Residual fusion
        out = self.out_norm(x + kg_seq)      # [B, L, D]

        # 7. Confidence
        confidence = self.conf_head(kg_retrieved)  # [B, 1]

        # 8. Optional EMA write
        if update_entities and self.training and self.config.ema_write_alpha > 0:
            self._ema_write(q, attn_w.squeeze(1))

        info: Dict = {
            "kg_attn_weights": attn_w.squeeze(1).detach(),              # [B, N]
            "kg_top_entity": attn_w.squeeze(1).argmax(dim=-1),          # [B]
            "kg_retrieved_norm": kg_retrieved.detach().norm(dim=-1),     # [B]
            "kg_confidence": confidence.detach().mean().item(),
        }
        return out, confidence, info

    # ──────────────────────────────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────────────────────────────

    def extra_repr(self) -> str:
        c = self.config
        return (
            f"num_entities={c.num_entities}, num_relations={c.num_relations}, "
            f"num_hops={c.num_hops}, ema_alpha={c.ema_write_alpha}"
        )
