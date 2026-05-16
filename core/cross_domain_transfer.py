"""
Cross-Domain Transfer Learning Module
======================================
Enables the OctoTetrahedral model to transfer knowledge between modalities:

  Domains
  -------
  arc_grid     — 2D spatial/visual patterns (ARC-AGI grids, vis_out)
  language     — token sequences (language_out)
  sensorimotor — physical/perceptual signals (perception_echo, embodiment)
  planning     — goal-state representations (planning_out)
  abstract     — symbolic/logical patterns (reasoning_out)
  memory       — episodic/semantic memory traces (memory_out)
  social       — empathy, emotion, ethics signals

Architecture
------------
  domain A representation
  domain B representation    ──→  DomainAdapter (per-domain) ──→ shared_dim
  domain C representation

                                     ↓ project
                           CrossDomainMemoryBank  (FIFO key-value store)
                                     ↓ retrieve top-k from other domains
                           TransferRouter (cross-attention)
                                     ↓ enriched representation
                           OutputProjector (shared_dim → hidden_dim)
                                     ↓
                    enriched domain representations  +  aux_losses

Training losses
---------------
  transfer_alignment_loss — NT-Xent: pull matched cross-domain reps together
  domain_adversarial_loss — gradient-reversed domain classifier (invariance)
  reconstruction_loss     — optional cross-domain reconstruction target

Integration point
-----------------
  Sits between KimiCognitiveBraid and CompoundBraid in model.py forward pass.
  Called with a dict of {domain_name: tensor[B, S, H]}, returns enriched
  tensors in the same dict shape plus an `aux_losses` entry.

Usage::

    layer = CrossDomainTransferLayer(
        hidden_dim=256,
        domains=["arc_grid", "language", "abstract", "memory"],
    )
    enriched, aux = layer({
        "arc_grid": spatial_out,
        "language": language_out,
        "abstract": reasoning_out,
        "memory":   memory_out,
    })
    total_loss += aux["transfer_alignment_loss"] + aux["domain_adversarial_loss"]
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class CrossDomainConfig:
    """All tuneable hyper-parameters for the transfer layer."""
    # Shared latent dimension (can differ from hidden_dim)
    shared_dim: int = 128

    # Memory bank FIFO capacity per domain
    bank_size_per_domain: int = 256

    # Number of cross-domain keys to retrieve per query
    top_k: int = 4

    # Cross-attention heads in TransferRouter
    num_heads: int = 4

    # Dropout across all sub-modules
    dropout: float = 0.1

    # NT-Xent temperature for alignment loss
    temperature: float = 0.07

    # Gradient reversal scale for domain adversarial loss
    grl_lambda: float = 0.1

    # Loss weights
    alignment_loss_weight: float = 0.02
    adversarial_loss_weight: float = 0.01

    # Whether to enable the domain adversarial classifier
    adversarial_enabled: bool = True

    # Domains known at construction time (order is preserved)
    domains: List[str] = field(default_factory=lambda: [
        "arc_grid", "language", "sensorimotor",
        "planning", "abstract", "memory", "social",
    ])


# ---------------------------------------------------------------------------
# Gradient Reversal Layer
# ---------------------------------------------------------------------------

class _GradReverse(torch.autograd.Function):
    """Flip gradient sign during backward pass (Ganin et al., 2016)."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lam: float) -> torch.Tensor:  # type: ignore[override]
        ctx.lam = lam
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        return -ctx.lam * grad_output, None


def grad_reverse(x: torch.Tensor, lam: float = 1.0) -> torch.Tensor:
    return _GradReverse.apply(x, lam)


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------

class DomainAdapter(nn.Module):
    """Projects one domain's hidden representations into shared_dim space."""

    def __init__(self, hidden_dim: int, shared_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, shared_dim, bias=False),
            nn.LayerNorm(shared_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(shared_dim, shared_dim, bias=False),
            nn.LayerNorm(shared_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, S, hidden_dim] → [B, S, shared_dim]"""
        return self.net(x)


class CrossDomainMemoryBank(nn.Module):
    """
    FIFO key-value store. Keys and values are in shared_dim space.
    Each entry is tagged with a domain index so we can exclude
    same-domain entries during retrieval (force cross-domain transfer).

    The bank is NOT a learnable parameter — it stores detached activations
    from the current batch, functioning as an episodic transfer cache.
    """

    def __init__(self, num_domains: int, bank_size_per_domain: int, shared_dim: int):
        super().__init__()
        total = num_domains * bank_size_per_domain
        self.register_buffer("keys",   torch.zeros(total, shared_dim))
        self.register_buffer("values", torch.zeros(total, shared_dim))
        self.register_buffer("domain_ids", torch.full((total,), -1, dtype=torch.long))
        self.register_buffer("write_ptr", torch.zeros(num_domains, dtype=torch.long))

        self.num_domains = num_domains
        self.bank_size = bank_size_per_domain
        self.shared_dim = shared_dim
        self._total = total

    # offset for domain d
    def _offset(self, d: int) -> int:
        return d * self.bank_size

    @torch.no_grad()
    def update(self, domain_id: int, keys: torch.Tensor, values: torch.Tensor) -> None:
        """
        Write new (key, value) pairs for a given domain into the FIFO bank.
        keys/values: [N, shared_dim] — typically the mean-pooled sequence rep.
        """
        N = keys.size(0)
        ptr = int(self.write_ptr[domain_id].item())
        offset = self._offset(domain_id)

        for i in range(N):
            slot = offset + (ptr % self.bank_size)
            self.keys[slot] = keys[i].detach()
            self.values[slot] = values[i].detach()
            self.domain_ids[slot] = domain_id
            ptr += 1
        self.write_ptr[domain_id] = ptr % self.bank_size

    def retrieve(
        self,
        query: torch.Tensor,
        exclude_domain_id: int,
        top_k: int = 4,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find top_k most similar keys from domains OTHER than exclude_domain_id.

        query:  [B, shared_dim]
        returns:
          retrieved_values: [B, top_k, shared_dim]
          scores:           [B, top_k]
        """
        # Only consider slots from other domains that have been written
        mask = (self.domain_ids >= 0) & (self.domain_ids != exclude_domain_id)
        valid_keys   = self.keys[mask]    # [M, D]
        valid_values = self.values[mask]  # [M, D]

        if valid_keys.size(0) == 0:
            B = query.size(0)
            return (
                torch.zeros(B, top_k, self.shared_dim, device=query.device),
                torch.zeros(B, top_k, device=query.device),
            )

        # Cosine similarity [B, M]
        q_norm = F.normalize(query, dim=-1)       # [B, D]
        k_norm = F.normalize(valid_keys, dim=-1)  # [M, D]
        sims = q_norm @ k_norm.T                   # [B, M]

        k = min(top_k, valid_keys.size(0))
        scores, indices = sims.topk(k, dim=-1)    # [B, k]

        retrieved = valid_values[indices]          # [B, k, D]
        if k < top_k:
            pad = torch.zeros(query.size(0), top_k - k, self.shared_dim, device=query.device)
            retrieved = torch.cat([retrieved, pad], dim=1)
            score_pad = torch.zeros(query.size(0), top_k - k, device=query.device)
            scores = torch.cat([scores, score_pad], dim=1)

        return retrieved, scores


class TransferRouter(nn.Module):
    """
    Attend from the current domain's shared representation to the
    retrieved cross-domain memory entries.

    Learns which aspects of foreign domains are relevant via multi-head
    cross-attention, then gates the transfer with a learnable scalar.
    """

    def __init__(self, shared_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert shared_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = shared_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(shared_dim, shared_dim, bias=False)
        self.k_proj = nn.Linear(shared_dim, shared_dim, bias=False)
        self.v_proj = nn.Linear(shared_dim, shared_dim, bias=False)
        self.out_proj = nn.Linear(shared_dim, shared_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Parameter(torch.zeros(1))  # starts closed, learned to open

    def forward(
        self,
        current: torch.Tensor,    # [B, S, shared_dim]
        retrieved: torch.Tensor,  # [B, top_k, shared_dim]
    ) -> torch.Tensor:
        """Returns [B, S, shared_dim] enriched with cross-domain knowledge."""
        B, S, D = current.shape
        _, K, _ = retrieved.shape

        # current attends to retrieved
        q = self.q_proj(current).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(retrieved).view(B, K, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(retrieved).view(B, K, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, S, K]
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)                                # [B, H, S, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        out = self.out_proj(out)

        # Gated residual: learned gate starts at 0 (no transfer) and opens gradually
        gate = torch.sigmoid(self.gate)
        return current + gate * out


class DomainClassifier(nn.Module):
    """
    MLP that predicts which domain a representation came from.
    Used with gradient reversal to enforce domain-invariant representations.
    """

    def __init__(self, shared_dim: int, num_domains: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(shared_dim, shared_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(shared_dim // 2, num_domains),
        )

    def forward(self, x: torch.Tensor, lam: float = 1.0) -> torch.Tensor:
        """x: [B, shared_dim] (mean-pooled) → logits [B, num_domains]"""
        return self.net(grad_reverse(x, lam))


class OutputProjector(nn.Module):
    """Projects shared_dim back to hidden_dim with a residual path."""

    def __init__(self, hidden_dim: int, shared_dim: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(shared_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        # Residual alignment if dims differ
        self.residual_proj = (
            nn.Linear(hidden_dim, hidden_dim, bias=False)
            if hidden_dim != shared_dim else nn.Identity()
        )

    def forward(self, shared: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
        """
        shared:   [B, S, shared_dim]
        original: [B, S, hidden_dim] (before adapter projection)
        returns:  [B, S, hidden_dim]
        """
        return self.residual_proj(original) + self.proj(shared)


# ---------------------------------------------------------------------------
# NT-Xent alignment loss
# ---------------------------------------------------------------------------

def nt_xent_loss(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Contrastive alignment loss (NT-Xent / SimCLR-style).

    z_a, z_b: [N, D] — paired representations from two different domains.
    Pulls matching pairs (i, i) together, pushes all other pairs apart.
    """
    N = z_a.size(0)
    if N < 2:
        return z_a.new_zeros(1).squeeze()

    z_a = F.normalize(z_a, dim=-1)
    z_b = F.normalize(z_b, dim=-1)
    z   = torch.cat([z_a, z_b], dim=0)   # [2N, D]

    sim = torch.matmul(z, z.T) / temperature  # [2N, 2N]
    # Mask out self-similarity on the diagonal
    mask = torch.eye(2 * N, device=z.device, dtype=torch.bool)
    sim.masked_fill_(mask, float("-inf"))

    # Positive pairs: (i, N+i) and (N+i, i)
    labels = torch.cat([
        torch.arange(N, 2 * N, device=z.device),
        torch.arange(0, N, device=z.device),
    ])
    return F.cross_entropy(sim, labels)


# ---------------------------------------------------------------------------
# Main module
# ---------------------------------------------------------------------------

class CrossDomainTransferLayer(nn.Module):
    """
    Full cross-domain transfer learning layer.

    Parameters
    ----------
    hidden_dim : int
        Model hidden dimension (must match limb output size).
    cfg : CrossDomainConfig
        All hyper-parameters.
    """

    def __init__(
        self,
        hidden_dim: int,
        cfg: Optional[CrossDomainConfig] = None,
    ):
        super().__init__()
        self.cfg = cfg or CrossDomainConfig()
        self.hidden_dim = hidden_dim
        self.shared_dim = self.cfg.shared_dim
        self.domain_list = list(self.cfg.domains)
        self.domain_index: Dict[str, int] = {d: i for i, d in enumerate(self.domain_list)}
        N = len(self.domain_list)

        # Per-domain projection into shared space
        self.adapters = nn.ModuleDict({
            d: DomainAdapter(hidden_dim, self.shared_dim, self.cfg.dropout)
            for d in self.domain_list
        })

        # Episodic cross-domain memory bank
        self.bank = CrossDomainMemoryBank(N, self.cfg.bank_size_per_domain, self.shared_dim)

        # Cross-domain attention router (one shared router for all domain pairs)
        self.router = TransferRouter(self.shared_dim, self.cfg.num_heads, self.cfg.dropout)

        # Back-project to hidden_dim (with residual from original)
        self.output_projectors = nn.ModuleDict({
            d: OutputProjector(hidden_dim, self.shared_dim, self.cfg.dropout)
            for d in self.domain_list
        })

        # Domain adversarial classifier (gradient reversal)
        if self.cfg.adversarial_enabled:
            self.domain_classifier = DomainClassifier(self.shared_dim, N, self.cfg.dropout)
        else:
            self.domain_classifier = None

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        domain_inputs: Dict[str, torch.Tensor],
        grl_lambda: Optional[float] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Parameters
        ----------
        domain_inputs : Dict[str, Tensor]
            Mapping domain_name → [B, S, hidden_dim]. Only domains present in
            this dict are processed; unknown domain names are passed through.
        grl_lambda : float, optional
            Override gradient reversal scale for curriculum scheduling.

        Returns
        -------
        enriched : Dict[str, Tensor]
            Same keys as domain_inputs, same shape [B, S, hidden_dim],
            but enriched with cross-domain knowledge.
        aux_losses : Dict[str, Tensor]
            Scalar losses: "transfer_alignment_loss", "domain_adversarial_loss".
        """
        lam = grl_lambda if grl_lambda is not None else self.cfg.grl_lambda
        known = {k: v for k, v in domain_inputs.items() if k in self.domain_index}
        unknown = {k: v for k, v in domain_inputs.items() if k not in self.domain_index}

        # 1. Project all known domains into shared space
        shared: Dict[str, torch.Tensor] = {}
        for name, x in known.items():
            shared[name] = self.adapters[name](x)  # [B, S, shared_dim]

        # 2. Update memory bank (mean-pool over sequence → one vec per sample)
        for name, s in shared.items():
            did = self.domain_index[name]
            keys_vec = s.mean(dim=1)        # [B, shared_dim]
            self.bank.update(did, keys_vec, keys_vec)

        # 3. Retrieve cross-domain neighbours and route
        enriched_shared: Dict[str, torch.Tensor] = {}
        for name, s in shared.items():
            did = self.domain_index[name]
            query = s.mean(dim=1)           # [B, shared_dim]
            retrieved, _ = self.bank.retrieve(query, exclude_domain_id=did, top_k=self.cfg.top_k)
            # retrieved: [B, top_k, shared_dim]; broadcast over sequence
            enriched_shared[name] = self.router(s, retrieved)

        # 4. Project back to hidden_dim
        enriched: Dict[str, torch.Tensor] = {}
        for name, s_enriched in enriched_shared.items():
            enriched[name] = self.output_projectors[name](s_enriched, known[name])

        # Pass through any unknown-domain tensors unchanged
        enriched.update(unknown)

        # 5. Compute auxiliary losses
        aux_losses: Dict[str, torch.Tensor] = {}

        # 5a. NT-Xent alignment loss across all domain pairs
        align_loss = self._alignment_loss(shared)
        aux_losses["transfer_alignment_loss"] = align_loss * self.cfg.alignment_loss_weight

        # 5b. Domain adversarial loss
        if self.domain_classifier is not None and len(shared) > 0:
            adv_loss = self._adversarial_loss(shared, lam)
            aux_losses["domain_adversarial_loss"] = adv_loss * self.cfg.adversarial_loss_weight
        else:
            device = next(iter(shared.values())).device if shared else torch.device("cpu")
            aux_losses["domain_adversarial_loss"] = torch.zeros(1, device=device).squeeze()

        return enriched, aux_losses

    # ------------------------------------------------------------------
    # Loss helpers
    # ------------------------------------------------------------------

    def _alignment_loss(self, shared: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        NT-Xent loss over all ordered domain pairs.
        Uses mean-pooled sequence representations as the NT-Xent anchors.
        Positive pairs are matched by batch position (same sample, different domain).
        """
        names = list(shared.keys())
        if len(names) < 2:
            v = next(iter(shared.values()))
            return v.new_zeros(1).squeeze()

        total = v = next(iter(shared.values())).new_zeros(1).squeeze()
        count = 0
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                z_a = shared[names[i]].mean(dim=1)  # [B, D]
                z_b = shared[names[j]].mean(dim=1)  # [B, D]
                total = total + nt_xent_loss(z_a, z_b, self.cfg.temperature)
                count += 1
        return total / max(count, 1)

    def _adversarial_loss(
        self,
        shared: Dict[str, torch.Tensor],
        lam: float,
    ) -> torch.Tensor:
        """
        Cross-entropy on domain labels with gradient reversal.
        Encourages shared representations to be domain-indistinguishable.
        """
        all_vecs, all_labels = [], []
        for name, s in shared.items():
            did = self.domain_index[name]
            pooled = s.mean(dim=1)  # [B, D]
            all_vecs.append(pooled)
            all_labels.append(torch.full((pooled.size(0),), did, device=pooled.device))

        vecs   = torch.cat(all_vecs, dim=0)    # [B*D, shared_dim]
        labels = torch.cat(all_labels, dim=0)  # [B*D]
        logits = self.domain_classifier(vecs, lam)
        return F.cross_entropy(logits, labels)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def add_domain(self, name: str) -> None:
        """Dynamically register a new domain at runtime (e.g., from a plugin limb)."""
        if name in self.domain_index:
            return
        did = len(self.domain_list)
        self.domain_list.append(name)
        self.domain_index[name] = did
        self.adapters[name] = DomainAdapter(self.hidden_dim, self.shared_dim, self.cfg.dropout)
        self.output_projectors[name] = OutputProjector(
            self.hidden_dim, self.shared_dim, self.cfg.dropout
        )
        # Note: bank and domain classifier need re-init if domains change significantly

    def transfer_summary(self) -> Dict[str, object]:
        """Return a diagnostic snapshot of the memory bank occupancy."""
        occupancy = {}
        for name, did in self.domain_index.items():
            offset = self.bank._offset(did)
            filled = int((self.bank.domain_ids[offset:offset + self.bank.bank_size] >= 0).sum().item())
            occupancy[name] = filled
        return {
            "domains": list(self.domain_index.keys()),
            "bank_occupancy": occupancy,
            "router_gate": float(torch.sigmoid(self.router.gate).item()),
            "shared_dim": self.shared_dim,
        }
