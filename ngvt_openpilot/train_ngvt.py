"""
NGVT Braid PyTorch Training Loop
==================================
Trains the NgvtBraidEngineTorch as a differentiable layer that learns to
predict adjusted lead-tracking confidence scores from openpilot log data.

Architecture:
  LogReader  →  LeadDataV3 (x, y, prob)
       ↓
  NgvtBraidEngineTorch   (torus projection + RBF Braid boost)
       ↓
  LeadScoringHead        (2-layer MLP, learned on top of torus coords)
       ↓
  Loss: BCE(adjusted_score, future_engagement_label)

The model learns:
  1. How to set σ (failure-zone influence radius) adaptively
  2. A small MLP that refines the Braid score using the full torus geometry

Usage:
  # Train from openpilot route logs
  python train_ngvt.py --route "a2a0ccea32023010|2023-07-27--13-01-19" --epochs 20

  # Train from a pre-built JSON dataset (output of ngvt_analysis.py)
  python train_ngvt.py --json-dataset results.json --epochs 20

  # Evaluate and export
  python train_ngvt.py --checkpoint ngvt_weights.pt --eval-only
  python train_ngvt.py --checkpoint ngvt_weights.pt --export ngvt_scripted.pt
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Import the differentiable engine
sys.path.insert(0, str(Path(__file__).parent))
from ngvt_braid_torch import NgvtBraidEngineTorch


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class NgvtLeadDataset(Dataset):
    """
    Dataset of (xy, raw_prob, label) tuples.

    Label = 1 if the lead was 'real' (high long-term engagement), 0 otherwise.
    Constructed from ngvt_analysis.py JSON output or directly from LogReader.

    When reading directly from logs, uses a simple heuristic label:
      label = 1  if  raw_prob > 0.7  (model was confident about this lead)
      label = 0  otherwise
    This is intentionally simple — replace with ground-truth labels (e.g.
    radar-confirmed leads) for production training.
    """

    CONFIDENCE_LABEL_THRESHOLD = 0.7

    def __init__(self, samples: List[dict]) -> None:
        """
        Args:
            samples: list of dicts with keys:
                'x'         float  — leadsV3[i].x[0]
                'y'         float  — leadsV3[i].y[0]
                'raw_prob'  float  — leadsV3[i].prob
                'label'     float  — 1.0 or 0.0 (optional; generated if absent)
        """
        self.xy    = torch.tensor([[s['x'], s['y']] for s in samples], dtype=torch.float32)
        self.probs = torch.tensor([s['raw_prob'] for s in samples], dtype=torch.float32)
        self.labels = torch.tensor(
            [s.get('label', float(s['raw_prob'] > self.CONFIDENCE_LABEL_THRESHOLD))
             for s in samples],
            dtype=torch.float32,
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.xy[idx], self.probs[idx], self.labels[idx]

    @classmethod
    def from_json(cls, path: str) -> "NgvtLeadDataset":
        """Load from ngvt_analysis.py JSON output."""
        frames = json.loads(Path(path).read_text())
        samples = []
        for frame in frames:
            for node in frame.get("nodes", []):
                samples.append({
                    "x":        node["raw_x"],
                    "y":        node["raw_y"],
                    "raw_prob": node["raw_prob"],
                    # Use flagged_unstable inverse as a simple label proxy:
                    # stable (non-flagged) nodes are treated as 'real' leads
                    "label":    0.0 if node["flagged_unstable"] else float(node["raw_prob"] > 0.5),
                })
        print(f"Loaded {len(samples)} lead samples from {path}")
        return cls(samples)

    @classmethod
    def from_logreader(cls, log_paths: List[str]) -> "NgvtLeadDataset":
        """Build dataset directly from openpilot .rlog files."""
        try:
            from openpilot.tools.lib.logreader import LogReader
        except ImportError:
            raise ImportError("Run from an openpilot checkout with its venv active.")

        samples = []
        for lp in log_paths:
            lr = LogReader(lp)
            for model in lr.filter("modelV2"):
                for lead in model.leadsV3:
                    samples.append({
                        "x":        float(lead.x[0]) if len(lead.x) > 0 else 0.0,
                        "y":        float(lead.y[0]) if len(lead.y) > 0 else 0.0,
                        "raw_prob": float(lead.prob),
                    })
        print(f"Loaded {len(samples)} lead samples from {len(log_paths)} log(s)")
        return cls(samples)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class NgvtBraidModel(nn.Module):
    """
    End-to-end trainable NGVT Braid model.

    Architecture:
      1. NgvtBraidEngineTorch  — differentiable torus projection + RBF boost
         (σ and boost_factor are learned parameters)
      2. LeadScoringHead       — 2-layer MLP that refines the score using
         the full 3D torus geometry alongside the Braid-adjusted score
    """

    def __init__(
        self,
        major_radius: float = 10.0,
        minor_radius: float = 3.0,
        hidden_dim: int = 32,
    ) -> None:
        super().__init__()

        # Core engine with learnable scalar parameters
        self.engine = NgvtBraidEngineTorch(
            major_radius=major_radius,
            minor_radius=minor_radius,
            boost_factor=3.0,
            sigma=1.28,
        )
        # Make boost_factor and sigma learnable
        self.log_boost = nn.Parameter(torch.log(torch.tensor(3.0)))   # exp → boost_factor > 0
        self.log_sigma = nn.Parameter(torch.log(torch.tensor(1.28)))  # exp → sigma > 0

        # MLP: (torus_x, torus_y, torus_z, braid_score) → refined_score
        self.head = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, xy: torch.Tensor, raw_probs: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            xy:         (B, 2) pixel coordinates
            raw_probs:  (B,)   existence probabilities
        Returns:
            (B,) refined confidence scores ∈ [0, 1]
        """
        # Update engine with current learned parameters
        self.engine.boost_factor = float(self.log_boost.detach().exp())
        self.engine.sigma        = float(self.log_sigma.detach().exp())

        coords, braid_scores = self.engine(xy, raw_probs)   # (B,3), (B,)

        # Concat torus coords + braid score → MLP input (B, 4)
        features = torch.cat([coords, braid_scores.unsqueeze(1)], dim=1)
        refined   = self.head(features).squeeze(1)           # (B,)
        return refined


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    model: NgvtBraidModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for xy, probs, labels in loader:
        xy, probs, labels = xy.to(device), probs.to(device), labels.to(device)
        optimizer.zero_grad()
        scores = model(xy, probs)
        loss = F.binary_cross_entropy(scores, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: NgvtBraidModel,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """Returns (loss, accuracy)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for xy, probs, labels in loader:
        xy, probs, labels = xy.to(device), probs.to(device), labels.to(device)
        scores = model(xy, probs)
        total_loss += F.binary_cross_entropy(scores, labels).item() * len(labels)
        correct += ((scores > 0.5).float() == labels).sum().item()
        total   += len(labels)
    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train the NGVT Braid scoring model.")
    src = p.add_mutually_exclusive_group(required=False)
    src.add_argument("--route",        help="openpilot route ID (requires openpilot checkout)")
    src.add_argument("--json-dataset", help="JSON file from tools/ngvt_analysis.py")
    p.add_argument("--epochs",      type=int,   default=20)
    p.add_argument("--batch-size",  type=int,   default=64)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--val-split",   type=float, default=0.2)
    p.add_argument("--checkpoint",  help="Path to save/load model weights (.pt)")
    p.add_argument("--eval-only",   action="store_true")
    p.add_argument("--export",      help="Export TorchScript model to this path")
    p.add_argument("--hidden-dim",  type=int, default=32)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")
    print(f"Device: {device}")

    # ---- Build model ----
    model = NgvtBraidModel(hidden_dim=args.hidden_dim).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    if args.checkpoint and Path(args.checkpoint).exists():
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"Loaded weights from {args.checkpoint}")

    if args.export:
        model.eval()
        scripted = torch.jit.script(model)
        scripted.save(args.export)
        print(f"Exported TorchScript model → {args.export}")
        return

    # ---- Build dataset ----
    if args.json_dataset:
        dataset = NgvtLeadDataset.from_json(args.json_dataset)
    elif args.route:
        try:
            from openpilot.tools.lib.route import Route
            r = Route(args.route)
            dataset = NgvtLeadDataset.from_logreader([str(p) for p in r.log_paths() if p])
        except ImportError:
            print("ERROR: openpilot not found. Use --json-dataset instead.")
            sys.exit(1)
    else:
        # Generate synthetic data for smoke-testing
        print("No data source provided — generating 2000 synthetic samples for smoke test.")
        rng = np.random.default_rng(42)
        samples = [{"x": float(rng.uniform(0, 1164)), "y": float(rng.uniform(0, 874)),
                    "raw_prob": float(rng.uniform(0, 1))} for _ in range(2000)]
        dataset = NgvtLeadDataset(samples)

    # ---- Train/val split ----
    n_val   = max(1, int(len(dataset) * args.val_split))
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"Train: {n_train}  Val: {n_val}")

    if args.eval_only:
        val_loss, val_acc = evaluate(model, val_loader, device)
        print(f"Val loss: {val_loss:.4f}  accuracy: {val_acc:.3f}")
        return

    # ---- Training loop ----
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_val_loss = float("inf")
    ckpt_path = args.checkpoint or "ngvt_weights.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, device)
        scheduler.step()

        sigma_val = float(model.log_sigma.exp())
        boost_val = float(model.log_boost.exp())
        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train={train_loss:.4f}  val={val_loss:.4f}  acc={val_acc:.3f}  "
              f"σ={sigma_val:.3f}  boost={boost_val:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_path)

    print(f"\nBest val loss: {best_val_loss:.4f}  →  saved to {ckpt_path}")
    print(f"Learned σ={float(model.log_sigma.exp()):.3f}  "
          f"boost={float(model.log_boost.exp()):.3f}")


if __name__ == "__main__":
    main()
