# OctoTetrahedral AGI

A novel AGI architecture combining tetrahedral geometry, octopus-inspired RNA editing, and distributed 8-limb processing.

## Architecture

```
Input (tokens/embeddings)
     ↓
Perception Limb (embedding + encoding)
     ↓
RNA Editing Layer (dynamic adaptation)
     ↓
Tetrahedral Core (geometry-aware transformer)
     ↓
┌─────────────────────────────────────────┐
│           8-Limb Processing             │
│  Memory ─── Planning ─── Language       │
│     │          │           │            │
│  Spatial ─── Reasoning ─── MetaCog      │
└─────────────────────────────────────────┘
     ↓
Hub Synchronization
     ↓
AGICognition (causal discovery, world model, meta-learning)
     ↓
Action Limb (output generation)
     ↓
Output (logits)
```

## Key Features

- **Tetrahedral Geometry**: 64-point structure for attention
- **Octopus-inspired RNA Editing**: Dynamic weight modulation
- **8 Specialized Limbs**: Perception, Memory, Planning, Language, Spatial, Reasoning, MetaCognition, Action
- **AGI Cognition**: Causal discovery, world model, meta-learning
- **~89M Parameters**

## Training Status

Training on ARC-AGI dataset:
- **Last checkpoint**: `arc_step_2500.pt` (step 2500, epoch 20)
- **Target**: 60 epochs
- **Resume command**: See below

## Installation

```bash
pip install -r requirements.txt
```

## Resume Training (GPU)

To continue training from the checkpoint:

```bash
python train_arc.py \
    --resume checkpoints/arc/arc_step_2500.pt \
    --max-steps 7500 \
    --batch-size 8
```

For full 60 epoch training (~7500 total steps):
```bash
python train_arc.py \
    --resume checkpoints/arc/arc_step_2500.pt \
    --max-steps 7500 \
    --batch-size 16  # Increase if GPU memory allows
```

## Data

Requires ARC-AGI dataset. Set the data path:
```bash
--data-dir /path/to/ARC-AGI/data
```

## Files

- `model.py` - Main OctoTetrahedral model
- `train_arc.py` - ARC-AGI training script
- `config.py` - Configuration
- `cognition.py` - AGI cognition module
- `core/` - Tetrahedral attention and geometry
- `limbs/` - 8 specialized processing limbs
- `adaptation/` - RNA editing and LoRA
- `sync/` - Hub synchronization
- `data/` - Dataset loaders
