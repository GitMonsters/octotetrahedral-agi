"""
Migration Guide: From Fragment to Unified Stack
=================================================

This guide explains how to migrate from the old fragmented architecture
(50+ individual solvers, scattered limbs) to the new unified cognitive stack.

Before: Fragmentation
  arc_solver_v1.py
  arc_solver_v2.py
  ...
  arc_solver_v50.py
  rearc_solver.py
  cognitive_layer.py (unused)
  memory_layer.py (unused)
  ...

After: Unified Stack
  unified/
    ├── limbs_orchestrator.py      (8 limbs + quantum hub)
    ├── rna_editing_layer.py       (adaptive gating)
    ├── quantum_coupling.py        (entanglement)
    ├── forward_model.py           (complete pipeline)
    ├── unified_solver.py          (1 parametric solver)
    └── __init__.py

═══════════════════════════════════════════════════════════════
Step 1: Installation
═══════════════════════════════════════════════════════════════

No new dependencies! Uses PyTorch + existing infrastructure.

  git checkout unified-cognitive-stack
  python -c "from unified import UnifiedForwardModel; print('✓ Import OK')"

═══════════════════════════════════════════════════════════════
Step 2: Load Pre-trained Weights (Optional)
═══════════════════════════════════════════════════════════════

If you have old arc_solver weights, you can partially transfer them:

  from unified import UnifiedForwardModel
  model = UnifiedForwardModel()
  
  # Transfer old perception layer
  old_embedding = old_model.embedding.weight
  model.embedding.weight.data[:old_embedding.shape[0]] = old_embedding
  
  print("Weights transferred!")

═══════════════════════════════════════════════════════════════
Step 3: Quick Start - Run Inference
═══════════════════════════════════════════════════════════════

  import torch
  from unified import UnifiedForwardModel
  from unified.unified_solver import UnifiedARCSolver
  
  # Create solver
  solver = UnifiedARCSolver(
      hidden_dim=512,
      num_limbs=8,
      enable_quantum=True
  )
  
  # Solve an ARC task
  input_grid = torch.randint(0, 11, (30, 30))
  result = solver.solve(input_grid)
  
  print(f"Task type: {solver.get_task_name(result['task_type'])}")
  print(f"Solution: {result['solution'].shape}")
  print(f"Confidence: {result['confidence'].item():.2%}")

═══════════════════════════════════════════════════════════════
Step 4: Training
═══════════════════════════════════════════════════════════════

  from unified import UnifiedForwardModel
  import torch.optim as optim
  
  model = UnifiedForwardModel(enable_quantum=True, enable_rna_editing=True)
  optimizer = optim.AdamW(model.parameters(), lr=1e-4)
  
  # Training loop
  for epoch in range(10):
      for batch_input, batch_labels in data_loader:
          # Forward pass
          output = model(batch_input, labels=batch_labels)
          loss = output['loss']
          
          # Backward
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          
          # Log metrics
          print(f"Loss: {loss.item():.4f}")
          print(f"RNA confidence: {output['metrics']['rna_confidence']:.3f}")
          print(f"Entanglement: {output['metrics']['entanglement_strength']:.3f}")

═══════════════════════════════════════════════════════════════
Step 5: Advanced - Customize Limb Emphasis
═══════════════════════════════════════════════════════════════

For specific task types, you can emphasize certain limbs:

  from unified import UnifiedForwardModel
  from unified.rna_editing_layer import RNAEditingLayer
  
  model = UnifiedForwardModel()
  
  # Get RNA editor
  rna = model.rna_editing
  
  # Manually set limb gates for a task
  hidden = torch.randn(1, 32, 512)
  rna_result = rna(hidden)
  
  # Emphasize spatial + reasoning limbs for geometric task
  rna_result['limb_gates'][0, 2] = 0.95  # Spatial limb
  rna_result['limb_gates'][0, 3] = 0.95  # Reasoning limb
  
  # Use edited gates in forward pass
  limbs_output = model.limbs(
      hidden,
      rna_gates=rna_result['limb_gates']
  )

═══════════════════════════════════════════════════════════════
Step 6: Debugging & Analysis
═══════════════════════════════════════════════════════════════

  # Get model statistics
  stats = model.get_stats()
  print(f"Total params: {stats['total_params']:,}")
  print(f"Quantum enabled: {stats['quantum_enabled']}")
  
  # Analyze limb activations
  limbs_stats = model.limbs.get_stats()
  print(f"Limb activations: {limbs_stats['limb_activations']}")
  
  # Get quantum coupling info
  coupling_stats = model.quantum_entanglement.quantum_coupling.get_coupling_statistics()
  print(f"Entanglement strength: {coupling_stats}")

═══════════════════════════════════════════════════════════════
Step 7: Deployment
═══════════════════════════════════════════════════════════════

  # Save model
  torch.save(model.state_dict(), 'unified_model.pt')
  
  # Load for inference
  model = UnifiedForwardModel()
  model.load_state_dict(torch.load('unified_model.pt'))
  model.eval()
  
  # Generate with beam search or sampling
  with torch.no_grad():
      generated = model.generate(
          input_ids,
          max_new_tokens=100,
          temperature=0.8,
          top_k=50
      )

═══════════════════════════════════════════════════════════════
FAQ
═══════════════════════════════════════════════════════════════

Q: Should I keep the old arc_solver_*.py files?
A: Optional. The unified stack is self-contained, but keeping them
   allows gradual migration. You can run both in parallel initially.

Q: How do I compare performance?
A: Use the unified solver on the same test set:
   - Before: Run arc_solver_v50.py on test set
   - After: Run unified_solver.py on same test set
   - Compare accuracy, speed, memory usage

Q: Can I disable quantum coupling?
A: Yes: UnifiedForwardModel(enable_quantum=False)
   This reduces complexity but may hurt reasoning performance.

Q: Can I disable RNA editing?
A: Yes: UnifiedForwardModel(enable_rna_editing=False)
   This uses equal limb emphasis (less adaptive).

Q: What about RE-ARC Bench compatibility?
A: The solver auto-detects task types. For RE-ARC, it routes
   through the reasoning limbs more aggressively.

Q: How do I add a custom limb?
A: 1. Create a new CognitiveLimb subclass
   2. Add to UnifiedLimbsOrchestrator.limbs list
   3. Update num_limbs in RNA editing (9 instead of 8)
   4. Retrain

═══════════════════════════════════════════════════════════════
Phased Rollout Strategy
═══════════════════════════════════════════════════════════════

Phase 1 (Week 1): Parallel Running
  - Keep old solvers running
  - Add unified solver alongside
  - Compare outputs on small test set
  - Monitor: accuracy, inference time, memory

Phase 2 (Week 2): Selective Replacement
  - Replace arc_solver_v1 through v10 with unified
  - Run 420 ARC-AGI puzzles
  - Verify: accuracy maintained or improved

Phase 3 (Week 3-4): Full Migration
  - Replace all 50+ solvers
  - Consolidate RE-ARC variant
  - Train on combined dataset
  - Deploy to production

Phase 4 (Ongoing): Optimization
  - Fine-tune hyperparameters
  - Add custom limbs for weak areas
  - Continuous evaluation

═══════════════════════════════════════════════════════════════
Expected Outcomes
═══════════════════════════════════════════════════════════════

Codebase:
  ✓ 50+ files consolidated → ~5 core files (90% reduction)
  ✓ Redundancy eliminated
  ✓ Unified API (all tasks use same solver)
  ✓ Easier maintenance + extensibility

Performance:
  ✓ More coherent reasoning (quantum coupling)
  ✓ Adaptive limb emphasis (RNA editing)
  ✓ Better generalization (shared representations)
  ✓ Faster inference (single model vs. multi-dispatch)

Research Value:
  ✓ Interpretable limb contributions (which limbs activate per task?)
  ✓ Learnable meta-patterns (RNA editing discovers best pathways)
  ✓ Genuine entanglement (quantum coherence metrics)
  ✓ Publication-ready architecture

═══════════════════════════════════════════════════════════════
Troubleshooting
═══════════════════════════════════════════════════════════════

Issue: CUDA out of memory
Solution: Reduce hidden_dim (256 instead of 512) or batch_size

Issue: Loss not decreasing
Solution:
  - Lower learning rate (1e-5 instead of 1e-4)
  - Increase warmup steps
  - Check RNA editing loss weight

Issue: Quantum coupling makes model slow
Solution:
  - Set enable_quantum=False for fast prototyping
  - Reduce num_qubits in QuantumEntanglementLayer
  - Profile to identify bottleneck

Issue: Limb activations always the same
Solution:
  - Increase RNA editing loss weight
  - Add task-type diversity to training set
  - Check if task detector is working (should output varied types)

═══════════════════════════════════════════════════════════════
Next Steps
═══════════════════════════════════════════════════════════════

1. Checkout branch: git checkout unified-cognitive-stack
2. Run tests: python unified/forward_model.py
3. Benchmark: Compare on 10 ARC-AGI puzzles
4. Train: Run training loop on full dataset
5. Deploy: Replace old solvers in production
6. Iterate: Gather metrics, optimize, publish

Good luck! 🚀
"""
