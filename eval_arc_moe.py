"""
OctoTetrahedral AGI — ARC-AGI Evaluation Pipeline

Evaluates a trained MoE model on the ARC-AGI benchmark.
Measures exact match accuracy, cell accuracy, and per-limb confidence breakdown.

Usage:
    # Evaluate on ARC training set (for dev)
    python eval_arc_moe.py --config 7b --checkpoint checkpoints/best.pt --split training

    # Evaluate on ARC evaluation set (for benchmark)
    python eval_arc_moe.py --config 7b --checkpoint checkpoints/best.pt --split evaluation

    # Quick eval on 10 tasks
    python eval_arc_moe.py --checkpoint checkpoints/best.pt --max-tasks 10
"""

import argparse
import copy
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_time_train(
    model,
    tokenizer,
    task,
    config,
    device: str,
    ttt_steps: int = 10,
    ttt_lr: float = 1e-5,
) -> torch.nn.Module:
    """
    Test-time training: fine-tune model on a task's train examples before predicting.
    
    Creates leave-one-out training pairs from the task's demonstrations,
    runs a few gradient steps, then returns the adapted model.
    The caller should save/restore original weights.
    """
    from data.arc_dataset import ARCTask, grid_to_tokens

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=ttt_lr)

    # Build training data from the task's train examples
    # Each train example becomes: prompt (all OTHER train examples) + this input -> this output
    train_examples = task.train_examples
    if len(train_examples) < 2:
        # Need at least 2 examples for leave-one-out
        # Use single example as self-supervised target
        pairs = [(train_examples, 0)]
    else:
        pairs = [(train_examples, i) for i in range(len(train_examples))]

    for step in range(ttt_steps):
        total_loss = 0.0
        for examples, target_idx in pairs:
            # Build prompt: all examples as context, target_idx as the "test"
            parts = []
            for j, ex in enumerate(examples):
                if j == target_idx and len(examples) > 1:
                    continue
                inp_str = grid_to_tokens(ex['input'])
                out_str = grid_to_tokens(ex['output'])
                parts.append(f"[{inp_str}]->[{out_str}]")

            target_ex = examples[target_idx]
            inp_str = grid_to_tokens(target_ex['input'])
            out_str = grid_to_tokens(target_ex['output'])
            prompt = ' '.join(parts) + f" [{inp_str}]->["
            target = out_str + "]"

            # Tokenize
            prompt_tokens = tokenizer.encode(prompt)
            target_tokens = tokenizer.encode(target)
            full_tokens = prompt_tokens + target_tokens

            if len(full_tokens) > config.model.max_seq_len:
                full_tokens = full_tokens[-config.model.max_seq_len:]
                prompt_tokens = full_tokens[:max(1, len(full_tokens) - len(target_tokens))]
                target_tokens = full_tokens[len(prompt_tokens):]

            input_ids = torch.tensor([full_tokens[:-1]], device=device)
            labels = torch.tensor([full_tokens[1:]], device=device)
            # Mask prompt portion
            labels[0, :len(prompt_tokens) - 1] = -100

            output = model(input_ids=input_ids, labels=labels)
            loss = output['loss']
            total_loss += loss.item()

            loss.backward()

        optimizer.step()
        optimizer.zero_grad()

    model.eval()
    return model


def evaluate_model(
    model,
    tokenizer,
    tasks,
    config,
    device: str,
    max_tasks: Optional[int] = None,
    max_gen_tokens: int = 256,
    temperature: float = 0.0,
    num_attempts: int = 3,
    ttt_enabled: bool = False,
    ttt_steps: int = 10,
    ttt_lr: float = 1e-5,
    gen_mode: str = 'autoregressive',
) -> Dict:
    """
    Evaluate model on ARC tasks.

    For each task:
    1. Format task as compact prompt (train examples + test input)
    2. Generate output grid tokens
    3. Parse and compare to ground truth
    4. Record exact match, cell accuracy, and limb confidences
    """
    from data.arc_dataset import ARCTask, evaluate_arc_prediction

    model.eval()
    results = []
    correct = 0
    total = 0

    task_list = list(tasks.items()) if isinstance(tasks, dict) else tasks
    if max_tasks:
        task_list = task_list[:max_tasks]

    for i, (task_id, task_data) in enumerate(task_list):
        task = ARCTask(task_id, task_data)

        # Test-time training: adapt model to this task's train examples
        saved_state = None
        if ttt_enabled:
            saved_state = copy.deepcopy(model.state_dict())
            logger.info(f"  TTT: fine-tuning on {len(task.train_examples)} train examples for {ttt_steps} steps...")
            test_time_train(model, tokenizer, task, config, device, ttt_steps, ttt_lr)

        for test_idx in range(task.num_test):
            total += 1
            input_text, target_text = task.format_compact(test_idx, include_answer=True)

            # Try multiple attempts with different temperatures
            best_result = None
            best_exact = 0.0

            for attempt in range(num_attempts):
                t = temperature if attempt == 0 else 0.3 + 0.2 * attempt

                tokens = tokenizer.encode(input_text)
                if len(tokens) > config.model.max_seq_len - max_gen_tokens:
                    tokens = tokens[-(config.model.max_seq_len - max_gen_tokens):]
                input_ids = torch.tensor([tokens], device=device)

                with torch.no_grad():
                    if gen_mode == 'diffusion':
                        generated = model.generate_diffusion(
                            input_ids=input_ids,
                            num_tokens=max_gen_tokens,
                            refine_steps=8,
                            temperature=max(t, 0.01),
                        )
                    else:
                        generated = model.generate(
                            input_ids=input_ids,
                            max_new_tokens=max_gen_tokens,
                            temperature=max(t, 1e-8),
                            do_sample=t > 0,
                        )

                new_tokens = generated[0, len(tokens):].tolist()
                predicted = tokenizer.decode(new_tokens)

                # Truncate at ] if present
                if ']' in predicted:
                    predicted = predicted[:predicted.index(']')]

                metrics = evaluate_arc_prediction(predicted, target_text)

                if metrics['exact_match'] > best_exact:
                    best_exact = metrics['exact_match']
                    best_result = {
                        'predicted': predicted,
                        'metrics': metrics,
                        'attempt': attempt,
                        'temperature': t,
                    }

                if best_exact == 1.0:
                    break

            if best_result is None:
                best_result = {
                    'predicted': predicted,
                    'metrics': metrics,
                    'attempt': num_attempts - 1,
                    'temperature': t,
                }

            # Get limb confidences
            confidences = {}
            try:
                with torch.no_grad():
                    output = model(input_ids=input_ids, return_confidences=True)
                    for limb_name in ['perception', 'reasoning', 'planning', 'language',
                                      'spatial', 'memory', 'metacognition', 'action']:
                        key = f'{limb_name}_confidence'
                        if key in output:
                            confidences[limb_name] = output[key].item()
            except Exception:
                pass

            is_correct = best_result['metrics']['exact_match'] == 1.0
            if is_correct:
                correct += 1

            result = {
                'task_id': task_id,
                'test_idx': test_idx,
                'correct': is_correct,
                'exact_match': best_result['metrics']['exact_match'],
                'grid_accuracy': best_result['metrics']['grid_accuracy'],
                'cell_accuracy': best_result['metrics']['cell_accuracy'],
                'predicted': best_result['predicted'][:200],
                'target': target_text[:200],
                'attempts_used': best_result['attempt'] + 1,
                'confidences': confidences,
            }
            results.append(result)

            status = "✅" if is_correct else "❌"
            logger.info(
                f"[{total}] {status} {task_id} test={test_idx} "
                f"exact={best_result['metrics']['exact_match']:.0f} "
                f"cell={best_result['metrics']['cell_accuracy']:.2f} "
                f"attempts={best_result['attempt']+1}"
            )

        # Restore original weights after TTT for this task
        if saved_state is not None:
            model.load_state_dict(saved_state)
            del saved_state

    accuracy = correct / total if total > 0 else 0.0

    # Compound learning: track cross-limb patterns from eval results
    compound_stats = {}
    try:
        from ngvt_compound_learning import CompoundLearningEngine, LearningExperience
        from datetime import datetime

        eval_engine = CompoundLearningEngine(max_patterns=5000, learning_rate=0.01)
        limb_names = ['perception', 'reasoning', 'planning', 'language',
                      'spatial', 'memory', 'metacognition', 'action']
        for ln in limb_names:
            eval_engine.register_model(ln, [ln])

        # Record each eval result as a learning experience
        for r in results:
            eval_engine.record_experience(LearningExperience(
                query=f"arc_eval_{r['task_id']}_t{r['test_idx']}",
                response=f"exact={r['exact_match']:.0f} cell={r['cell_accuracy']:.2f}",
                latency_ms=0,
                success=r['correct'],
                timestamp=datetime.now().isoformat(),
                confidence=r['cell_accuracy'],
                metadata={
                    'task_id': r['task_id'],
                    'confidences': r['confidences'],
                },
            ))

            # Record cross-limb transfers: high-confidence limbs helping low ones
            confs = r['confidences']
            if len(confs) >= 2:
                sorted_limbs = sorted(confs.items(), key=lambda x: x[1], reverse=True)
                best_limb = sorted_limbs[0][0]
                for other_limb, other_conf in sorted_limbs[1:]:
                    if other_conf > 0:
                        from ngvt_compound_learning import CompoundLearningPattern
                        pattern = CompoundLearningPattern(
                            pattern_id=f"{best_limb}_to_{other_limb}",
                            query_hash=f"{r['task_id']}",
                            response_template="cross_limb_transfer",
                            accuracy=r['cell_accuracy'],
                            frequency=1,
                            avg_latency_ms=0,
                        )
                        transfer_score = min(1.0, confs[best_limb] / max(0.01, other_conf))
                        eval_engine.record_cross_model_transfer(
                            pattern, best_limb, other_limb,
                            success=r['correct'],
                        )

        # Run learning cycle to extract patterns
        cycle = eval_engine.compound_learning_cycle()
        compound_stats = {
            'total_patterns': cycle.get('total_patterns', 0),
            'transfer_efficiency': cycle.get('transfer_efficiency', 0),
            'cumulative_accuracy': cycle.get('cumulative_accuracy', 0),
            'learning_stats': eval_engine.get_learning_stats(),
        }

        # Find complementary limbs
        for limb in ['spatial', 'reasoning', 'perception']:
            complementary = eval_engine.find_complementary_models(limb)
            if complementary:
                compound_stats[f'{limb}_best_partners'] = dict(
                    sorted(complementary.items(), key=lambda x: x[1], reverse=True)[:3]
                )

        logger.info(f"Compound eval: {compound_stats.get('total_patterns', 0)} patterns, "
                     f"transfer_eff={compound_stats.get('transfer_efficiency', 0):.3f}")
    except Exception as e:
        logger.warning(f"Compound learning analysis skipped: {e}")

    # Analyze limb contributions
    limb_stats = {}
    for limb in ['perception', 'reasoning', 'planning', 'language',
                 'spatial', 'memory', 'metacognition', 'action']:
        correct_confs = [r['confidences'].get(limb, 0) for r in results if r['correct']]
        wrong_confs = [r['confidences'].get(limb, 0) for r in results if not r['correct']]
        limb_stats[limb] = {
            'avg_confidence_correct': sum(correct_confs) / len(correct_confs) if correct_confs else 0,
            'avg_confidence_wrong': sum(wrong_confs) / len(wrong_confs) if wrong_confs else 0,
        }

    # Cell accuracy stats
    cell_accs = [r['cell_accuracy'] for r in results]
    avg_cell_acc = sum(cell_accs) / len(cell_accs) if cell_accs else 0

    summary = {
        'total_tasks': total,
        'correct': correct,
        'accuracy': accuracy,
        'accuracy_pct': f"{accuracy*100:.1f}%",
        'avg_cell_accuracy': avg_cell_acc,
        'limb_analysis': limb_stats,
        'compound_learning': compound_stats,
        'results': results,
    }

    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate OctoTetrahedral on ARC-AGI")
    parser.add_argument("--config", type=str, default="default", choices=["default", "7b", "70b", "1.72t"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="evaluation", choices=["training", "evaluation"])
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--max-gen-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-attempts", type=int, default=3, help="Attempts per task (pass@k)")
    parser.add_argument("--ttt", action="store_true", help="Enable test-time training (fine-tune on each task's train examples)")
    parser.add_argument("--ttt-steps", type=int, default=10, help="TTT gradient steps per task")
    parser.add_argument("--ttt-lr", type=float, default=1e-5, help="TTT learning rate")
    parser.add_argument("--gen-mode", type=str, default="autoregressive",
                        choices=["autoregressive", "diffusion"], help="Generation mode")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    import tiktoken
    from train_distributed import load_config
    from model import OctoTetrahedralModel

    # Load checkpoint first to detect config
    logger.info(f"Loading checkpoint from {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    # Determine config: checkpoint may store a string name or a dict
    config_name = args.config
    if 'config' in ckpt:
        saved = ckpt['config']
        if isinstance(saved, str) and config_name == "default":
            config_name = saved
        elif isinstance(saved, dict) and config_name == "default":
            config_name = saved.get('name', config_name)

    config = load_config(config_name)
    device = args.device or config.device
    config.device = device

    # Apply saved dict overrides if checkpoint stored a config dict
    if 'config' in ckpt and isinstance(ckpt['config'], dict):
        saved = ckpt['config']
        if 'model' in saved:
            for k, v in saved['model'].items():
                if hasattr(config.model, k):
                    setattr(config.model, k, v)
        if 'moe' in saved:
            for k, v in saved['moe'].items():
                if hasattr(config.moe, k):
                    setattr(config.moe, k, v)

    config.device = device
    logger.info(f"Building {config_name} model...")
    model = OctoTetrahedralModel(config, use_geometric_physics=False)

    # Filter state dict to skip size mismatches
    model_state = model.state_dict()
    pretrained = ckpt['model_state_dict']
    filtered = {}
    skipped = 0
    for k, v in pretrained.items():
        if k in model_state and model_state[k].shape == v.shape:
            filtered[k] = v
        elif k in model_state:
            logger.warning(f"Skipping {k}: shape mismatch {v.shape} vs {model_state[k].shape}")
            skipped += 1
    model.load_state_dict(filtered, strict=False)
    logger.info(f"Loaded {len(filtered)}/{len(model_state)} params ({skipped} skipped)")

    # Cast to bf16 if checkpoint was trained in bf16 (saves memory on eval too)
    if device != "cpu" and any(v.dtype == torch.bfloat16 for v in pretrained.values()):
        model = model.to(torch.bfloat16)
        logger.info("Model cast to bfloat16 (matching checkpoint)")

    model = model.to(device)
    model.eval()

    total = model.get_num_params()
    active = model.get_active_params()
    logger.info(f"Model: {total/1e9:.2f}B total, {active/1e9:.2f}B active")
    if 'step' in ckpt:
        logger.info(f"Checkpoint from step {ckpt['step']}"
                     + (f", val_loss={ckpt['val_loss']:.4f}" if 'val_loss' in ckpt else ""))

    tokenizer = tiktoken.get_encoding("cl100k_base")

    # Load ARC tasks — search multiple locations (prefer AGI-2)
    arc_candidates = [
        Path.home() / "ARC_AMD_TRANSFER" / "data" / "ARC-AGI-2" / "data",
        Path.home() / "ARC_AMD_TRANSFER" / "data" / "ARC-AGI" / "data",
        Path.cwd() / "ARC_AMD_TRANSFER" / "data" / "ARC-AGI-2" / "data",
        Path.cwd() / "ARC_AMD_TRANSFER" / "data" / "ARC-AGI" / "data",
        Path.cwd() / "data" / "ARC-AGI" / "data",
    ]
    task_dir = None
    for arc_dir in arc_candidates:
        candidate = arc_dir / args.split
        if candidate.exists():
            task_dir = candidate
            break

    if task_dir is None:
        logger.error(f"ARC {args.split} data not found. Searched:")
        for c in arc_candidates:
            logger.error(f"  {c / args.split}")
        return

    tasks = {}
    for json_file in sorted(task_dir.glob("*.json")):
        with open(json_file) as f:
            tasks[json_file.stem] = json.load(f)
    logger.info(f"Loaded {len(tasks)} {args.split} tasks")

    # Evaluate
    t0 = time.time()
    summary = evaluate_model(
        model, tokenizer, tasks, config, device,
        max_tasks=args.max_tasks,
        max_gen_tokens=args.max_gen_tokens,
        temperature=args.temperature,
        num_attempts=args.num_attempts,
        ttt_enabled=args.ttt,
        ttt_steps=args.ttt_steps,
        ttt_lr=args.ttt_lr,
        gen_mode=args.gen_mode,
    )
    elapsed = time.time() - t0

    # Print results
    print("\n" + "=" * 60)
    print("ARC-AGI EVALUATION RESULTS")
    print("=" * 60)
    print(f"Model:           {args.config} ({total/1e9:.2f}B params)")
    print(f"Checkpoint:      {args.checkpoint}")
    print(f"Split:           {args.split}")
    print(f"Tasks evaluated: {summary['total_tasks']}")
    print(f"Correct:         {summary['correct']}")
    print(f"Accuracy:        {summary['accuracy_pct']}")
    print(f"Avg cell acc:    {summary['avg_cell_accuracy']:.3f}")
    print(f"Time:            {elapsed:.1f}s")
    print()

    # Limb analysis
    print("Limb Confidence Analysis (correct vs wrong):")
    for limb, stats in summary['limb_analysis'].items():
        c = stats['avg_confidence_correct']
        w = stats['avg_confidence_wrong']
        delta = c - w
        indicator = "↑" if delta > 0.05 else "↓" if delta < -0.05 else "→"
        print(f"  {limb:15s}  correct={c:.3f}  wrong={w:.3f}  {indicator}")

    # AGI assessment
    print()
    acc = summary['accuracy'] * 100
    if acc >= 85:
        print("🏆 ARC-AGI PRIZE THRESHOLD MET (≥85%)")
    elif acc >= 50:
        print("🔥 Competitive performance (≥50%). Getting closer to AGI.")
    elif acc >= 20:
        print("📈 Above baseline. Room for improvement.")
    else:
        print("🔧 Below baseline. Needs more training or architecture tuning.")

    # Save results
    output_path = args.output or f"arc_eval_{args.split}_{args.config}_{int(time.time())}.json"
    summary['config'] = args.config
    summary['checkpoint'] = args.checkpoint
    summary['split'] = args.split
    summary['elapsed_seconds'] = elapsed
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
