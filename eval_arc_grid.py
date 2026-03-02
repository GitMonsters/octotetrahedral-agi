"""
ARC Grid-Level Evaluation for OctoTetrahedral AGI

Evaluates the grid prediction head on ARC tasks:
- Exact match (full grid correct)
- Cell accuracy (per-cell color match)
- Dimension accuracy (correct height × width)

Supports test-time training (TTT) for adaptation on each task.
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import tiktoken
import torch
import torch.nn.functional as F

from config import Config
from model import OctoTetrahedralModel
from grid_head import GridPredictionHead, grid_loss, predict_grid, MAX_GRID_SIZE
from data.arc_dataset import grid_to_tokens

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def encode_context(train_examples: List[Dict], test_input: List[List[int]],
                   tokenizer, max_seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode ARC task context as token sequence."""
    parts = []
    for ex in train_examples:
        inp = grid_to_tokens(ex['input'])
        out = grid_to_tokens(ex['output'])
        parts.append(f"[{inp}] -> [{out}]")
    test_inp = grid_to_tokens(test_input)
    parts.append(f"[{test_inp}] ->")
    text = " ".join(parts)

    tokens = tokenizer.encode(text)
    if len(tokens) > max_seq_len:
        tokens = tokens[-max_seq_len:]

    input_ids = torch.zeros(max_seq_len, dtype=torch.long)
    attn_mask = torch.zeros(max_seq_len, dtype=torch.long)
    input_ids[:len(tokens)] = torch.tensor(tokens)
    attn_mask[:len(tokens)] = 1
    return input_ids, attn_mask


def grid_to_tensor(grid: List[List[int]], max_grid: int = MAX_GRID_SIZE):
    """Convert grid list to padded tensor + mask."""
    h = len(grid)
    w = len(grid[0]) if h > 0 else 0
    t = torch.zeros(max_grid, max_grid, dtype=torch.long)
    m = torch.zeros(max_grid, max_grid, dtype=torch.bool)
    for i in range(h):
        for j in range(w):
            t[i, j] = grid[i][j]
            m[i, j] = True
    return t, m, h, w


def ttt_adapt(model, grid_head, train_examples: List[Dict],
              tokenizer, max_seq_len: int, device: torch.device,
              ttt_steps: int = 5, ttt_lr: float = 1e-4):
    """Test-time training: fine-tune grid head on task's training examples."""
    grid_head.train()
    optimizer = torch.optim.Adam(grid_head.parameters(), lr=ttt_lr)

    for step in range(ttt_steps):
        total_loss = 0
        for i, ex in enumerate(train_examples):
            # Use all other examples as context, predict this example's output
            context_exs = train_examples[:i] + train_examples[i+1:]
            if not context_exs:
                context_exs = train_examples  # fallback: use all including self

            input_ids, attn_mask = encode_context(
                context_exs, ex['input'], tokenizer, max_seq_len)
            input_ids = input_ids.unsqueeze(0).to(device)
            attn_mask = attn_mask.unsqueeze(0).to(device)

            target, mask, h, w = grid_to_tensor(ex['output'])
            target = target.unsqueeze(0).to(device)
            mask = mask.unsqueeze(0).to(device)
            target_h = torch.tensor([h], device=device)
            target_w = torch.tensor([w], device=device)

            with torch.no_grad():
                output = model(input_ids=input_ids, attention_mask=attn_mask)
            hidden = output['hidden_states'].detach()

            pred = grid_head(hidden)
            losses = grid_loss(pred, target, target_h, target_w, mask)
            total_loss += losses['total']

        avg_loss = total_loss / len(train_examples)
        avg_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    grid_head.eval()


def evaluate_task(model, grid_head, task: Dict, tokenizer,
                  max_seq_len: int, device: torch.device,
                  num_attempts: int = 1) -> Dict:
    """Evaluate a single ARC task.

    Returns dict with exact_match, cell_accuracy, dim_correct, etc.
    """
    train_examples = task['train']
    results = []

    for test_ex in task['test']:
        test_input = test_ex['input']
        test_output = test_ex['output']
        target, mask, target_h, target_w = grid_to_tensor(test_output)

        input_ids, attn_mask = encode_context(
            train_examples, test_input, tokenizer, max_seq_len)
        input_ids = input_ids.unsqueeze(0).to(device)
        attn_mask = attn_mask.unsqueeze(0).to(device)

        best_cell_acc = -1.0
        best_match = False
        best_pred = []
        best_dims = (0, 0)
        best_dim_correct = False

        for attempt in range(num_attempts):
            with torch.no_grad():
                output = model(input_ids=input_ids, attention_mask=attn_mask)
                hidden = output['hidden_states']
                pred = grid_head(hidden)

            pred_grid, pred_h, pred_w = predict_grid(pred)
            pred_grid = pred_grid.cpu()

            # Dimension accuracy
            dim_correct = (pred_h == target_h) and (pred_w == target_w)

            # Cell accuracy (on overlapping region)
            min_h = min(pred_h, target_h)
            min_w = min(pred_w, target_w)
            if min_h > 0 and min_w > 0:
                pred_crop = pred_grid[:min_h, :min_w]
                target_crop = target[:min_h, :min_w]
                matching = (pred_crop == target_crop).float().sum().item()
                total_cells = target_h * target_w
                cell_acc = matching / total_cells
            else:
                cell_acc = 0.0

            # Exact match
            exact = False
            if dim_correct:
                pred_out = pred_grid[:target_h, :target_w]
                target_out = target[:target_h, :target_w]
                exact = torch.equal(pred_out, target_out)

            if cell_acc > best_cell_acc:
                best_cell_acc = cell_acc
                best_match = exact
                best_pred = pred_grid[:pred_h, :pred_w].tolist()
                best_dims = (pred_h, pred_w)
                best_dim_correct = dim_correct

        results.append({
            'exact_match': best_match,
            'cell_accuracy': best_cell_acc,
            'dim_correct': best_dim_correct,
            'pred_dims': best_dims,
            'target_dims': (target_h, target_w),
            'pred_grid': best_pred,
        })

    return results


def main():
    parser = argparse.ArgumentParser(description='ARC Grid-Level Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to grid checkpoint (grid_best.pt or grid_step_N.pt)')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='ARC data directory (auto-detected if not set)')
    parser.add_argument('--split', type=str, default='evaluation',
                        help='ARC split to evaluate (training or evaluation)')
    parser.add_argument('--max-seq-len', type=int, default=2048)
    parser.add_argument('--num-attempts', type=int, default=1,
                        help='Number of prediction attempts per test example')
    parser.add_argument('--ttt', action='store_true',
                        help='Enable test-time training')
    parser.add_argument('--ttt-steps', type=int, default=5)
    parser.add_argument('--ttt-lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--max-tasks', type=int, default=None,
                        help='Limit number of tasks to evaluate')
    parser.add_argument('--output', type=str, default=None,
                        help='Save detailed results to JSON')
    args = parser.parse_args()

    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    logger.info(f"Device: {device}")

    # Find ARC data
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        for candidate in [
            Path.home() / "ARC_AMD_TRANSFER" / "data" / "ARC-AGI-2" / "data",
            Path.home() / "ARC_AMD_TRANSFER" / "data" / "ARC-AGI" / "data",
            Path("Prime-directive/arc-data/data"),
        ]:
            if (candidate / args.split).exists():
                data_dir = candidate
                break
        else:
            raise FileNotFoundError("Could not find ARC data directory")

    split_dir = data_dir / args.split
    task_files = sorted(split_dir.glob("*.json"))
    if args.max_tasks:
        task_files = task_files[:args.max_tasks]
    logger.info(f"Evaluating {len(task_files)} tasks from {split_dir}")

    # Load model
    cfg = Config()
    model = OctoTetrahedralModel(cfg).to(device)
    grid_head = GridPredictionHead(
        hidden_dim=cfg.model.hidden_dim,
        num_layers=3, num_heads=4, max_grid=MAX_GRID_SIZE
    ).to(device)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        logger.info("Loaded backbone from checkpoint")
    if 'grid_head_state_dict' in ckpt:
        grid_head.load_state_dict(ckpt['grid_head_state_dict'], strict=False)
        logger.info("Loaded grid head from checkpoint")
    else:
        logger.warning("No grid_head_state_dict in checkpoint — using random init!")

    model.eval()
    grid_head.eval()
    for p in model.parameters():
        p.requires_grad = False

    tokenizer = tiktoken.get_encoding('cl100k_base')

    # Evaluate
    total_exact = 0
    total_cell_acc = 0.0
    total_dim_correct = 0
    total_examples = 0
    task_results = {}

    start_time = time.time()
    for i, tf in enumerate(task_files):
        with open(tf) as f:
            task = json.load(f)

        task_id = tf.stem

        # Optional TTT
        if args.ttt:
            # Save grid head state for reset after TTT
            gh_state = {k: v.clone() for k, v in grid_head.state_dict().items()}
            ttt_adapt(model, grid_head, task['train'], tokenizer,
                      args.max_seq_len, device, args.ttt_steps, args.ttt_lr)

        results = evaluate_task(
            model, grid_head, task, tokenizer,
            args.max_seq_len, device, args.num_attempts)

        # Reset grid head after TTT
        if args.ttt:
            grid_head.load_state_dict(gh_state)
            grid_head.eval()

        task_exact = sum(r['exact_match'] for r in results)
        task_cell = sum(r['cell_accuracy'] for r in results) / len(results)
        task_dim = sum(r['dim_correct'] for r in results)

        total_exact += task_exact
        total_cell_acc += task_cell
        total_dim_correct += task_dim
        total_examples += len(results)

        status = "✅" if task_exact == len(results) else "❌"
        logger.info(f"[{i+1}/{len(task_files)}] {task_id}: {status} "
                    f"exact={task_exact}/{len(results)} "
                    f"cell_acc={task_cell:.1%} "
                    f"dim={task_dim}/{len(results)}")

        task_results[task_id] = {
            'exact_match': task_exact,
            'total_tests': len(results),
            'cell_accuracy': task_cell,
            'dim_correct': task_dim,
            'details': results,
        }

    elapsed = time.time() - start_time

    # Summary
    n_tasks = len(task_files)
    tasks_solved = sum(1 for r in task_results.values()
                       if r['exact_match'] == r['total_tests'])
    avg_cell = total_cell_acc / n_tasks if n_tasks > 0 else 0
    dim_rate = total_dim_correct / total_examples if total_examples > 0 else 0

    logger.info("=" * 60)
    logger.info(f"ARC Grid Evaluation Results ({args.split})")
    logger.info(f"  Tasks evaluated: {n_tasks}")
    logger.info(f"  Tasks solved (exact): {tasks_solved}/{n_tasks} ({tasks_solved/n_tasks:.1%})")
    logger.info(f"  Exact match examples: {total_exact}/{total_examples}")
    logger.info(f"  Avg cell accuracy: {avg_cell:.1%}")
    logger.info(f"  Dimension accuracy: {dim_rate:.1%}")
    logger.info(f"  Time: {elapsed:.1f}s ({elapsed/n_tasks:.1f}s/task)")
    logger.info(f"  TTT: {'enabled' if args.ttt else 'disabled'}")
    logger.info("=" * 60)

    # Save results
    if args.output:
        summary = {
            'split': args.split,
            'checkpoint': args.checkpoint,
            'n_tasks': n_tasks,
            'tasks_solved': tasks_solved,
            'solve_rate': tasks_solved / n_tasks if n_tasks > 0 else 0,
            'total_exact': total_exact,
            'total_examples': total_examples,
            'avg_cell_accuracy': avg_cell,
            'dim_accuracy': dim_rate,
            'elapsed_seconds': elapsed,
            'ttt_enabled': args.ttt,
            'ttt_steps': args.ttt_steps if args.ttt else 0,
            'tasks': task_results,
        }
        with open(args.output, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"Results saved to {args.output}")


if __name__ == '__main__':
    main()
