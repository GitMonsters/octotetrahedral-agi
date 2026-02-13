"""
OctoTetrahedral AGI - ARC Evaluation Script
Test the model on ARC tasks and show example generations.
"""

import torch
import torch.nn.functional as F
from pathlib import Path
import json
import random

try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False

from config import get_config
from model import OctoTetrahedralModel
from data.arc_dataset import (
    ARCDataset,
    ARCTask,
    create_arc_dataloader,
    evaluate_arc_prediction,
    grid_to_tokens,
    tokens_to_grid
)


def get_tokenizer():
    if HAS_TIKTOKEN:
        import tiktoken as _tiktoken
        return _tiktoken.get_encoding("cl100k_base")
    else:
        class SimpleTokenizer:
            def encode(self, text):
                return [ord(c) % 1000 for c in text]
            def decode(self, tokens):
                return ''.join(chr(t % 256) for t in tokens)
        return SimpleTokenizer()


def generate(
    model,
    tokenizer,
    input_text,
    max_new_tokens=100,
    temperature=0.3,
    top_k=50,
    greedy=False,
    debug_tokens=0,
    device='mps'
):
    """Generate output from input text"""
    model.eval()
    
    input_tokens = tokenizer.encode(input_text)
    input_ids = torch.tensor([input_tokens]).to(device)
    
    generated = input_ids.clone()
    
    stop_reason = "max_new_tokens"
    debug_decoded = []

    with torch.no_grad():
        for step in range(max_new_tokens):
            # Truncate if needed
            if generated.shape[1] > 512:
                context = generated[:, -512:]
            else:
                context = generated
            
            output = model(input_ids=context)
            logits = output['logits'][:, -1, :]

            if temperature and temperature > 0:
                logits = logits / temperature

            if greedy:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                # Top-k sampling
                if top_k and top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float('-inf')

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            
            generated = torch.cat([generated, next_token], dim=1)
            
            decoded = tokenizer.decode([next_token.item()])
            if debug_tokens and step < debug_tokens:
                debug_decoded.append(decoded)

            # Stop at ] which marks end of grid
            if ']' in decoded:
                stop_reason = "saw_close_bracket"
                break
    
    # Return only generated part
    generated_text = tokenizer.decode(generated[0, len(input_tokens):].tolist())
    return generated_text, {
        'stop_reason': stop_reason,
        'debug_decoded': debug_decoded,
        'input_tokens': len(input_tokens),
        'total_tokens': generated.shape[1]
    }


def print_grid(grid, title="Grid"):
    """Pretty print a grid"""
    print(f"\n{title}:")
    for row in grid:
        print("  " + " ".join(str(x) for x in row))


def evaluate_on_task(
    model,
    tokenizer,
    task,
    *,
    max_new_tokens=256,
    temperature=0.3,
    top_k=50,
    greedy=False,
    debug_tokens=0,
    device='mps'
):
    """Evaluate model on a single ARC task"""
    print(f"\n{'='*60}")
    print(f"Task: {task.task_id}")
    print(f"Train examples: {task.num_train}, Test examples: {task.num_test}")
    print('='*60)
    
    # Show training examples
    for i in range(task.num_train):
        inp, out = task.get_train_pair(i)
        print(f"\nExample {i+1}:")
        print_grid(inp, "Input")
        print_grid(out, "Output")
    
    results = []
    
    for test_idx in range(task.num_test):
        print(f"\n--- Test {test_idx + 1} ---")
        
        # Get input and target
        # NOTE: ARCTask.format_compact returns (input_text, target_text). The previous
        # code incorrectly treated the second value as the full concatenated text.
        input_text, _ = task.format_compact(test_idx=test_idx, include_answer=False)
        _, target_text = task.format_compact(test_idx=test_idx, include_answer=True)
        
        # Show test input
        test_input = task.get_test_input(test_idx)
        print_grid(test_input, "Test Input")
        
        # Generate prediction
        print("\nGenerating prediction...")
        generated, gen_info = generate(
            model,
            tokenizer,
            input_text,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            greedy=greedy,
            debug_tokens=debug_tokens,
            device=device,
        )
        
        if gen_info.get('debug_decoded'):
            dbg = ''.join(gen_info['debug_decoded'])
            print(f"\nFirst decoded tokens: {repr(dbg)}")
        print(f"Stop reason: {gen_info.get('stop_reason')}")
        print(f"\nGenerated: {repr(generated[:200])}...")
        print(f"Target:    {target_text[:100]}...")
        
        # Evaluate
        metrics = evaluate_arc_prediction(generated, target_text)
        
        # Try to parse and show generated grid
        try:
            pred_grid = tokens_to_grid(generated.strip().rstrip(']'))
            print_grid(pred_grid, "Predicted Output")
        except:
            print("\n(Could not parse generated grid)")
        
        # Show actual target
        target_output = task.get_test_output(test_idx)
        if target_output:
            print_grid(target_output, "Actual Output")
        
        print(f"\nMetrics:")
        print(f"  Exact match: {metrics['exact_match']:.0%}")
        print(f"  Grid accuracy: {metrics['grid_accuracy']:.1%}")
        print(f"  Cell accuracy: {metrics['cell_accuracy']:.1%}")
        
        results.append(metrics)
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate OctoTetrahedral on ARC')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/arc/arc_final.pt',
                       help='Checkpoint to load')
    parser.add_argument('--data-dir', type=str,
                       default='/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data',
                       help='ARC data directory')
    parser.add_argument('--num-tasks', type=int, default=5,
                       help='Number of tasks to evaluate')
    parser.add_argument('--task-id', type=str, default=None,
                       help='Specific task ID to evaluate')
    parser.add_argument('--split', type=str, default='evaluation',
                       help='Data split (training/evaluation)')
    parser.add_argument('--max-new-tokens', type=int, default=256,
                       help='Max new tokens to generate per test')
    parser.add_argument('--temperature', type=float, default=0.3,
                       help='Sampling temperature (ignored with --greedy)')
    parser.add_argument('--top-k', type=int, default=50,
                       help='Top-k for sampling (ignored with --greedy)')
    parser.add_argument('--greedy', action='store_true',
                       help='Use greedy decoding (argmax)')
    parser.add_argument('--debug-tokens', type=int, default=0,
                       help='Decode and print first N generated tokens')
    args = parser.parse_args()
    
    # Config and device
    config = get_config()
    device = config.device
    
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    
    # Load model
    print("\nLoading model...")
    model = OctoTetrahedralModel(config)
    
    checkpoint_path = Path(args.checkpoint)
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from step {checkpoint.get('global_step', 'unknown')}")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        print("Using randomly initialized model")
    
    model.to(device)
    model.eval()
    
    print(f"Model parameters: {model.get_num_params():,}")
    
    # Tokenizer
    tokenizer = get_tokenizer()
    
    # Load tasks
    print(f"\nLoading ARC tasks from {args.data_dir}/{args.split}...")
    dataset = ARCDataset(
        data_dir=args.data_dir,
        split=args.split,
        tokenizer=None  # We'll handle tokenization manually
    )
    
    print(f"Loaded {len(dataset.tasks)} tasks")
    
    # Select tasks to evaluate
    if args.task_id:
        tasks = [t for t in dataset.tasks if t.task_id == args.task_id]
        if not tasks:
            print(f"Task {args.task_id} not found!")
            return
    else:
        # Random sample
        tasks = random.sample(dataset.tasks, min(args.num_tasks, len(dataset.tasks)))
    
    # Evaluate
    all_results = []
    for task in tasks:
        results = evaluate_on_task(
            model,
            tokenizer,
            task,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            greedy=args.greedy,
            debug_tokens=args.debug_tokens,
            device=device,
        )
        all_results.extend(results)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if all_results:
        avg_exact = sum(r['exact_match'] for r in all_results) / len(all_results)
        avg_grid = sum(r['grid_accuracy'] for r in all_results) / len(all_results)
        avg_cell = sum(r['cell_accuracy'] for r in all_results) / len(all_results)
        
        print(f"Tasks evaluated: {len(tasks)}")
        print(f"Test cases: {len(all_results)}")
        print(f"Average exact match: {avg_exact:.1%}")
        print(f"Average grid accuracy: {avg_grid:.1%}")
        print(f"Average cell accuracy: {avg_cell:.1%}")


if __name__ == "__main__":
    main()
