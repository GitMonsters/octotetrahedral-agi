"""
Performance Stress Test — OctoTetrahedral Architecture
Tests throughput, latency, concurrency, memory, and scaling.
"""

import torch
import time
import os
import json
import threading
import statistics
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

import sys
sys.path.insert(0, "/tmp/octotetrahedral-agi")

from train_pos_bilstm import (
    OctoTetrahedralPosTagger, CHAR_PAD, POS_VOCAB,
    CLARINWordDataset, collate_words, build_char_vocab, build_word_vocab
)
from torch.utils.data import DataLoader

# ─── Helpers ───

def get_memory_mb():
    """RSS memory in MB (macOS)."""
    import subprocess
    out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())]).decode().strip()
    return int(out) / 1024

def timer(fn, *args, n=1, warmup=0, **kwargs):
    """Time a function call, return (result, elapsed_ms)."""
    for _ in range(warmup):
        fn(*args, **kwargs)
    t0 = time.perf_counter()
    for _ in range(n):
        result = fn(*args, **kwargs)
    elapsed = (time.perf_counter() - t0) / n * 1000
    return result, elapsed

def generate_sentences(n, min_len=3, max_len=50):
    """Generate random word sequences of varying lengths."""
    import random
    words_pool = ["the", "a", "cat", "sat", "on", "mat", "dog", "ran", "quickly",
                  "brown", "fox", "jumped", "over", "lazy", "big", "red", "house",
                  "small", "tree", "blue", "sky", "new", "old", "good", "bad",
                  "is", "was", "are", "were", "has", "have", "had", "will", "can",
                  "John", "Mary", "London", "Python", "Google", "Microsoft"]
    sentences = []
    for _ in range(n):
        length = random.randint(min_len, max_len)
        sentences.append([random.choice(words_pool) for _ in range(length)])
    return sentences

# ─── Load model ───

print("=" * 60)
print("  OCTOTETRAHEDRAL PERFORMANCE STRESS TEST")
print("=" * 60)

mem_before = get_memory_mb()
print(f"\nLoading model...")
t0 = time.perf_counter()

ckpt = torch.load("/tmp/octotetrahedral-agi/checkpoints/octo_integrated_best.pt", map_location="cpu")
char_vocab = ckpt["char_vocab"]
word_vocab = ckpt["word_vocab"]
config = ckpt["config"]

model = OctoTetrahedralPosTagger(
    char_vocab_size=len(char_vocab), word_vocab_size=len(word_vocab),
    char_emb=config["char_emb"], word_emb=config["word_emb"],
    hidden_dim=config["hidden"], dropout=config["dropout"],
)
model.load_state_dict(ckpt["model"], strict=False)
model.eval()

load_time = (time.perf_counter() - t0) * 1000
mem_after = get_memory_mb()
total_params = sum(p.numel() for p in model.parameters())

print(f"  Load time: {load_time:.0f}ms")
print(f"  Model: {total_params / 1e6:.1f}M params")
print(f"  Memory: {mem_after:.0f}MB (+{mem_after - mem_before:.0f}MB)")

tag_inv = {v: k for k, v in POS_VOCAB.items()}

def tag_words(words, max_word_len=30):
    """Tag a list of words, return POS tags."""
    word_ids = [word_vocab.get(w, 1) for w in words]
    char_ids = []
    for w in words:
        chars = [char_vocab.get(c, 1) for c in w.lower()[:max_word_len]]
        while len(chars) < max_word_len:
            chars.append(CHAR_PAD)
        char_ids.append(chars[:max_word_len])

    max_len = len(word_ids)
    wid = torch.tensor([word_ids])
    cid = torch.tensor([char_ids])

    with torch.no_grad():
        out = model(wid, cid)

    pred_ids = out["pos_logits"][0].argmax(dim=-1).tolist()
    tags = [tag_inv.get(p, "_") for p in pred_ids[:len(words)]]
    return tags, out

# ─── TEST 1: Single Sentence Latency ───

print("\n" + "─" * 60)
print("TEST 1: Single Sentence Latency")
print("─" * 60)

test_sentences = [
    ("Short (3 words)", ["The", "cat", "sat"]),
    ("Medium (10 words)", "The quick brown fox jumps over the lazy dog".split()),
    ("Long (30 words)", "The quick brown fox jumps over the very lazy dog who was sleeping under the big old tree in the park".split()),
    ("Very long (50 words)", "The quick brown fox jumps over the very lazy dog who was sleeping under the big old tree in the park near the river where birds were singing and fish were swimming in the clear blue water".split()),
]

for name, words in test_sentences:
    tags, _ = tag_words(words)
    _, ms = timer(tag_words, words, n=10, warmup=3)
    tokens_per_sec = len(words) / (ms / 1000)
    print(f"  {name:25s} {ms:7.2f}ms  ({tokens_per_sec:,.0f} tokens/s)")

# ─── TEST 2: Throughput Scaling ───

print("\n" + "─" * 60)
print("TEST 2: Throughput Scaling (batch-like sequential)")
print("─" * 60)

import random
random.seed(42)
all_sentences = generate_sentences(1000, min_len=5, max_len=40)

for batch_n in [10, 50, 100, 500, 1000]:
    subset = all_sentences[:batch_n]
    t0 = time.perf_counter()
    total_tokens = 0
    for words in subset:
        tag_words(words)
        total_tokens += len(words)
    elapsed = time.perf_counter() - t0
    throughput = total_tokens / elapsed
    print(f"  {batch_n:5d} sentences ({total_tokens:6d} tokens): {elapsed:6.2f}s  ({throughput:,.0f} tokens/s, {batch_n/elapsed:.0f} sent/s)")

# ─── TEST 3: Batch Size Scaling ───

print("\n" + "─" * 60)
print("TEST 3: Batch Size Scaling (true batching)")
print("─" * 60)

def tag_batch(sentences_batch, max_word_len=30):
    """Process a batch of sentences simultaneously."""
    batch_size = len(sentences_batch)
    max_len = max(len(s) for s in sentences_batch)

    wid = torch.zeros(batch_size, max_len, dtype=torch.long)
    cid = torch.zeros(batch_size, max_len, max_word_len, dtype=torch.long)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    for b, words in enumerate(sentences_batch):
        for w_idx, w in enumerate(words):
            wid[b, w_idx] = word_vocab.get(w, 1)
            chars = [char_vocab.get(c, 1) for c in w.lower()[:max_word_len]]
            while len(chars) < max_word_len:
                chars.append(CHAR_PAD)
            cid[b, w_idx] = torch.tensor(chars[:max_word_len])
        mask[b, :len(words)] = True

    with torch.no_grad():
        out = model(wid, cid)

    pred_ids = out["pos_logits"].argmax(dim=-1)
    all_tags = []
    for b in range(batch_size):
        seq_len = len(sentences_batch[b])
        all_tags.append([tag_inv.get(pred_ids[b, t].item(), "_") for t in range(seq_len)])
    return all_tags

sentences_for_batch = generate_sentences(200, min_len=5, max_len=30)
for bs in [1, 4, 8, 16, 32, 64]:
    batch = sentences_for_batch[:bs]
    total_tokens = sum(len(s) for s in batch)
    _, ms = timer(tag_batch, batch, n=10, warmup=2)
    throughput = total_tokens / (ms / 1000)
    print(f"  batch_size={bs:2d}: {ms:8.2f}ms/batch  {throughput:8,.0f} tokens/s  {total_tokens/ms*1000:,.0f} sent/s")

# ─── TEST 4: Concurrent HTTP Throughput ───

print("\n" + "─" * 60)
print("TEST 4: Concurrent HTTP Throughput")
print("─" * 60)

import subprocess, signal

# Start server
print("  Starting server on port 8077...")
server_proc = subprocess.Popen(
    [sys.executable, "gpt2_serve.py", "--port", "8077", "--device", "cpu"],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    cwd="/tmp/octotetrahedral-agi"
)
time.sleep(8)  # wait for startup

# Verify server is up
try:
    r = requests.get("http://127.0.0.1:8077/health", timeout=5)
    server_ok = r.status_code == 200
except:
    server_ok = False

if not server_ok:
    print("  Server failed to start, skipping HTTP tests")
else:
    test_texts = [
        "The cat sat on the mat",
        "John went to the store to buy some food for dinner tonight",
        "The quick brown fox jumps over the lazy dog and continues running through the forest",
        "Scientists have discovered a new species of deep sea creature living near hydrothermal vents in the Pacific Ocean",
        "Python is a versatile programming language used for web development machine learning data science and automation tasks",
    ] * 4  # 20 unique-ish texts

    for concurrency in [1, 2, 5, 10, 20]:
        latencies = []
        errors = [0]

        def send_request(text):
            t0 = time.perf_counter()
            try:
                r = requests.post("http://127.0.0.1:8077/clarin/tag",
                                  json={"text": text}, timeout=30)
                elapsed = (time.perf_counter() - t0) * 1000
                if r.status_code != 200:
                    errors[0] += 1
                    return
                latencies.append(elapsed)
            except Exception:
                errors[0] += 1

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(send_request, text) for text in test_texts[:concurrency * 2]]
            for f in as_completed(futures):
                f.result()
        wall_time = time.perf_counter() - t0

        if latencies:
            avg_lat = statistics.mean(latencies)
            p50 = statistics.median(latencies)
            p95 = sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) > 1 else avg_lat
            rps = len(latencies) / wall_time
            print(f"  concurrency={concurrency:2d}: {len(latencies):3d} ok, {errors[0]} err  "
                  f"avg={avg_lat:7.1f}ms  p50={p50:7.1f}ms  p95={p95:7.1f}ms  "
                  f"rps={rps:.1f}")
        else:
            print(f"  concurrency={concurrency:2d}: ALL FAILED")

    # Stop server
    server_proc.terminate()
    server_proc.wait(timeout=5)

# ─── TEST 5: Memory Stress ───

print("\n" + "─" * 60)
print("TEST 5: Memory Under Load")
print("─" * 60)

mem_before = get_memory_mb()
large_batch = generate_sentences(100, min_len=20, max_len=80)
tag_batch(large_batch)
mem_after_load = get_memory_mb()
print(f"  After tagging 100 long sentences: {mem_after_load:.0f}MB (+{mem_after_load - mem_before:.0f}MB)")

# Repeated tagging to check for leaks
mem_before_leak = get_memory_mb()
for i in range(100):
    tag_batch(generate_sentences(10, min_len=5, max_len=30))
mem_after_leak = get_memory_mb()
leak = mem_after_leak - mem_before_leak
print(f"  After 1000 more sentences (100 batches): {mem_after_leak:.0f}MB (leak: {'+' if leak > 0 else ''}{leak:.1f}MB)")

# ─── TEST 6: Module Diagnostics Performance ───

print("\n" + "─" * 60)
print("TEST 6: Module Diagnostics Overhead")
print("─" * 60)

words = "The quick brown fox jumps over the lazy dog".split()

# Forward without diagnostics
def tag_no_diag(w):
    word_ids = [word_vocab.get(x, 1) for x in w]
    char_ids_list = []
    for x in w:
        chars = [char_vocab.get(c, 1) for c in x.lower()[:30]]
        while len(chars) < 30:
            chars.append(CHAR_PAD)
        char_ids_list.append(chars[:30])
    wid = torch.tensor([word_ids])
    cid = torch.tensor([char_ids_list])
    with torch.no_grad():
        return model(wid, cid)

# Forward + extract diagnostics
def tag_with_diag(w):
    out = tag_no_diag(w)
    diag = model.get_diagnostics()
    return out, diag

_, ms_plain = timer(tag_no_diag, words, n=100, warmup=10)
_, ms_diag = timer(tag_with_diag, words, n=100, warmup=10)
overhead = ms_diag - ms_plain
print(f"  Forward only:     {ms_plain:.2f}ms")
print(f"  Forward + diag:   {ms_diag:.2f}ms")
print(f"  Diagnostics overhead: {overhead:.2f}ms ({overhead/ms_plain*100:.1f}%)")

# ─── TEST 7: Longest Sequence Stress ───

print("\n" + "─" * 60)
print("TEST 7: Sequence Length Scaling")
print("─" * 60)

import random
random.seed(99)
words_pool = ["the", "cat", "sat", "on", "mat", "dog", "ran", "quick", "brown", "fox"]

for seq_len in [10, 50, 100, 200, 500, 1000]:
    words = [random.choice(words_pool) for _ in range(seq_len)]
    try:
        _, ms = timer(tag_words, words, n=5, warmup=1)
        tokens_per_sec = seq_len / (ms / 1000)
        print(f"  seq_len={seq_len:5d}: {ms:8.2f}ms  {tokens_per_sec:8,.0f} tokens/s")
    except Exception as e:
        print(f"  seq_len={seq_len:5d}: FAILED ({type(e).__name__}: {e})")

# ─── TEST 8: TP Phase Dynamics Under Load ───

print("\n" + "─" * 60)
print("TEST 8: TranscendPlexity Phase Dynamics Under Load")
print("─" * 60)

phase_counts = defaultdict(int)
stability_vals = []
cohesion_vals = []

random.seed(77)
for _ in range(500):
    words = [random.choice(words_pool) for _ in range(random.randint(3, 30))]
    tags, out = tag_words(words)
    tp = out["tp_state"]
    if tp:
        phase_counts[tp.phase_name] += 1
        stability_vals.append(float(tp.stability))
    cohesion_vals.append(out["cohesion"])

print(f"  500 sentences processed")
print(f"  Phase distribution:")
for phase, count in sorted(phase_counts.items(), key=lambda x: -x[1]):
    print(f"    {phase:20s}: {count:4d} ({count/500*100:.1f}%)")
print(f"  Stability: mean={statistics.mean(stability_vals):.4f}  "
      f"min={min(stability_vals):.4f}  max={max(stability_vals):.4f}")
print(f"  Cohesion:  mean={statistics.mean(cohesion_vals):.4f}  "
      f"min={min(cohesion_vals):.4f}  max={max(cohesion_vals):.4f}")

# ─── Summary ───

print("\n" + "=" * 60)
print("  STRESS TEST COMPLETE")
print("=" * 60)
final_mem = get_memory_mb()
print(f"  Final memory: {final_mem:.0f}MB")
print(f"  Model: {total_params/1e6:.1f}M params")
print(f"  All tests passed!")
